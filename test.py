import cv2 as cv
import dv_processing as dv
import torch
from torchvision import transforms
from datetime import timedelta
import time
import rootutils
import os
from threading import Thread
from queue import Queue

ROOTPATH = rootutils.setup_root(__file__, indicator=".project_root", pythonpath=True)
CONFIG_PATH = str(ROOTPATH / "configs")

import src.models as models
from src.data.events_encoding import accumulate_dv_encode

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

capture = None
model = None
slicer = None
hidden_states = None
frame_queue = Queue()
running = True
source_type = "file"
no_press_counter = 0
reset_threshold = 30

idx_to_label = {0: "No press", 1: "Press"}

# FPS calculation
last_fps_time = 0
fps_counter = 0

def load_model(model_name):
    global model, hidden_states
    checkpoint_path = f"{ROOTPATH}/saved_models/{model_name}"

    model_type = model_name.split('_')[0].lower()
    if model_type == "convsnn":
        model = models.ConvSNN.load_from_checkpoint(checkpoint_path).to(device)
    elif model_type == "convsnnl":
        model = models.ConvSNN_L.load_from_checkpoint(checkpoint_path).to(device)
    elif model_type == "convslstm":
        model = models.SpikingConvLSTM.load_from_checkpoint(checkpoint_path).to(device)
    elif model_type == "convslstmcbam2":
        model = models.SpikingConvLSTM_CBAM.load_from_checkpoint(checkpoint_path, strict=False).to(device)
    else:
        raise Exception("Invalid model")

    model.to_inference_mode()
    hidden_states = model.init_hidden_states()

def load_data(data_name=None, source_type="file"):
    global capture, slicer
    raw_path = f'{ROOTPATH}/data/raw_data/{data_name}'
    capture = dv.io.MonoCameraRecording(raw_path)
    slicer = dv.EventStreamSlicer()
    slicer.doEveryTimeInterval(timedelta(milliseconds=33), slicing_callback)


transform = transforms.Compose([
    transforms.ToPILImage(),
    transforms.Resize((32, 32)),
    transforms.Grayscale(num_output_channels=1),
    transforms.ToTensor(),
    transforms.Normalize(mean=0.5, std=0.01),
])

def slicing_callback(events: dv.EventStore):
    if events.size() == 0:
        return
    frame = accumulate_dv_encode(events, capture.getEventResolution())
    frame_queue.put(frame)  # Không bỏ frame nào

def inference_loop(output_path="output.avi"):
    global hidden_states, no_press_counter, last_fps_time, fps_counter

    # Setup video writer
    fourcc = cv.VideoWriter_fourcc(*'XVID')
    out = cv.VideoWriter(output_path, fourcc, 30.0, (640, 480))

    while running or not frame_queue.empty():
        try:
            frame = frame_queue.get(timeout=1)

            # Resize for display
            frame_vis = cv.resize(frame, (640, 480))
            frame_input = transform(frame).unsqueeze(0).to(device)

            output, hidden_states = model.process_frame(frame_input, hidden_states)
            pred = output.argmax(dim=1).item()
            label = idx_to_label[pred]

            # Reset logic
            if pred == 0:
                no_press_counter += 1
                if no_press_counter >= reset_threshold:
                    hidden_states = model.init_hidden_states()
                    no_press_counter = 0
                    print("Resetting hidden states")
            else:
                no_press_counter = 0

            # Overlay prediction
            cv.putText(frame_vis, f"Prediction: {label}", (10, 30), cv.FONT_HERSHEY_SIMPLEX,
                       1, (0, 255, 0) if pred else (0, 0, 255), 2)

            # Write to video
            out.write(cv.cvtColor(frame_vis, cv.COLOR_GRAY2BGR) if len(frame_vis.shape) == 2 else frame_vis)

        except Exception as e:
            print(f"Error in inference: {e}")
            continue

    out.release()
    cv.destroyAllWindows()

def event_processing_loop():
    while running and capture.isRunning():
        events = capture.getNextEventBatch()
        if events is not None:
            slicer.accept(events)
            if source_type == "file":
                duration = events.getHighestTime() - events.getLowestTime()
                time.sleep(duration / 1e6)

if __name__ == '__main__':
    model_name = "convslstmcbam2_acc.ckpt"  # Thay bằng tên model thực tế
    data_file = "raw2.aedat4"         # Thay bằng file dữ liệu thực tế
    load_model(model_name)
    load_data(data_file)

    t1 = Thread(target=event_processing_loop)
    t2 = Thread(target=inference_loop, args=("prediction_output3.avi",))
    t1.start()
    t2.start()

    t1.join()
    t2.join()
