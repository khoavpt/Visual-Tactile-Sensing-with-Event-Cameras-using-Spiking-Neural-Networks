import cv2 as cv
import dv_processing as dv
import torch
from torchvision import transforms
from flask import Flask, render_template, request, jsonify
from flask_socketio import SocketIO
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

app = Flask(__name__)
socketio = SocketIO(app, cors_allowed_origins="*")

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Globals
capture = None
model = None
slicer = None
hidden_states = None
frame_queue = Queue(maxsize=1)
running = False
source_type = "file"
no_press_counter = 0
reset_threshold = 30  # Số frame No Press liên tiếp trước khi reset

# FPS tracking
last_fps_time = 0
fps_counter = 0

idx_to_label = {0: "No press", 1: "Press"}

def load_model(model_name):
    global model, hidden_states
    checkpoint_path = f"{ROOTPATH}/saved_models/{model_name}"

    mtype = model_name.split('_')[0].lower()
    if mtype == "convsnn":
        model = models.ConvSNN.load_from_checkpoint(checkpoint_path, strict=False).to(device)
    elif mtype == "convsnnl":
        model = models.ConvSNN_L.load_from_checkpoint(checkpoint_path, strict=False).to(device)
    elif mtype == "convslstm":
        model = models.SpikingConvLSTM.load_from_checkpoint(checkpoint_path, strict=False).to(device)
    elif mtype == "convslstmcbamspike":
        model = models.SpikingConvLSTM_CBAM.load_from_checkpoint(checkpoint_path, strict=False).to(device)
    elif mtype == "convslstmcbam2":
        model = models.SpikingConvLSTM_CBAM.load_from_checkpoint(checkpoint_path, strict=False).to(device)
    elif mtype == "convslstmcbam3":
        model = models.SpikingConvLSTM_CBAM.load_from_checkpoint(checkpoint_path, strict=False).to(device)
    else:
        raise Exception("Invalid model type")

    model.to_inference_mode()
    hidden_states = model.init_hidden_states()

def load_data(data_name=None, source_type="file"):
    global capture, slicer
    if source_type == "live":
        capture = dv.io.CameraCapture()
    else:
        raw_path = f'{ROOTPATH}/data/raw_data/{data_name}'
        capture = dv.io.MonoCameraRecording(raw_path)
    slicer = dv.EventStreamSlicer()
    slicer.doEveryTimeInterval(timedelta(milliseconds=10), slicing_callback)

class ClipTransform:
    def __call__(self, img):
        return torch.clamp(img, 0, 5)

transform = transforms.Compose([
    transforms.ToPILImage(),
    transforms.Resize((32, 32)),
    transforms.Grayscale(num_output_channels=1),
    transforms.ToTensor(),
    transforms.Normalize(mean=0.5, std=0.01),
    ClipTransform(),
])

def slicing_callback(events: dv.EventStore):
    if events.size() == 0:
        return
    frame = accumulate_dv_encode(events, capture.getEventResolution())
    frame_vis = cv.resize(frame, (640, 480))
    _, buffer = cv.imencode('.jpg', frame_vis)
    socketio.emit('frame', {'image': buffer.tobytes()})
    try:
        frame_queue.put_nowait(frame)
    except:
        frame_queue.get_nowait()
        frame_queue.put_nowait(frame)

def inference_loop():
    global hidden_states, no_press_counter, last_fps_time, fps_counter
    while running:
        try:
            # t0 = time.perf_counter()
            frame = frame_queue.get(timeout=0.1)
            inp = transform(frame).unsqueeze(0).to(device)
            output, hidden_states = model.process_frame(inp, hidden_states)
            pred = output.argmax(dim=1).item()
            # t1 = time.perf_counter()
            # print(f"Processing time: {t1 - t0:.4f} seconds")

            # FPS
            now = time.time()
            fps_counter += 1
            if now - last_fps_time >= 1.0:
                socketio.emit('prediction', {
                    'label': idx_to_label[pred],
                    'fps': fps_counter
                })
                fps_counter = 0
                last_fps_time = now
            else:
                socketio.emit('prediction', {'label': idx_to_label[pred]})

            # Reset hidden nếu No Press quá threshold
            if pred == 0:
                no_press_counter += 1
                if no_press_counter >= reset_threshold:
                    hidden_states = model.init_hidden_states()
                    no_press_counter = 0
                    # print("Resetting hidden states")
            else:
                no_press_counter = 0

        except Exception:
            continue

def event_processing_loop():
    while running and capture.isRunning():
        events = capture.getNextEventBatch()
        if events is not None:
            slicer.accept(events)
            if source_type == "file":
                duration = events.getHighestTime() - events.getLowestTime()
                time.sleep(duration / 1e6)

@app.route('/')
def index():
    model_files = [f for f in os.listdir(f"{ROOTPATH}/saved_models") if f.endswith('.ckpt')]
    data_files = [f for f in os.listdir(f"{ROOTPATH}/data/raw_data") if f.endswith('.aedat4')]
    return render_template('index.html', models=model_files, data_files=data_files)

@app.route('/start_processing', methods=['POST'])
def start_processing():
    global source_type, running
    data = request.get_json()
    model_name = data.get('model')
    data_file = data.get('data')
    source_type = data.get('source_type', 'file')
    try:
        running = True
        load_model(model_name)
        load_data(data_file, source_type)
        Thread(target=event_processing_loop, daemon=True).start()
        Thread(target=inference_loop, daemon=True).start()
        return jsonify({"status": "success"})
    except Exception as e:
        return jsonify({"status": "error", "message": str(e)})

@app.route('/stop_processing', methods=['POST'])
def stop_processing():
    global running, capture, frame_queue, hidden_states, no_press_counter
    running = False
    # Dừng camera nếu đang chạy
    try:
        if capture is not None and capture.isRunning():
            capture.stop()
    except Exception:
        pass
    # Reset lại trạng thái
    frame_queue = Queue(maxsize=1)
    hidden_states = None
    no_press_counter = 0
    return jsonify({"status": "success"})

if __name__ == '__main__':
    socketio.run(app, host='0.0.0.0', port=8000, debug=False, allow_unsafe_werkzeug=True)
