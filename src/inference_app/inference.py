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

from src.data.events_encoding import accumulate_dv_encode
from src.models.convsnn import ConvSNN
from src.models.convsnnl import ConvSNN_L
from src.models.spiking_convlstm import SpikingConvLSTM

app = Flask(__name__)
socketio = SocketIO(app, cors_allowed_origins="*")

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

capture = None
model = None
slicer = None
hidden_states = None
frame_buffer = None
idx_to_label = {0: "No press", 1: "Press"}
frame_queue = Queue(maxsize=1)
running = True
source_type = "file"
no_press_counter = 0
reset_threshold = 30  # Number of consecutive "No Press" frames before reset

def load_model(model_name):
    global model, hidden_states
    checkpoint_path = f"{ROOTPATH}/saved_models/{model_name}"

    model_type = model_name.split('_')[0].lower()
    if model_type == "convsnn":
        model = ConvSNN.load_from_checkpoint(checkpoint_path).to(device)
    elif model_type == "convsnnl":
        model = ConvSNN_L.load_from_checkpoint(checkpoint_path).to(device)
    elif model_type == "convslstm":
        model = SpikingConvLSTM.load_from_checkpoint(checkpoint_path).to(device)
    else:
        raise Exception("Invalid model")

    model.eval()
    hidden_states = model.init_hidden_states()


def load_data(data_name=None, source_type="file"):
    global capture, slicer
    try:
        if source_type == "live":
            capture = dv.io.CameraCapture()
        else:
            raw_path = f'{ROOTPATH}/data/raw_data/{data_name}'
            capture = dv.io.MonoCameraRecording(raw_path)
        
        slicer = dv.EventStreamSlicer()
        slicer.doEveryTimeInterval(timedelta(milliseconds=33), slicing_callback)
    except Exception as e:
        raise Exception(str(e))

class ClipTransform:
    def __call__(self, img):
        return torch.clamp(img, 0, 1)

transform = transforms.Compose([
    transforms.ToPILImage(),
    transforms.Resize((32, 32)),
    transforms.Grayscale(num_output_channels=1),
    transforms.ToTensor(),
    ClipTransform(),
    transforms.Normalize(mean=[0.5], std=[0.01])
])

def slicing_callback(events: dv.EventStore):
    """Adds frames to the queue for processing."""
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
    """Continuously runs inference on frames from the queue."""
    global frame_buffer, hidden_states, no_press_counter
    while running:
        try:
            frame = frame_queue.get(timeout=0.1)
            
            frame_input = transform(frame).unsqueeze(0).to(device)
            output, hidden_states = model.process_frame(frame_input, hidden_states)
            pred = output.argmax(dim=1).item()
            socketio.emit('prediction', {'label': idx_to_label[pred]})

            # Reset hidden states after consecutive "No Press" predictions
            if pred == 0:
                no_press_counter += 1
                if no_press_counter >= reset_threshold:
                    hidden_states = model.init_hidden_states()
                    no_press_counter = 0
                    print("Resetting hidden states")
            else:
                no_press_counter = 0  # Reset counter if "Press" is detected

        except:
            continue

def event_processing_loop():
    """Continuously fetches event batches."""
    while running and capture.isRunning():
        events = capture.getNextEventBatch()
        if events is not None:
            slicer.accept(events)

            # Sleep for duration of events if source = file
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
    global source_type
    data = request.get_json()
    model_name = data.get('model')
    data_file = data.get('data')
    source_type = data.get('source_type', 'file')
    
    try:
        load_model(model_name)
        load_data(data_file, source_type)
        
        Thread(target=event_processing_loop, daemon=True).start()
        Thread(target=inference_loop, daemon=True).start()
        
        return jsonify({"status": "success"})
    except Exception as e:
        return jsonify({"status": "error", "message": str(e)})

if __name__ == '__main__':
    socketio.run(app, host='0.0.0.0', port=5000, debug=False, allow_unsafe_werkzeug=True)
