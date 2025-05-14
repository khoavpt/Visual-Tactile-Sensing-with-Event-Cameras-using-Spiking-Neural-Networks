import cv2 as cv
import dv_processing as dv
import torch
from torchvision import transforms
from datetime import timedelta
import time
import rootutils
from threading import Thread
from queue import Queue
import tkinter as tk
from PIL import Image, ImageTk

ROOTPATH = rootutils.setup_root(__file__, indicator=".project_root", pythonpath=True)

import src.models as models
from src.models.slstm1d import SpikingConvLSTM1d
from src.data.events_encoding import accumulate_dv_encode

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Configs
frame_queue = Queue(maxsize=1)
running = True
reset_threshold = 30  # Number of consecutive "No Press" before reset
no_press_counter = 0
idx_to_label = {0: "No press", 1: "Press"}

model = None
hidden_states = None
capture = None
slicer = None
source_type = "file"

# GUI Elements
tk_root = None
tk_image_label = None
tk_prediction_var = None  # Using StringVar for dynamic text
tk_fps_var = None         # Using StringVar for dynamic text

# FPS calculation state for GUI loop
last_time_fps_calc = 0
frame_count_fps_calc = 0

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
    elif model_type == "convslstm1d":
        model = SpikingConvLSTM1d.load_from_checkpoint(checkpoint_path).to(device)
    elif model_type == "convslstmcbam2":
        model = models.SpikingConvLSTM_CBAM.load_from_checkpoint(checkpoint_path, strict=False).to(device)
    else:
        raise Exception("Invalid model")

    model.to_inference_mode()
    hidden_states = model.init_hidden_states()

    slicer = dv.EventStreamSlicer()
    slicer.doEveryTimeInterval(timedelta(milliseconds=33), slicing_callback)

def slicing_callback(events: dv.EventStore):
    if events.size() == 0:
        return
    frame = accumulate_dv_encode(events, capture.getEventResolution())
    try:
        frame_queue.put_nowait(frame)
    except:
        frame_queue.get_nowait()
        frame_queue.put_nowait(frame)

def update_gui():
    global hidden_states, no_press_counter, running
    global last_time_fps_calc, frame_count_fps_calc
    global tk_root, tk_image_label, tk_prediction_var, tk_fps_var, model, device, transform, idx_to_label, frame_queue

    if not running:
        return

    try:
        frame = frame_queue.get(timeout=0.05)  # Shorter timeout for GUI responsiveness
        frame_input = transform(frame).unsqueeze(0).to(device)
        output, hidden_states_updated = model.process_frame(frame_input, hidden_states)
        hidden_states = hidden_states_updated # Ensure hidden_states is updated
        pred = output.argmax(dim=1).item()
        # print(f"Prediction: {pred}")
        label = idx_to_label[pred]

        # FPS calculation
        frame_count_fps_calc += 1
        current_time = time.time()
        fps_display_val = "Calculating..."
        if current_time - last_time_fps_calc >= 1.0:
            fps_display_val = str(frame_count_fps_calc)
            frame_count_fps_calc = 0
            last_time_fps_calc = current_time
        elif tk_fps_var.get() != "Calculating..." and "FPS:" in tk_fps_var.get() : # Keep old FPS if not updating
             fps_display_val = tk_fps_var.get().split(":")[1].strip()


        # Update GUI labels
        tk_prediction_var.set(f"Prediction: {label}")
        if fps_display_val != "Calculating...": # only update if not default
            tk_fps_var.set(f"FPS: {fps_display_val}")


        # Prepare frame for display
        display_frame = cv.cvtColor(frame, cv.COLOR_GRAY2BGR) # Assuming frame is grayscale
        display_frame_resized = cv.resize(display_frame, (640, 480))

        # Convert to Tkinter format
        img = Image.fromarray(display_frame_resized)
        imgtk = ImageTk.PhotoImage(image=img)
        tk_image_label.imgtk = imgtk  # Keep a reference!
        tk_image_label.configure(image=imgtk)

        # Reset logic
        if pred == 0:
            no_press_counter += 1
            if no_press_counter >= reset_threshold:
                hidden_states = model.init_hidden_states()
                no_press_counter = 0
                print("Reset hidden states")
        else:
            no_press_counter = 0

    except Exception as e:
        # print(f"Error in update_gui: {e}") # Optional: for debugging
        pass # Continue if queue is empty or other minor error

    if running:
        tk_root.after(33, update_gui) # Schedule next update (matches slicer approx)

def start_gui_and_inference():
    global running, tk_root, tk_image_label, tk_prediction_var, tk_fps_var
    global last_time_fps_calc, frame_count_fps_calc

    running = True # Ensure running is true when starting

    tk_root = tk.Tk()
    tk_root.title("DV Prediction GUI")

    # Initialize FPS calculation timers
    last_time_fps_calc = time.time()
    frame_count_fps_calc = 0

    # Create and place GUI elements
    tk_image_label = tk.Label(tk_root)
    tk_image_label.pack(padx=10, pady=10)

    tk_prediction_var = tk.StringVar()
    tk_prediction_var.set("Prediction: Waiting for data...")
    prediction_label_display = tk.Label(tk_root, textvariable=tk_prediction_var, font=("Helvetica", 16))
    prediction_label_display.pack(pady=5)

    tk_fps_var = tk.StringVar()
    tk_fps_var.set("FPS: Calculating...")
    fps_label_display = tk.Label(tk_root, textvariable=tk_fps_var, font=("Helvetica", 14))
    fps_label_display.pack(pady=5)

    def on_quit():
        global running
        running = False
        if tk_root: # Check if root exists
             tk_root.destroy()

    quit_button = tk.Button(tk_root, text="Quit", command=on_quit, width=10)
    quit_button.pack(pady=10)

    tk_root.protocol("WM_DELETE_WINDOW", on_quit) # Handle window close button

    update_gui()  # Start the GUI update loop
    tk_root.mainloop()

def load_data(data_name=None, src_type="file"):
    global capture, slicer, source_type
    source_type = src_type
    if src_type == "live":
        capture = dv.io.CameraCapture()
    else:
        path = f"{ROOTPATH}/data/raw_data/{data_name}"
        capture = dv.io.MonoCameraRecording(path)

    slicer = dv.EventStreamSlicer()
    slicer.doEveryTimeInterval(timedelta(milliseconds=33), slicing_callback)

def event_processing_loop():
    while running and capture.isRunning():
        events = capture.getNextEventBatch()
        if events:
            slicer.accept(events)
            if source_type == "file":
                duration = events.getHighestTime() - events.getLowestTime()
                time.sleep(duration / 1e6)

def main(model_name, data_name=None, source="file"):
    load_model(model_name)
    load_data(data_name, source)
    Thread(target=event_processing_loop, daemon=True).start()
    start_gui_and_inference()  # Changed from inference_loop()

if __name__ == '__main__':
    # Example usage
    model_file = "convslstm1d_acc.ckpt"       # Replace with your model
    data_file = "raw2.aedat4"          # Replace with your AEDAT4 file (if source is "file")
    main(model_file, data_file, source="file")  # or use source="live" for camera