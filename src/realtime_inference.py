import dv_processing as dv
import time
import cv2
import rootutils

ROOTPATH = rootutils.setup_root(__file__, indicator=".project_root", pythonpath=True)

from src.data.events_encoding import accumulate_dv_encode
from src.data.preprocessing import preprocess_frames

# Open any camera
capture = dv.io.CameraCapture()

cv2.namedWindow("Resized_Window", cv2.WINDOW_NORMAL) 
cv2.resizeWindow("Resized_Window", 240*4, 180*4 ) 

# Run the loop while camera is still connected
while capture.isRunning():
    # Read batch of events
    events = capture.getNextEventBatch()

    # The method does not wait for data arrive, it returns immediately with
    # latest available data or if no data is available, returns a `None`
    if events is not None:
        # Print received packet time range
        # print(f"{events}")
        # Encode events into a frame
        frames = accumulate_dv_encode(events, capture.getEventResolution())
        # frames = preprocess_frames([frames])
        # print(frames.shape)
        # Display the frame live in a window at 10fps
        cv2.imshow("Resized_Window", frames)
    else:
        # No data has arrived yet, short sleep to reduce CPU load
        time.sleep(0.001)
    cv2.waitKey(1)  # Add delay to control frame rate
