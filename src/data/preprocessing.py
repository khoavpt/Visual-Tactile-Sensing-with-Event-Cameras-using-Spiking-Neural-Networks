import os
from datetime import timedelta
import dv_processing as dv
import torch
from torchvision import transforms
import rootutils
import pandas as pd
import numpy as np

ROOTPATH = rootutils.setup_root(__file__, indicator=".project_root", pythonpath=True)

import src.data.events_encoding as ev_enc

normalize_param = {
    'accumulate': (0.5, 0.01),
    'time_surface': (0.0, 0.03),
    'custom': (0.0, 0.003)
}


def dv_data_frame_tSlice(file_path, press_times_list, duration=10, encoding_type='accumulate' ):
    """
    Extract frames from a .aedat4 file using a time slice.
    Args:
        file_path: str
        duration (milliseconds): int
        press_times_list: list of time (every 2 elements are a pair of start time and end time)
        encoding_type: str ('accumulate', 'time_surface', 'custom')
    Returns:
        list[np.ndarray]: frames (each of shape (H, W))
        list[int]: labels (0 for no press and 1 for press)
    """
    capture = dv.io.MonoCameraRecording(file_path)
    
    frames = []
    labels = [] # 0 for no press and 1 for press
    if not capture.isEventStreamAvailable():
        raise RuntimeError("Input camera does not provide an event stream.")

    slicer = dv.EventStreamSlicer()
    if encoding_type == 'accumulate':
        encoding_func = ev_enc.accumulate_dv_encode
    elif encoding_type == 'time_surface':
        encoding_func = ev_enc.timesurface_dv_encode
    elif encoding_type == 'custom':
        encoding_func = ev_enc.custom_encode

    def slicing_callback(events: dv.EventStore):
        frame = encoding_func(events, capture.getEventResolution())
        frames.append(frame)
        
        if len(press_times_list) == 0:
            labels.append(0)
        else:
            event_start_time = events.getLowestTime()
            event_end_time = events.getHighestTime()
            label = 0
            for start, end in zip(press_times_list[::2], press_times_list[1::2]):
                if event_start_time <= end and event_end_time >= start: # A frame is considered as a press if it contains any press event
                    label = 1
                    break
            labels.append(label)

    slicer.doEveryTimeInterval(timedelta(milliseconds=duration), slicing_callback)

    while capture.isRunning():
        events = capture.getNextEventBatch()
        if events is not None:
            slicer.accept(events)
    return frames, labels

def preprocess_frames(frames, encoding_type='accumulate', target_size=(32, 32)):
    mean, std = normalize_param[encoding_type]

    # class MinMaxTransform:
    #     def __call__(self, img):
    #         min_val = img.min()
    #         max_val = img.max()
    #         if max_val > min_val:  # Avoid division by zero
    #             img = (img - min_val) / (max_val - min_val)
    #         return img
    
    class ClipTransform:
        def __call__(self, img):
            return torch.clamp(img, 0, 5)

    transform = transforms.Compose([
        transforms.ToPILImage(),
        transforms.Resize(target_size),  # Resize the images if needed
        transforms.Grayscale(num_output_channels=1),
        transforms.ToTensor(),         # Convert images to tensor
        transforms.Normalize(mean=mean, std=std),  # Normalize
        ClipTransform(),               # Clip values to be inside [0, 1]
        # MinMaxTransform()
    ])

    # class SplitChannelsTransform:
    #     def __call__(self, img):
    #         return img[0], img[1]

    # class MergeChannelsTransform:
    #     def __call__(self, imgs):
    #         return torch.stack(imgs)

    # class RemoveExtraChannelTransform:
    #     def __call__(self, img):
    #         return img.squeeze(0)  # Remove the extra channel dimension

    # transform = transforms.Compose([
    #     SplitChannelsTransform(),
    #     transforms.Lambda(lambda imgs: [transforms.ToPILImage()(img) for img in imgs]),
    #     transforms.Lambda(lambda imgs: [transforms.Resize(target_size)(img) for img in imgs]),
    #     transforms.Lambda(lambda imgs: [transforms.ToTensor()(img) for img in imgs]),
    #     transforms.Lambda(lambda imgs: [ClipTransform()(img) for img in imgs]),
    #     # transforms.Lambda(lambda imgs: [transforms.Normalize(mean=[mean], std=[std])(img) for img in imgs]),
    #     transforms.Lambda(lambda imgs: [RemoveExtraChannelTransform()(img) for img in imgs]),
    #     MergeChannelsTransform()
    # ])
    
    preprocessed_frames = [transform(frame) for frame in frames] # List of tensors, each of shape (2, H, W)
    return preprocessed_frames

def create_sequence(frames, labels, sequence_length=300, steps=100):
    """
    Split a list of frames into sequences of a fixed length using the sliding window approach, each with their corresponding labels.
    Args:
        frames: list of frames
        labels: list of labels
        sequence_length: int
        step: int, step size for the sliding window
    Returns:
        list of sequences of frames
        list of sequences of labels
    """
    sequence = []
    sequence_labels = []
    for i in range(0, len(frames) - sequence_length + 1, steps):
        if labels[i] == 1:
            continue
        sequence.append(frames[i:i + sequence_length])
        sequence_labels.append(labels[i:i + sequence_length])

    return sequence, sequence_labels

def aedat4_to_sequences(
        input_dir: str, 
        output_dir: str, 
        duration: int = 10,
        sequence_length: int = 300,
        steps: int = 100,
        encoding_type: str = 'accumulate'
    ):
    """
    Convert a directory of aedat4 files (with labels as subdirectories) to sequences of frames and save them as pytorch tensors.
    """
    press_df = pd.read_csv(os.path.join(input_dir, 'labels.csv'))
    capture_info_df = pd.read_csv(os.path.join(input_dir, 'capture_info.csv'))

    for file in os.listdir(input_dir):
        if file.endswith('.aedat4'):
            print("Processing file:", file)
            file_path = os.path.join(input_dir, file)
            file_prefix = file.replace('.aedat4', '')
            file_df = press_df.loc[press_df['id'] == file_prefix] # 3 columns: id, start, end
            press_times_list = np.array(file_df[['start', 'end']].values.flatten().tolist()) + capture_info_df.loc[capture_info_df['id'] == file_prefix, 'capture_start'].values[0]
            frames, labels = dv_data_frame_tSlice(
                file_path=file_path,
                press_times_list=press_times_list,
                duration=duration,
                encoding_type=encoding_type
            )
            frames = preprocess_frames(frames, encoding_type=encoding_type) # add transformations
            sequences, sequence_labels = create_sequence(frames, labels, sequence_length=sequence_length, steps=steps)
            
            os.makedirs(output_dir, exist_ok=True)
            for i, (sequence, sequence_label) in enumerate(zip(sequences, sequence_labels)):
                sequence_data = {
                    'sequence': torch.stack(sequence),
                    'label': torch.tensor(sequence_label)
                }
                torch.save(sequence_data, os.path.join(output_dir, f"{file_prefix}_seq_{i}.pt"))