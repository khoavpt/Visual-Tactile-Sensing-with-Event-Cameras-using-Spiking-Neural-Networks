import numpy as np
import dv_processing as dv  

def init_accumulator(resolution: tuple[int, int]) -> dv.Accumulator:
    """
    Initialize an accumulator with some resolution
    """
    accumulator = dv.Accumulator(resolution)

    # Apply configuration, these values can be modified to taste
    accumulator.setMinPotential(0.0)
    accumulator.setMaxPotential(1.0)
    accumulator.setNeutralPotential(0.5)
    accumulator.setEventContribution(0.15)
    accumulator.setDecayFunction(dv.Accumulator.Decay.EXPONENTIAL)
    accumulator.setDecayParam(1e+6)
    accumulator.setIgnorePolarity(False)
    accumulator.setSynchronousDecay(False)

    return accumulator

def accumulate_dv_encode(events: dv.EventStore, resolution: tuple[int, int]) -> np.ndarray:
    """
    Turn an event store into a frame using dv.Accumulator
    Args:
        events: dv.EventStore
        resolution: tuple[int, int]
    Returns:
        np.ndarray: frame
    """
    accumulator = init_accumulator(resolution)
    accumulator.accept(events)
    frame = accumulator.generateFrame()
    return frame.image

def timesurface_dv_encode(events: dv.EventStore, resolution: tuple[int, int]) -> np.ndarray:
    """
    Turn an event store into a timesurface using dv.TimeSurface
    Args:
        events: dv.EventStore
        resolution: tuple[int, int]
    Returns:
        np.ndarray: timesurface frame
    """
    surfacer = dv.TimeSurface(resolution)
    surfacer.accept(events)

    frame = surfacer.generateFrame()
    return frame.image # Shape: (H, W)

def custom_encode(events: dv.EventStore, resolution: tuple[int, int], dimension=8) -> np.ndarray:
    """
    Encode a DVS event store into a multi-channel sinusoidal-encoded tensor with shape (dimension, H, W).
    
    Args:
        events: dv.EventStore, containing events with (timestamp, x, y, polarity).
        resolution: tuple[int, int], (H, W) of the sensor grid.
        dimension: int, number of encoding dimensions (d).
    
    Returns:
        np.ndarray: Tensor of shape (dimension, H, W) with sinusoidal encoding of the most recent event per pixel.
    """
    H, W = resolution
    d = dimension
    
    # Initialize output tensor with shape (d, H, W)
    tensor = np.zeros((d, H, W), dtype=np.float32)
    
    # Handle empty event store
    if events.size() == 0:
        return tensor
    
    # Extract event data as numpy array
    arr = events.numpy()  # dtype [('timestamp','<i8'), ('x','<i2'), ('y','<i2'), ('polarity','i1')]
    t_array = arr['timestamp']
    x_array = arr['x']
    y_array = arr['y']
    p_array = arr['polarity']
    
    # Compute batch duration (in microseconds)
    lowest_time = t_array.min()
    highest_time = t_array.max()
    batch_time = highest_time - lowest_time + 1
    
    # Initialize arrays to track the most recent event per pixel
    latest_time = np.zeros((H, W), dtype=np.int64)
    latest_polarity = np.zeros((H, W), dtype=np.int8)
    
    # Find the most recent event per pixel
    for x, y, t, p in zip(x_array, y_array, t_array, p_array):
        if t > latest_time[x, y]:
            latest_time[x, y] = t
            latest_polarity[x, y] = p
    
    # Compute sinusoidal encoding for each pixel
    for x in range(H):
        for y in range(W):
            if latest_time[x, y] > 0:  # Event exists at this pixel
                # Normalize timestamp
                t = latest_time[x, y] - lowest_time
                # Polarity scaling: +0.15 for ON, -0.15 for OFF
                delta = 0.15 if latest_polarity[x, y] == 1 else -0.15
                # Compute sinusoidal encoding
                for i in range(d // 2):
                    freq = batch_time * (10000 ** (2 * i / d))
                    tensor[2*i, x, y] = delta * np.sin(t / freq)
                    tensor[2*i+1, x, y] = delta * np.cos(t / freq)
    
    # print(tensor.shape)
    return tensor # shape: (d, H, W) default: (8, 240, 180)