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

def custom_encode(events: dv.EventStore, resolution: tuple[int, int]) -> np.ndarray:
    H, W = resolution
    frame_on = np.zeros((H, W), dtype=np.float32)
    frame_off = np.zeros((H, W), dtype=np.float32)

    if events.size() == 0:
        return np.stack([frame_on, frame_off], axis=0)

    arr = events.numpy()  # dtype [('timestamp','<i8'), ('x','<i2'), ('y','<i2'), ('polarity','i1')]
    t_array = arr['timestamp']
    x_array = arr['x']
    y_array = arr['y']
    p_array = arr['polarity']

    lowest_time = t_array.min()
    highest_time = t_array.max()
    batch_time = highest_time - lowest_time + 1

    # Compute encoded times
    scale = 2 * np.pi / batch_time
    encoded_times = np.sin(scale * (t_array - lowest_time))

    on_mask = (p_array == 1)
    off_mask = ~on_mask

    np.add.at(frame_on, (x_array[on_mask], y_array[on_mask]), encoded_times[on_mask])
    # Accumulate OFF events
    np.add.at(frame_off, (x_array[off_mask], y_array[off_mask]), encoded_times[off_mask])

    return np.stack([frame_on, frame_off], axis=0) # Shape: (2, H, W)