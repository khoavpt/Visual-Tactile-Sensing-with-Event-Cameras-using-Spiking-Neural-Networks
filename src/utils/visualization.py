import torch
import matplotlib.pyplot as plt
import snntorch.spikeplot as splt
import numpy as np
import wandb
from matplotlib.animation import FuncAnimation
import matplotlib.animation as animation

import src.models as models

def create_spike_animation(spike_data, layer_name, num_neurons=100):
    """
    Create an animation of spike activity for a layer
    
    Args:
        spike_data: Tensor of shape (sequence_length, height, width)
        layer_name: Name of the layer for the plot title
        num_neurons: Number of neurons to visualize (for large layers)
    """
    sequence_length, height, width = spike_data.shape
    
    fig, ax = plt.subplots()
    plt.title(f"Spike Activity (first channel) - {layer_name}")
    
    anim = splt.animator(spike_data, fig, ax)
    
    anim.save(f"spike_activity_{layer_name}.gif", writer='pillow', fps=10)
    return anim

def log_spike_activity_to_wandb(model, batch, layer_names=None):
    """
    Log spike activity of model layers to wandb
    
    Args:
        model: The spiking neural network model
        batch: Input batch (sequence, target)
        layer_names: List of layer names to visualize. If None, visualize all spiking layers
    """
    sequence, _ = batch
    batch_size, sequence_length = sequence.shape[0], sequence.shape[1]
    
    hidden_states = model.init_hidden_states()
    spike_data = {
        'inputs': [],
    }
    
    for t in range(sequence_length):
        x = sequence[:, t] # (batch_size, 1, 28, 28)
        output, hidden_states = model.process_frame(x, hidden_states)
        
        spike_data['inputs'].append(x[0][0])

        # ConvSNN
        if isinstance(model, models.ConvSNN):
            if 'conv1' not in spike_data:
                spike_data['conv1'] = []
            if 'conv2' not in spike_data:
                spike_data['conv2'] = []
                
            spike_data['conv1'].append(model.conv_block1.spk_list) # (height, width)
            spike_data['conv2'].append(model.conv_block2.spk_list) # (height, width)

        # SpikingConvLSTM
        elif isinstance(model, models.SpikingConvLSTM):
            if 'conv1' not in spike_data:
                spike_data['conv1'] = []
            if 'conv2' not in spike_data:
                spike_data['conv2'] = []
            if 'linear1' not in spike_data:
                spike_data['linear1'] = []
            # if 'slstm' not in spike_data:
            #     spike_data['slstm'] = []
            if 'linear2' not in spike_data:
                spike_data['linear2'] = []
            spike_data['conv1'].append(model.conv_block1.spk_list) # (height, width)
            spike_data['conv2'].append(model.conv_block2.spk_list) # (height, width)
            spike_data['linear1'].append(model.linear_block1.spk_list) # (height, width)
            # spike_data['slstm'].append(model.slstm.spk_list) # (height, width)
            spike_data['linear2'].append(model.linear_block2.spk_list) # (height, width)
                
        # SpikingConvLSTM_CBAM
        elif isinstance(model, models.SpikingConvLSTM_CBAM):
            if 'conv1' not in spike_data:
                spike_data['conv1'] = []
            if 'conv2' not in spike_data:
                spike_data['conv2'] = []
            if 'conv3' not in spike_data:
                spike_data['conv3'] = []
            if 'sconv2dlstm' not in spike_data:
                spike_data['sconv2dlstm-out'] = []
                spike_data['sconv2dlstm-channel-attention'] = []
                spike_data['sconv2dlstm-spatial-attention'] = []
            if 'linear1' not in spike_data:
                spike_data['linear1'] = []

            spike_data['conv1'].append(model.conv_block1.spk_list) # (height, width)
            spike_data['conv2'].append(model.conv_block2.spk_list) # (height, width)
            spike_data['conv3'].append(model.conv_block3.spk_list) # (height, width)
            spike_data['sconv2dlstm-out'].append(model.sconv2dlstm.spk_list) # (height, width)
            spike_data['sconv2dlstm-channel-attention'].append(model.sconv2dlstm.cbam.channel_spikes) # (height, width)
            spike_data['sconv2dlstm-spatial-attention'].append(model.sconv2dlstm.cbam.spatial_spikes) # (height, width)
            spike_data['linear1'].append(model.linear_block1.spk_list) # (height, width)

    for layer_name, spikes in spike_data.items():
        spikes = torch.stack(spikes, dim=0)  # (sequence_length, height, width)
        anim = create_spike_animation(spikes, layer_name)
        wandb.log({f"spike_animation/{layer_name}": wandb.Video(f"spike_activity_{layer_name}.gif")}) 
        print(f"Logged spike animation for {layer_name}")