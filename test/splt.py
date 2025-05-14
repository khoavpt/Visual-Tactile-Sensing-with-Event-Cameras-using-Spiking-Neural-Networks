import torch
import snntorch.spikeplot as splt
import matplotlib.pyplot as plt

#  spike_data contains 128 binary samples, each of 100 time steps in duration
spike_data = torch.randn(100, 128, 1, 28, 28)
spike_data = torch.where(spike_data > 0.5, 1, 0)

print(spike_data.size())

#  Index into a single sample from a minibatch
spike_data_sample = spike_data[:, 0, 0]
spike_data_sample = spike_data_sample.view(100, -1)

print(spike_data_sample.size())

fig = plt.figure(facecolor="w", figsize=(10, 5))
ax = fig.add_subplot(111)

#  s: size of scatter points; c: color of scatter points
splt.raster(spike_data_sample, ax, s=1.5, c="black")
plt.title("Input Layer")
plt.xlabel("Time step")
plt.ylabel("Neuron Number")
plt.savefig("figures/input_layer.png")