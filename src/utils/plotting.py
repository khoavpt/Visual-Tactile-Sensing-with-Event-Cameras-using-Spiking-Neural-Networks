import matplotlib.pyplot as plt
import torch
from matplotlib import animation

def plot_results(train_losses, val_losses, train_accs, val_accs, model_name, save_path=None):
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(10, 6))

    ax1.plot(train_losses, label='Train Loss')
    ax1.plot(val_losses, label='Validation Loss')
    ax1.set_xlabel('Epoch')
    ax1.set_ylabel('Loss')

    ax2.plot(train_accs, label='Train Accuracy')
    ax2.plot(val_accs, label='Validation Accuracy')
    ax2.set_xlabel('Epoch')
    ax2.set_ylabel('Accuracy')

    plt.legend()
    plt.suptitle(f'Training and Validation Loss and Accuracy of the {model_name} model')

    if save_path:
        plt.savefig(save_path)
    else:
        plt.show()

def visualize_top_errors(model, test_dataloader, model_name, top_n=5, save_path='top_errors.mp4'):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)
    model.eval()
    
    errors = []  # list of (sequence index, error_count_of_each_sequence)

    with torch.no_grad():
        batch_count = 0
        for sequence, target in test_dataloader:
            sequence = sequence.to(device)
            target = target.to(device)
            output = model(sequence)  # (batch_size, sequence_length, 2)

            for i in range(sequence.size(0)):
                error_count = (output[i].argmax(dim=-1) != target[i]).sum().item()
                errors.append((batch_count * test_dataloader.batch_size + i, error_count))

            batch_count += 1

    top_errors = sorted(errors, key=lambda x: x[1], reverse=True)[:top_n]

    sequences = []
    targets = []
    outputs = []
    class_to_idx = {0: 'no_press', 1: 'press'}

    for i, (sequence_idx, error_count) in enumerate(top_errors):
        sequence, target = test_dataloader.dataset[sequence_idx] # sequence: (sequence_length, 1, 32, 32), target: int
        output = model(sequence.unsqueeze(0).to(device))
        sequences.append(sequence)
        targets.append(target)
        outputs.append(output)

    fig, axs = plt.subplots(1, top_n, figsize=(top_n * 4, 4))
    for ax in axs:
        ax.axis('off')
    plt.suptitle(f'Top {top_n} errors of the {model_name} model')

    def init():
        for i, (sequence_idx, error_count) in enumerate(top_errors):
            img = axs[i].imshow(sequences[i][0].squeeze(), cmap='gray')
            axs[i].set_title(f"Target: {class_to_idx[targets[i]]}, Error count: {error_count} ")
        return img,

    def animate(i):
        for j, (sequence_idx, error_count) in enumerate(top_errors):
            img = axs[j].imshow(sequences[j][i].squeeze(), cmap='gray')
            axs[j].set_title(f"Target: {class_to_idx[targets[j]]}, Error count: {error_count} ")
            # Clear previous text
            for txt in axs[j].texts:
                txt.set_visible(False)
            # Display the prediction of the model on the top right corner
            axs[j].text(0.95, 0.05, f"Prediction: {class_to_idx[outputs[j][0][i].argmax(dim=-1).item()]}", color='white',
                        ha='right', va='bottom', transform=axs[j].transAxes)

        return img,

    ani = animation.FuncAnimation(fig, animate, frames=sequences[0].size(0), init_func=init, interval=100, blit=True)
    ani.save(save_path, writer='ffmpeg', fps=30)
    plt.show()