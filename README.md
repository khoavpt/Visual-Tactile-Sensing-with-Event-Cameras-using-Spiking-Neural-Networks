# Visual Tactile Sensing with Event Cameras using Spiking Neural Networks
<div align="center">
  <img src="figures/demo.gif" alt="Demo Video" style="width: 100%; height: auto;">
</div>

This repository experiments with different methods for tactile sensing using event cameras and spiking neural networks.

## Usage

### 1. Clone the Repository
```bash
git clone https://github.com/khoavpt/Visual-Tactile-Sensing-with-Event-Cameras-using-Spiking-Neural-Networks.git
```

### 2. Install Dependencies
```bash
pip install -r requirements.txt
```

## Training Models for Tactile Sensing

### Step 1: Download Raw Data
Download raw AEDAT4 data from the provided [Google Drive link](https://drive.google.com/drive/folders/1Aac3Xi6cK6aUR3ELnsJXgvIMCjjYVNi0) and place it in the `data/raw_data` directory.

### Step 2: Process Raw Data for Training
Convert the raw AEDAT4 data into sequences of frames and save them as `.pt` files:
```bash
python src/process_train_data.py
```

#### Configuration File
This script uses the configuration specified in `configs/data_module/data_cf.yaml`.

#### Customizable Data Processing Parameters
- `batch_size`: Batch size for data loading.
- `num_workers`: Number of worker threads for data loading.
- `frame_duration`: Duration of each frame in milliseconds.
- `encoding_type`: Encoding type (`accumulate`, `time_surface`, `custom`).
- `sequence_length`: Length of each sequence.
- `input_dir`: Directory containing raw AEDAT4 data.
- `output_dir`: Directory to save processed sequences.

#### Overriding Parameters
You can override parameters directly from the command line:
```bash
python src/process_train_data.py data_module.batch_size=64 data_module.frame_duration=20 data_module.input_dir=data/raw_data data_module.output_dir=data/seq_data_accumulate
```

### Step 3: Train the Model
After processing the data, train the model using `train.py`:
```bash
python src/train.py
```

This script uses the configuration specified in `configs/model`. You can switch between models and adjust hyperparameters as needed.

#### Switching Between Models
To switch between different models (e.g., ConvLSTM , ConvSNN), specify the model configuration in the command line:
```bash
python src/train.py model=convsnn_cf
```

#### Customizable Model Parameters
- **For ConvSNN:**
  - `feature_size`: Size of the feature vector.
  - `beta_init`: Initial beta value.
  - `spikegrad`: Surrogate gradient function (`fast_sigmoid`, `arctan`, `heaviside`).
  - `in_channels`: Number of input channels (`custom` encoding methods require 2 channels)

- **For ConvLSTM:**
  - `feature_size`: Size of the feature vector for LSTM
  - `in_channels`: Number of input channels (`custom` encoding methods require 2 channels)

#### Customizable Trainer Parameters
- `max_epochs`: Maximum number of training epochs.
- `accelerator`: Hardware accelerator (`cpu`, `gpu`).

Example:
```bash
python src/train.py trainer.max_epochs=50 trainer.accelerator=gpu
```

## Real-Time Inference 
(Under development)
