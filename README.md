# Human Activity Recognition Efficiency Estimation

A comprehensive machine learning system for recognizing worker activities during agricultural harvesting and estimating worker efficiency based on multimodal timeseries GPS, IMU and load cell data.

## Overview

This project implements a custom neural network architecture (YieldNN) to classify agricultural worker activities into **Pick** and **NoPick** categories using sensor data, followed by efficiency analysis of harvesting operations. YieldNN has CNN-LSTM based architecture.
## Features

- **Activity Classification**: Classification of worker activities (Pick vs NoPick)
- **Efficiency Estimation**: Calculate and evaluate worker harvesting efficiency


## Installation

### Prerequisites

- Python 3.8+

### Dependencies

Install all required dependencies using the provided requirements file:

```bash
pip install -r requirements.txt
```
## Usage

### 1. Model Training and Testing

Run the training notebook to train a new model or evaluate existing models:

```python
# Open Step1_train_test_yieldnn_classfication.ipynb
# Set train_val_test=True for training, False for evaluation only
```

**Key Parameters:**
- `train_val_test`: Boolean to enable/disable training
- Input shape: `(9600, N)` - 9600 time steps with N sensor channels

### 2. Individual Day Efficiency Analysis

Process and analyze efficiency for specific harvest days:

```python
# Open Step2_1_compute_evaluate_efficiecny.ipynb
# Configure root directory and date parameters
```

### 3. Season-long Processing

#### 3.1 Season-long Activity Classification

Classify activities across entire harvesting seasons using the trained model:

```python
# Manual execution:
# Open Step3_1_classify_seasonlong_data_pick_nopick.ipynb
# Set parameters: data_root, date_time, field name

# Automated batch processing:
python RunAll_Step3_1_classify_seasonlong_data_pick_nopick.py
```


#### 3.2 Season-long Efficiency Computation

Compute and analyze picker efficiency metrics across the entire season:

```python
# Open Step3_2_compute_seasonlong_picker_efficiency.ipynb
# Configure field parameters and data paths
```

## Model Architecture

### YieldNN Architecture

The YieldNN model is a custom 1D CNN-based architecture designed for time-series activity recognition:

- **Input**: 4-channel sensor data (accelerometer + mass sensor)
- **Architecture**: Encoder-decoder with skip connections
- **Pooling Rates**: [8, 6, 4, 2] for multi-scale feature extraction
- **Filters**: Progressive increase from 16 to 128 channels
- **Output**: Binary classification (Pick/NoPick)

### Data Format

Expected input data format:
```
CSV Headers: ["rpi_utc_time", "gps_utc_time", "GPS_TOW", "LAT", "LON", "HEIGHT", "ax", "ay", "az", "raw_mass"]
Used Channels: ["GPS_TOW", "LAT", "LON", "HEIGHT", "ax", "ay", "az", "raw_mass"]
```
