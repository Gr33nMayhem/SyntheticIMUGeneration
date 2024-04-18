# import the necessary packages
import os
import sys

sys.path.append(os.path.join(".."))

import pandas as pd
import torch

from model_train.preprocess_mag import DataPreprocessMag
from model_train.experiment import Experiment

reference_point = 'shoulder'
window_size = 100
step_size = 5
imu_position = 'hand'
learning_rate = 0.0001
batch_size = 256
num_epochs = 60
loss_function = 'mse'

pd.options.display.float_format = '{:.2f}'.format

# check if cuda is available
print(torch.cuda.is_available())
torch.autograd.set_detect_anomaly(True)

data_path = '../data/VTT_ConIot_Dataset'
IMU_path = data_path + '/IMU'
Keypoint_path = data_path + '/Keypoint'

num_input_features = 18
num_output_features = 1

# main method

preprocess = DataPreprocessMag(reference_point, window_size, step_size)
# get the processed data
keypoint_data, imu_data, sliding_windows = preprocess.processed_data()

# replace any nan value as the mean of the value before and after it
keypoint_data = keypoint_data.fillna(keypoint_data.mean())
imu_data = imu_data.fillna(imu_data.mean())

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
input_size = (window_size, num_input_features)
output_size = (window_size, num_output_features)

save_path = f'../data/results/encoder_decoder_model_new_{reference_point}_{window_size}_{step_size}_{imu_position}_{learning_rate}_{batch_size}_{num_epochs}_{loss_function}'

experiment = Experiment(participants=[1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13], input_size=input_size,
                        keypoint_data=keypoint_data, imu_data=imu_data, sliding_window=sliding_windows,
                        output_size=output_size, device=device, save_path=save_path, batch_size=batch_size,
                        num_epochs=num_epochs, learning_rate=learning_rate, loss_function=loss_function)

experiment.run_experiment()
