# import the necessary packages
import sys
import os

import numpy as np

sys.path.append(os.path.join(".."))

import pandas as pd
import torch
from scipy.signal import butter, resample, filtfilt

pd.options.display.float_format = '{:.2f}'.format

# check if cuda is available
print(torch.cuda.is_available())
torch.autograd.set_detect_anomaly(True)

data_path = '../data/VTT_ConIot_Dataset'
IMU_path = data_path + '/IMU'
Keypoint_path = data_path + '/Keypoint'
activities = [1, 2, 3, 4]
users = [1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13]


class DataPreprocessMag:
    def __init__(self, reference_point, window_size, step_size):
        self.reference_point = reference_point
        self.window_size = window_size
        self.step_size = step_size

    def lowpass_filter(self, data, low_cut_off=0.1, fs=100):
        b, a = butter(4, low_cut_off, fs=fs, btype='lowpass', analog=False)
        y = filtfilt(b, a, data)
        return b, a, y

    def highpass_filter(self, data, high_cut_off=0.1, fs=100):
        b, a = butter(4, high_cut_off, fs=fs, btype='highpass', analog=False)
        y = filtfilt(b, a, data)
        return b, a, y

    '''
            "keypoints": [
                    "nose","left_eye","right_eye","left_ear","right_ear",
                    "left_shoulder","right_shoulder","left_elbow","right_elbow",
                    "left_wrist","right_wrist","left_hip","right_hip",
                    "left_knee","right_knee","left_ankle","right_ankle"
                ],
            "corresponding points":[
                    "0", "1", "2", "3", "4",
                    "5", "6", "7", "8",
                    "9", "10", "11", "12",
                    "13", "14", "15", "16"
                ]
    '''

    def create_keypoint_and_imu_data(self, subject, activity):
        '''
        This function reads the keypoint and imu data for a given subject and activity
        :param subject:
        :param activity:
        :return:
        '''
        # check if file exists
        if not os.path.exists(Keypoint_path + f'/Subject_{subject:02d}_Task_{activity}.m2ts_keyPoints.csv'):
            print(f'Keypoint file for Subject_{subject:02d}_Task_{activity} does not exist')
            # return two empty dataframes
            return pd.DataFrame(), pd.DataFrame()

        keypoint_data = pd.read_csv(Keypoint_path + f'/Subject_{subject:02d}_Task_{activity}.m2ts_keyPoints.csv')
        imu_data = pd.read_csv(IMU_path + f'/activity_{activity}_user_{subject}_combined.csv')

        ''' Remove any na values '''
        keypoint_data = keypoint_data.dropna()
        imu_data = imu_data.dropna()
        ''' --------------------------------------------------- '''

        ''' Remove any unnecessary columns '''
        # only keep the columns in imu data that are of accelerometer, i.e. that have _A in the name and are for hand
        imu_data = imu_data[[col for col in imu_data.columns if 'hand_A' in col]]
        # remove frame_number and timestamp columns from keypoint data
        keypoint_data = keypoint_data.drop(columns=['frame_number', 'timestamp', 'detection_score'])
        ''' --------------------------------------------------- '''

        ''' Get the approximate shoulder and hip '''
        shoulder_mean_x = (keypoint_data['x5'] + keypoint_data['x6']) / 2
        shoulder_mean_y = (keypoint_data['y5'] + keypoint_data['y6']) / 2
        hip_mean_x = (keypoint_data['x11'] + keypoint_data['x12']) / 2
        hip_mean_y = (keypoint_data['y11'] + keypoint_data['y12']) / 2
        ''' --------------------------------------------------- '''

        ''' Get the approximate distance from shoulder to hip '''
        distance = np.sqrt((shoulder_mean_x - hip_mean_x) ** 2 + (shoulder_mean_y - hip_mean_y) ** 2)
        # remove the high frequency noise from the distance data
        b, a, distance = self.lowpass_filter(distance, low_cut_off=2, fs=25)
        ''' --------------------------------------------------- '''

        ''' Acquire the gravity component and remove from imu '''
        b, a, x_g = self.lowpass_filter(imu_data['hand_Ax_g'].values, low_cut_off=0.1, fs=400)
        b, a, y_g = self.lowpass_filter(imu_data['hand_Ay_g'].values, low_cut_off=0.1, fs=400)
        b, a, z_g = self.lowpass_filter(imu_data['hand_Az_g'].values, low_cut_off=0.1, fs=400)
        # remove gravity component
        imu_data['hand_Ax_g'] = imu_data['hand_Ax_g'] - x_g
        imu_data['hand_Ay_g'] = imu_data['hand_Ay_g'] - y_g
        imu_data['hand_Az_g'] = imu_data['hand_Az_g'] - z_g
        ''' --------------------------------------------------- '''

        ''' Remove the high frequency noise from the acceleration data '''
        b, a, imu_data['hand_Ax_g'] = self.lowpass_filter(imu_data['hand_Ax_g'].values, low_cut_off=20, fs=400)
        b, a, imu_data['hand_Ay_g'] = self.lowpass_filter(imu_data['hand_Ay_g'].values, low_cut_off=20, fs=400)
        b, a, imu_data['hand_Az_g'] = self.lowpass_filter(imu_data['hand_Az_g'].values, low_cut_off=20, fs=400)
        ''' --------------------------------------------------- '''

        ''' Create a new column for the magnitude of the acceleration '''
        imu_data['hand_A_mag'] = np.sqrt(
            imu_data['hand_Ax_g'] ** 2 + imu_data['hand_Ay_g'] ** 2 + imu_data['hand_Az_g'] ** 2)
        # drop the x, y, z columns
        imu_data = imu_data.drop(columns=['hand_Ax_g', 'hand_Ay_g', 'hand_Az_g'])
        ''' --------------------------------------------------- '''

        ''' Resample imu to be the same length as keypoint data '''
        # resample the imu data to the same frequency as the keypoint data
        imu_data = resample(imu_data, keypoint_data.shape[0])
        # convert the resampled data to a dataframe
        imu_data = pd.DataFrame(imu_data, columns=['hand_A'])
        ''' --------------------------------------------------- '''

        ''' Keypoints made relative to the shoulder (0 to 16) '''
        for i in range(0, 17):
            keypoint_data[f'x{i}'] = keypoint_data[f'x{i}'] - shoulder_mean_x
            keypoint_data[f'y{i}'] = keypoint_data[f'y{i}'] - shoulder_mean_y
        # drop 0 to 6 columns
        keypoint_data = keypoint_data.drop(columns=[f'x{i}' for i in range(0, 7)] + [f'y{i}' for i in range(0, 7)])
        keypoint_data = keypoint_data.drop(columns=[f'prob{i}' for i in range(0, 7)])
        # drop 13 to 16 columns
        keypoint_data = keypoint_data.drop(columns=[f'x{i}' for i in range(13, 17)] + [f'y{i}' for i in range(13, 17)])
        keypoint_data = keypoint_data.drop(columns=[f'prob{i}' for i in range(13, 17)])
        ''' --------------------------------------------------- '''

        ''' Divide all the keypoint data by the distance '''
        for i in range(7, 13):
            keypoint_data[f'x{i}'] = keypoint_data[f'x{i}'] / distance
            keypoint_data[f'y{i}'] = keypoint_data[f'y{i}'] / distance
        ''' --------------------------------------------------- '''

        keypoint_data['subject'] = subject
        keypoint_data['activity'] = activity
        imu_data['subject'] = subject
        imu_data['activity'] = activity
        # make sure subject and activity are of type float
        keypoint_data['subject'] = keypoint_data['subject'].astype(float)
        keypoint_data['activity'] = keypoint_data['activity'].astype(float)
        imu_data['subject'] = imu_data['subject'].astype(float)
        imu_data['activity'] = imu_data['activity'].astype(float)
        return keypoint_data, imu_data

    def processed_data(self):
        '''
        This function reads the keypoint and imu data for all subjects and activities. Then processes it.
        :return:
        '''

        # create a dataframe with all the keypoint and imu data
        full_keypoint_data = pd.DataFrame()
        full_imu_data = pd.DataFrame()
        for activity in activities:
            for user in users:
                keypoint_data, imu_data = self.create_keypoint_and_imu_data(user, activity)
                full_keypoint_data = pd.concat([full_keypoint_data, keypoint_data])
                full_imu_data = pd.concat([full_imu_data, imu_data])

        # get the sliding windows
        sliding_windows = pd.DataFrame(columns=['start', 'end', 'subject', 'activity'])
        for activity in activities:
            for user in users:
                # preserve the index
                keypoint_data = full_keypoint_data[
                    (full_keypoint_data['subject'] == user) & (full_keypoint_data['activity'] == activity)]
                # if either of the data is empty, skip
                if keypoint_data.shape[0] == 0:
                    continue

                # split the data into windows of window_size
                for i in range(0, keypoint_data.shape[0], self.step_size):
                    # check if the window has 25 elements
                    if i + self.window_size > keypoint_data.shape[0] - 1:
                        continue
                    # get the start and end index of the window in cropped_keypoint_data
                    start = keypoint_data.index[i]
                    end = keypoint_data.index[i + self.window_size]
                    # concat to the sliding_windows dataframe
                    sliding_windows = pd.concat([sliding_windows, pd.DataFrame(
                        {'start': [start], 'end': [end], 'subject': [user], 'activity': [activity]})])

        sliding_windows = sliding_windows.reset_index(drop=True)

        return full_keypoint_data, full_imu_data, sliding_windows
