# import the necessary packages
import sys
import os

sys.path.append(os.path.join(".."))

import pandas as pd
import torch
import torch.nn as nn
from scipy.signal import resample
from sklearn.preprocessing import StandardScaler
from torch.utils.data import DataLoader

from dataloaders.VTT_dataloader import DataSet_VTT
from models.EncoderDecoderModel import EncoderDecoder

pd.options.display.float_format = '{:.2f}'.format

# check if cuda is available
print(torch.cuda.is_available())
torch.autograd.set_detect_anomaly(True)

data_path = '../data/VTT_ConIot_Dataset'
IMU_path = data_path + '/IMU'
Keypoint_path = data_path + '/Keypoint'
activities = [1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15]
users = [1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13]


class DataPreprocess:
    def __init__(self, reference_point, window_size, step_size, imu_position):
        self.reference_point = reference_point
        self.window_size = window_size
        self.step_size = step_size
        self.imu_position = imu_position

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
        # only keep the columns in imu data that are of accelerometer, i.e. that have _A in the name
        imu_data = imu_data[[col for col in imu_data.columns if '_A' in col]]
        # remove frame_number and timestamp columns from keypoint data
        keypoint_data = keypoint_data.drop(columns=['frame_number', 'timestamp'])
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

    def create_relative_keypoints(self, df_keypoints, relative_to='head'):
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
        There are 17 points with each point having their x, y and confidence values
        :param df_keypoints:
        :param relative_to: Can be 'head' or 'shoulder'
        :param relative_motion:
        :return:
        '''
        # drop the column "detection_score" as they are all high
        df_keypoints = df_keypoints.drop(columns=['detection_score'])

        if relative_to == 'head':
            '''create a average column of the x and y of nose, left eye, right eye, left ear and right ear 
               and call it x_head, y_head, prob_headwhere the columns for x and y are labelled as 
               x0,y0,prob0,x1,y1,prob1,....,x16,y16,prob16
               get the x and y values for the nose, left eye, right eye, left ear and right ear'''
            x_root = (df_keypoints['x0'] + df_keypoints['x1'] + df_keypoints['x2'] + df_keypoints['x3'] +
                      df_keypoints['x4']) / 5
            y_root = (df_keypoints['y0'] + df_keypoints['y1'] + df_keypoints['y2'] + df_keypoints['y3'] +
                      df_keypoints['y4']) / 5

            # set prob_head as the maximum of the probabilities of the nose, left eye, right eye, left ear and right ear
            prob_root = df_keypoints[['prob0', 'prob1', 'prob2', 'prob3', 'prob4']].max(axis=1)
        else:
            ''' Use the mean of the left and the right shoulder as the root point'''
            x_root = (df_keypoints['x5'] + df_keypoints['x6']) / 2
            y_root = (df_keypoints['y5'] + df_keypoints['y6']) / 2
            prob_root = (df_keypoints['prob5'] + df_keypoints['prob6']) / 2

        # add the columns to the dataframe
        df_keypoints['x_root'] = x_root
        df_keypoints['y_root'] = y_root
        df_keypoints['prob_root'] = prob_root

        # drop the columns for the nose, left eye, right eye, left ear and right ear
        df_keypoints = df_keypoints.drop(columns=['x0', 'y0', 'prob0', 'x1', 'y1', 'prob1', 'x2', 'y2', 'prob2',
                                                  'x3', 'y3', 'prob3', 'x4', 'y4', 'prob4'])

        # convert the x and y values to be relative to the head
        for i in range(5, 17):
            df_keypoints[f'x{i}'] = df_keypoints[f'x{i}'] - df_keypoints['x_root']
            df_keypoints[f'y{i}'] = df_keypoints[f'y{i}'] - df_keypoints['y_root']

        # drop the columns for the x and y values of the head
        df_keypoints = df_keypoints.drop(columns=['x_root', 'y_root', 'prob_root'])

        return df_keypoints

    def create_delta_keypoints(self, df_keypoints):
        '''
        This function creates the delta keypoints from the keypoint data
        :param df_keypoints:
        :return:
        '''
        # get the columns that have x and y values
        x_cols = [col for col in df_keypoints.columns if 'x' in col]
        y_cols = [col for col in df_keypoints.columns if 'y' in col]
        # get the columns that have prob values
        prob_cols = [col for col in df_keypoints.columns if 'prob' in col]

        final_keypoints = pd.DataFrame()
        # create a new dataframe to store the delta values
        delta_keypoints = pd.DataFrame()
        # for x and y cols, reset their values as the difference between the current value and the previous value
        for x_col, y_col in zip(x_cols, y_cols):
            delta_keypoints[f'delta_{x_col}'] = df_keypoints[x_col].diff()
            delta_keypoints[f'delta_{y_col}'] = df_keypoints[y_col].diff()

        # replace the nan values with 0

        delta_keypoints = delta_keypoints.fillna(0.5)

        # recreate the dataframe in columns format x , y, prob
        for x_col, y_col, prob_col in zip(x_cols, y_cols, prob_cols):
            final_keypoints[x_col] = delta_keypoints[f'delta_{x_col}']
            final_keypoints[y_col] = delta_keypoints[f'delta_{y_col}']
            final_keypoints[prob_col] = df_keypoints[prob_col]

        # add the activity and subject columns back
        final_keypoints['activity'] = df_keypoints['activity']
        final_keypoints['subject'] = df_keypoints['subject']

        return final_keypoints

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

        ''' Resampling and Cropping of Data Started'''
        # pre-pocess by removing the extra time stamps fom either keypoint or imu data
        # per participant and activity pair there should be 4x the number of imu data points as keypoint data points
        cropped_keypoint_data = pd.DataFrame()
        cropped_imu_data = pd.DataFrame()
        for activity in activities:
            for user in users:
                # print shape of keypoint and imu data
                keypoint_data = full_keypoint_data[
                    (full_keypoint_data['subject'] == user) & (full_keypoint_data['activity'] == activity)]
                imu_data = full_imu_data[(full_imu_data['subject'] == user) & (full_imu_data['activity'] == activity)]

                # if either of the data is empty, skip
                if keypoint_data.shape[0] == 0 or imu_data.shape[0] == 0:
                    continue

                if keypoint_data.shape[0] * 4 > imu_data.shape[0]:
                    # remove the extra keypoint data
                    keypoint_data = keypoint_data.iloc[:imu_data.shape[0] // 4]
                else:
                    # remove the extra imu data
                    imu_data = imu_data.iloc[:keypoint_data.shape[0] * 4]
                # remove keypoint values such that keypoint data is multiple of 25
                keypoint_data = keypoint_data.iloc[:keypoint_data.shape[0] // 25 * 25]

                # remove imu values such that imu data is multiple of 100
                imu_data = imu_data.iloc[:imu_data.shape[0] // 100 * 100]

                # make copy of the columns subject and activity before dropping
                subject_col = imu_data['subject']
                activity_col = imu_data['activity']
                imu_data = imu_data.drop(columns=['subject', 'activity'])
                # resample the imu data to be the same length as the keypoint data
                imu_resampled = resample(imu_data, keypoint_data.shape[0])
                # create a pd dataframe from the resampled data using the columns in imu_data
                imu_data = pd.DataFrame(imu_resampled, columns=imu_data.columns)
                # add the subject and activity columns back
                imu_data['subject'] = subject_col
                imu_data['activity'] = activity_col

                cropped_keypoint_data = pd.concat([cropped_keypoint_data, keypoint_data])
                cropped_imu_data = pd.concat([cropped_imu_data, imu_data])

        # reset the index keypoint
        cropped_keypoint_data = cropped_keypoint_data.reset_index(drop=True)
        ''' Resampling and Cropping of Data Completed'''

        ''' Relative Keypoints and Sliding Windows Started'''
        # preprocess the keypoints for each point of interest from pose estimation
        cropped_keypoint_data = self.create_relative_keypoints(cropped_keypoint_data, relative_to=self.reference_point)

        # reset the index for imu
        cropped_imu_data = cropped_imu_data.reset_index(drop=True)
        ''' Relative Keypoints and Sliding Windows Completed'''

        ''' Create Delta KeyPoints Started'''
        cropped_keypoint_data = self.create_delta_keypoints(cropped_keypoint_data)
        ''' Create Delta KeyPoints Completed'''

        ''' Sliding Windows Started'''
        # create a new df between 25 key-points and 100 IMU data points for each activity and user
        # with columns start, end, subject, activity
        sliding_windows = pd.DataFrame(columns=['start', 'end', 'subject', 'activity'])

        for activity in activities:
            for user in users:
                # preserve the index
                keypoint_data = cropped_keypoint_data[
                    (cropped_keypoint_data['subject'] == user) & (cropped_keypoint_data['activity'] == activity)]
                # if either of the data is empty, skip
                if keypoint_data.shape[0] == 0:
                    continue

                # split the data into windows of window_size
                for i in range(0, keypoint_data.shape[0], self.step_size):
                    # check if the window has 25 elements
                    if i + self.window_size > keypoint_data.shape[0]:
                        continue
                    # get the start and end index of the window in cropped_keypoint_data
                    start = keypoint_data.index[i]
                    end = keypoint_data.index[i + self.window_size - 1]
                    # concat to the sliding_windows dataframe
                    sliding_windows = pd.concat([sliding_windows, pd.DataFrame(
                        {'start': [start], 'end': [end], 'subject': [user], 'activity': [activity]})])

        sliding_windows = sliding_windows.reset_index(drop=True)
        ''' Sliding Windows Completed'''

        ''' Normalize Data Started'''
        # normalize the keypoint data
        # create a copy of column activity and subject before dropping
        activity = cropped_keypoint_data['activity']
        subject = cropped_keypoint_data['subject']
        # create a copy of all columns starting with prob
        prob_cols = cropped_keypoint_data.filter(regex='prob').copy()
        # drop the columns activity, subject and all columns starting with prob
        cropped_keypoint_data = cropped_keypoint_data.drop(columns=['activity', 'subject'])
        cropped_keypoint_data = cropped_keypoint_data.drop(columns=cropped_keypoint_data.filter(regex='prob').columns)
        # Do a normalization
        scaler = StandardScaler()
        cropped_keypoint_data = pd.DataFrame(scaler.fit_transform(cropped_keypoint_data),
                                             columns=cropped_keypoint_data.columns)
        # add the columns activity and subject back
        cropped_keypoint_data['activity'] = activity
        cropped_keypoint_data['subject'] = subject
        # add the columns starting with prob back
        cropped_keypoint_data = pd.concat([cropped_keypoint_data, prob_cols], axis=1)
        ''' Normalize Data Completed'''

        ''' Remove any IMU position columns that don't start with self.imu_position Started'''
        # remove any imu position columns that don't start with self.imu_position
        imu_cols = cropped_imu_data.filter(regex=self.imu_position).columns
        # create a copy of column activity and subject before dropping
        activity = cropped_imu_data['activity']
        subject = cropped_imu_data['subject']
        # drop the columns activity, subject
        cropped_imu_data = cropped_imu_data.drop(columns=['activity', 'subject'])
        # drop the columns that don't start with self.imu_position
        cropped_imu_data = cropped_imu_data.drop(columns=cropped_imu_data.columns.difference(imu_cols))
        # Add the columns activity and subject back
        cropped_imu_data['activity'] = activity
        cropped_imu_data['subject'] = subject
        ''' Remove any IMU position columns that don't start with self.imu_position Completed'''

        return cropped_keypoint_data, cropped_imu_data, sliding_windows
