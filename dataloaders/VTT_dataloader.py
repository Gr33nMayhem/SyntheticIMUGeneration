# import the necessary packages
import numpy as np
import pandas as pd
from torch.utils.data import Dataset

pd.options.display.float_format = '{:.2f}'.format


class DataSet_VTT(Dataset):
    def __init__(self, keypoint_data, imu_data, sliding_windows, flag='train'):
        self.flag = flag
        # if train use all data except for user 1
        if self.flag == 'train':
            self.keypoint_data = keypoint_data[keypoint_data['subject'] != 1]
            self.imu_data = imu_data[imu_data['subject'] != 1]
            self.sliding_windows_map = sliding_windows[sliding_windows['subject'] != 1]
            # reset the index
            self.sliding_windows_map = self.sliding_windows_map.reset_index(drop=True)
        # if test use only user 1
        elif self.flag == 'test':
            self.keypoint_data = keypoint_data[keypoint_data['subject'] == 1]
            self.imu_data = imu_data[imu_data['subject'] == 1]
            self.sliding_windows_map = sliding_windows[sliding_windows['subject'] == 1]
            # reset the index
            self.sliding_windows_map = self.sliding_windows_map.reset_index(drop=True)

    def __len__(self):
        return len(self.sliding_windows_map["start"])

    def __getitem__(self, idx):
        start = self.sliding_windows_map["start"][idx]
        end = self.sliding_windows_map["end"][idx]
        user = self.sliding_windows_map["subject"][idx]
        activity = self.sliding_windows_map["activity"][idx]
        # get the keypoint data for user and activity between start and end
        keypoint = self.keypoint_data[
                       (self.keypoint_data['subject'] == user) & (self.keypoint_data['activity'] == activity)].loc[
                   start:end]
        # check if any nan values in keypoint
        # print(np.isnan(keypoint).any())
        keypoint = keypoint.drop(columns=['subject', 'activity'])
        # expand dimensions to make it 4D
        keypoint = np.expand_dims(keypoint, axis=0)
        imu = self.imu_data[(self.imu_data['subject'] == user) & (self.imu_data['activity'] == activity)].loc[
              start:end]
        imu = imu.drop(columns=['subject', 'activity'])
        # imu to numpy
        imu = imu.to_numpy()

        # check if any nan values in keypoint
        # print(np.isnan(keypoint).any())
        # print('----------------')
        return keypoint, imu
