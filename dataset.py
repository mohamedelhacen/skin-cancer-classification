import os
import cv2
from PIL import Image
import shutil  
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split

import torch
from torch.utils.data import Dataset

def preprocessing_PAD(path_to_data, destination_folder):
    
    os.mkdir(destination_folder)
    def move_file_to_dest_folder(path, dest):
        for file in os.listdir(path):
            shutil.copy(os.path.join(path, file), os.path.join(dest, file))
        
    move_file_to_dest_folder(os.path.join(path_to_data, 'imgs_part_1', 'imgs_part_1'), destination_folder)
    move_file_to_dest_folder(os.path.join(path_to_data, 'imgs_part_2', 'imgs_part_2'), destination_folder)
    move_file_to_dest_folder(os.path.join(path_to_data, 'imgs_part_3', 'imgs_part_3'), destination_folder)

    metadata = pd.read_csv(os.path.join(path_to_data, 'metadata.csv'))
    labels = metadata[['diagnostic', 'img_id']]
    class_idx = {'NEV':0, 'BCC':1, 'ACK':2, 'SEK':3, 'SCC':4, 'MEL':5}
    labels['targets'] = labels['diagnostic'].map(class_idx)

    return labels, class_idx

class PADCancerDataset(Dataset):
    
    def __init__(self, labels_df, transform=None, images_path='all_images'):
        self.labels = labels_df
        self.transform = transform
        self.root_path = images_path
        
    def __len__(self):
        return len(self.labels)
    
    def __getitem__(self, idx):
        if torch.is_tensor(idx):
            idx = idx.tolist()
            
        img_id = self.labels.iloc[idx, 1]
        target = self.labels.iloc[idx, 2]
        im_path = os.path.join(self.root_path, img_id)
        image = cv2.imread(im_path)
        image = cv2.resize(image, (512, 512))
#         image = image.transpose(2, 0, 1)
        
        
        if self.transform:
            image = self.transform(image)
        
        return image, target