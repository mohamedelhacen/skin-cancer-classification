import argparse
from sklearn.model_selection import train_test_split

import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms

from dataset import preprocessing_PAD, PADCancerDataset
from train import Train
from network import Network

parser = argparse.ArgumentParser(description='Process and Train the Network')

parser.add_argument('--data.original_path', type=str, default='skin-cancer', metavar='DS',
                    help="data path name (default: skin-cancer)")
parser.add_argument('--data.destination_path', type=str, default='all_images', metavar='DS',
                    help="destination path name (default: all_images)")

parser.add_argument('--train.epochs', type=int, default=4, metavar='DS',
                    help="number of epochs (default: 4)")
parser.add_argument('--train.batch_size', type=int, default=8, metavar='DS',
                    help="number of epochs (default: 8)")
parser.add_argument('--log.chek_name', type=str, default='skin_cancer_classifier', metavar='DS',
                    help="number of epochs (default: skin_cancer_classifier)")


transform = transforms.Compose([
    transforms.ToTensor(),
    transforms.Resize(128)])

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')


if __name__ == '__main__':

    args = parser.parse_args()
    
    path_to_data = args.data.original_path
    destination_folder = args.data.destination_path

    labels_df, class_idx = preprocessing_PAD(path_to_data, destination_folder)
    train, test = train_test_split(labels_df, test_size=0.15, shuffle=True)
    train, val = train_test_split(train, test_size=0.15)

    train_data = PADCancerDataset(train, transform, destination_folder)
    val_data = PADCancerDataset(val, transform, destination_folder)
    test_data = PADCancerDataset(test, transform, destination_folder)


    train_loader = DataLoader(train_data, batch_size=args.train.batch_size)
    val_loader = DataLoader(val_data, batch_size=args.train.batch_size)
    test_loader = DataLoader(test_data, batch_size=args.train.batch_size)

    # Define the model
    model = Network(device)

    # Traning and testing the network
    training = Train()
    training.train(model, train_loader, val_loader, args.train.epochs)
    training.test(model, test_loader)
    
    # Saving checkpoints
    training.save_checkpoint(model, args.log.chek_name, class_idx)

