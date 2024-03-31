import torch
from torch.utils.data import Dataset
#Dataset is a class used in pytorch to represent a Dataset

class ChatDataset(Dataset):
        def __init__(self,X_train, Y_train):
            #initializes the dataset object, takes 2 arguments X and Y train, which represents Input and Output of the dataset
            self.n_samples = len(X_train)
            #calculates the number of sample in the datasets by taking the lenght of input data X train 
            self.x_data = X_train
            self.y_data = Y_train
            #it assigns the input data x train and output data y train

        def __getitem__(self, index):
            #this method allows indexxing of dataset object. it returns the input -output pair for a given index values
            return self.x_data[index], self.y_data[index]

        def __len__(self):
            #returns total number of values in the dataset
            return self.n_samples