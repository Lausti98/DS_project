from tkinter import HIDDEN
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.utils.data
from torch.utils.data import TensorDataset, DataLoader
from torch.optim.lr_scheduler import ReduceLROnPlateau
import psycopg2


class NeuralNet(nn.Module):
    '''Definition of Neural network architecture'''

    def __init__(self, input_size, hid_size1, hid_size2):
        # Call base class constructor (this line must be present)
        super(NeuralNet, self).__init__()

        # Define layers
        # 2 inputs to 1 output
        self.layer1 = nn.Linear(input_size, hid_size1)
        ###
        self.RelU = nn.ReLU()
        ###
        self.sig = nn.Sigmoid()
        self.hid1 = nn.Linear(hid_size1, hid_size2)
        self.relu = nn.ReLU()
        self.layer2 = nn.Linear(hid_size2, 1)    # 1 input to 1 output

    def forward(self, x):
        '''Define forward operation (backward is automatically deduced)'''
        x = self.layer1(x)
        x = self.RelU(x)
        x = self.hid1(x)
        x = self.relu(x)
        x = self.layer2(x)
        x = self.sig(x)
        # x = F.softmax(x, dim = 1)

        return x
