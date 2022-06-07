import joblib
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.utils.data
from torch.utils.data import TensorDataset, DataLoader
from torch.optim.lr_scheduler import ReduceLROnPlateau
import psycopg2
import AdvancedModel

from sklearn.feature_extraction.text import CountVectorizer
from sklearn.feature_extraction.text import TfidfTransformer
from sklearn.metrics import f1_score
from torchmetrics import F1Score

import matplotlib
import matplotlib.pyplot as plt

with torch.no_grad():
    # split data into training, validation, and test data (features and labels, x and y)
    split_idx = int(0.6*len(x))  # about 80%
    train_x, remaining_x = x[:split_idx], x[split_idx:]
    train_y, remaining_y = y[:split_idx], y[split_idx:]

    test_idx = (len(x)-split_idx)//2
    val_x, test_x = remaining_x[:test_idx-2], remaining_x[test_idx-2:-4]
    val_y, test_y = remaining_y[:test_idx-2], remaining_y[test_idx-2:-4]
    test_x_raw = x_df[split_idx+test_idx-2:-4]

    inputsize = x.shape[1]
    hidden_size = 8
    model = AdvancedModel.NeuralNet(
        input_size=inp_size, hid_size=hidden_size)
    model.load_state_dict(torch.load("./adv_model.pth"))
    model.eval()

    predictions = []
    for batch_idx, (x, y) in enumerate(test_data):
        out = model(x.float())
        # print(out)
        predictions.append(torch.round(out).int())

    ys = [y for x, y in test_data]
    f1 = F1Score(num_classes=2)
    f1score = f1(torch.tensor(ys), torch.tensor(predictions))
    print(f'advanced model F1: {f1score}')

    df_x = AdvancedModel.test_x_raw
    df_y = AdvancedModel.test_y
    # df_y = [pd.DataFrame(y.numpy()) for x, y in test_data]
    # print(df_x)
    loaded_model = joblib.load('SVC_finalized.sav')
    result = loaded_model.predict(df_x)
    modf1Score = f1_score(df_y, result)
    print(f'f1 score of svc model: {modf1Score}')
