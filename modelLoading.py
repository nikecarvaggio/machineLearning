import sys
import os
import pandas as pd
import torch
from torch import nn

# Load tensor normalization parameters
X_mean = torch.load('./model/X_mean.pt', weights_only=True)
X_sdt = torch.load('./model/X_sdt.pt', weights_only=True)
y_mean = torch.load('./model/y_mean.pt', weights_only=True)
y_std = torch.load('./model/y_std.pt', weights_only=True) 

# Load the model
model = torch.load('./model/model.pt', weights_only=True)

# create prediction tensor

X_data = torch.tensor([
    [5, 10000],
    [2, 10000],
    [5, 20000],
                       ], dtype=torch.float32)  # Example: 5 years old car with 20,000 miles
prediction = model((X_data - X_mean)/ X_sdt)   # Normalize the input data
print(prediction * y_std + y_mean)  # Denormalize the prediction
