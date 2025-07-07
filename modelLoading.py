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

# Load the state model
model = nn.Linear(2,1)  # Assuming a simple linear model with 2 input features
model.load_state_dict(torch.load('./model/model.pt', weights_only=True))

model.eval()  # Set the model to evaluation mode


# create prediction tensor

X_data = torch.tensor([
    [5, 10000],
    [2, 10000],
    [5, 20000],
], dtype=torch.float32)  # Example: 5 years old car with 20,000 miles


with torch.no_grad():
    predictions = model((X_data - X_mean) / X_sdt)  # Normalize the input data
    predictions = predictions * y_std + y_mean  # Denormalize the output data

