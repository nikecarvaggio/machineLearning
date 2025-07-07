import sys
import os
from matplotlib import axis
import matplotlib.pyplot as plt
import pandas as pd
import torch 
from torch import nn

# Read in the CSV file
df = pd.read_csv('./data/used_cars.csv')

# Obtain model year and milage
age = df['model_year'].max() - df['model_year']

# Obtain milage
milage = df['milage']
milage = milage.str.replace('mi.', '')
milage = milage.str.replace(',', '').astype(int)

# Obtain price
price = df['price']
price = price.str.replace('$', '').str.replace(',', '').astype(float)

if not os.path.isdir('./model'):
    os.mkdir('./model')

#Torch: Create X as age and y milage tensors 
X = torch.column_stack([
    torch.tensor(age, dtype=torch.float32),
    torch.tensor(milage, dtype=torch.float32)
    ])
X_mean = X.mean(axis=0)
X_sdt = X.std(axis=0)

# Save the mean and standard deviation of X for normalization
torch.save(X_mean, './model/X_mean.pt')
torch.save(X_sdt, './model/X_sdt.pt')


X = (X - X_mean) / X_sdt  # Normalize the X features
y = torch.tensor(price, dtype=torch.float32).reshape(-1, 1)
print(y)

y_mean = y.mean()
y_std = y.std()

# Save the mean and standard deviation of y for normalization
torch.save(y_mean, './model/y_mean.pt')
torch.save(y_std, './model/y_std.pt')

# Normalize the y features
y = (y - y_mean) / y_std



# Create a neural network model with 2 inputs, 1 hidden layer, and 1 output

model = nn.Linear(2, 1)
# Define loss fuction and optimizer
loss_fn = torch.nn.MSELoss()
optimizer = torch.optim.SGD(model.parameters(), lr=0.001 )

losses = []

for i in range(0, 2500):
    # Trainning pass
    optimizer.zero_grad()
    outputs = model(X)
    loss = loss_fn(outputs, y)
    loss.backward()
    optimizer.step()
    
    losses.append(loss.item())
    #if i % 100 == 0:
        #print(f'Loss: {loss.item()}')
        #print(f'Epoch {i}, Loss: {loss.item()}')
        #print(f'Weights: {model.weight.data}, Bias: {model.bias.data}')

torch.save(model.state_dict(), './model/model.pt')

plt.plot(losses)
plt.xlabel('Epochs')
plt.ylabel('Loss')
plt.title('Training Loss Over Epochs')
plt.show()

