import torch.nn as nn
import torch.nn.functional as F
import pandas as pd
import numpy as np
import torch
from torch.utils.data import Dataset
from torch.utils.data import random_split
from torch.utils.data import DataLoader
from tqdm import trange


class CustomDataset(Dataset):
    def __init__(self, data, labels=None):
        self.data = torch.tensor(data)
        if labels is not None:
            self.labels = torch.tensor(labels)
        else:
            self.labels = None

    def __len__(self):
        return len(self.data)

    def __getitem__(self, index):
        sample = self.data[index]
        if self.labels is not None:
            label = self.labels[index]
            return sample, label
        else:
            return sample


data = pd.read_csv("small.csv", header=None)
data = data.iloc[1:, 1:].astype(np.float64)
my_dataset = CustomDataset(data.to_numpy(), None)  # Provide labels if applicable


train_size = int(0.8 * len(my_dataset))  # 80% for training
test_size = len(my_dataset) - train_size
train_dataset, test_dataset = random_split(my_dataset, [train_size, test_size])

train_loader = DataLoader(train_dataset, batch_size=4, shuffle=True)
test_loader = DataLoader(test_dataset, batch_size=4)  # No shuffling for the test set typically


# **2. Define Autoencoder Architecture**
class Autoencoder(nn.Module):
    def __init__(self):
        super().__init__()
        # Encoder
        self.conv1 = nn.Conv2d(1, 32, kernel_size=3, padding=1)
        self.conv2 = nn.Conv2d(32, 32, kernel_size=3, padding=1)
        self.pool = nn.MaxPool2d(2, 2)

        # Decoder
        self.t_conv1 = nn.ConvTranspose2d(32, 32, 2, stride=2)
        self.t_conv2 = nn.ConvTranspose2d(32, 1, 2, stride=2)

    def forward(self, x):
        x = F.relu(self.conv1(x))
        x = self.pool(x)
        x = F.relu(self.conv2(x))
        x = self.pool(x)  # Compressed representation

        x = F.relu(self.t_conv1(x))
        x = torch.sigmoid(self.t_conv2(x))
        return x


model = Autoencoder()
criterion = nn.BCELoss()
optimizer = torch.optim.Adam(model.parameters(), lr=0.001)

# **3. Train the Autoencoder**
for epoch in trange(50, desc="Training"):
    for batch in train_loader:
        optimizer.zero_grad()
        outputs = model(batch)
        loss = criterion(outputs, batch)
        loss.backward()
        optimizer.step()

        trange(epoch).set_description(f"Epoch {epoch} Loss: {loss :.3f}")


# **4. Use the Autoencoder (same ideas as Keras example)**


