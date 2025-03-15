import os
import pandas as pd
import numpy as np
from tqdm import tqdm
import torch
import torch.utils.data as data
import torchvision
from torchvision import models
import torchvision.transforms.v2 as tfs_v2
import torch.nn as nn
import torch.optim as optim
from sklearn.model_selection import train_test_split
import ast
class KinoRNN(nn.Module):
    def __init__(self, in_features, out_features):
        super().__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.hidden_size = 96

        self.rnn = nn.RNN(self.in_features, self.hidden_size,batch_first=True)
        self.out = nn.Linear(self.hidden_size, self.out_features)

    def forward(self, x):
        x, h = self.rnn(x)
        y = self.out(x[:, -1, :])
        return y

class KinoDataset(data.Dataset):
    def __init__(self, df):
        self.df = df

    def __getitem__(self, idx):
        embedding = self.df.iloc[idx]["embedding"]
        label = torch.tensor(self.df.iloc[idx]["label"], dtype=torch.long)
        return embedding, label

    def __len__(self):
        return len(self.df)

def parse_embedding(embedding_str):
    embedding_array = np.fromstring(embedding_str.strip("[]"), sep=" ")
    return torch.tensor(embedding_array, dtype=torch.float32)

df = pd.read_csv('datasets/end_data.csv')
df["embedding"] = df["embedding"].apply(parse_embedding)
print(df.head())
df_train, df_test = train_test_split(df, test_size=0.2, random_state=52)

d_train = KinoDataset(df_train)
d_test = KinoDataset(df_test)

train_data = data.DataLoader(d_train, batch_size=32, shuffle=True)
test_data = data.DataLoader(d_test, batch_size=32, shuffle=False)

for embeddings, labels in train_data:
    print(embeddings.shape)  # Ожидаем (batch_size, embedding_dim)
    print(labels.shape)      # Ожидаем (batch_size,)
    break


