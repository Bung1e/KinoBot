import os
import pandas as pd
import numpy as np
from tqdm import tqdm
import torch
import torch.utils.data as data
import torch.nn as nn
import torch.optim as optim
from sklearn.model_selection import train_test_split

class KinoRNN(nn.Module):
    def __init__(self, in_features, out_features):
        super().__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.hidden_size = 96

        self.rnn = nn.LSTM(in_features, self.hidden_size, batch_first=True)
        self.out = nn.Linear(self.hidden_size, self.out_features)

    def forward(self, x):
        h0 = torch.zeros(1, x.size(0), self.hidden_size)
        c0 = torch.zeros(1, x.size(0), self.hidden_size)
        x, (h, c) = self.rnn(x, (h0, c0))
        y = self.out(x[:, -1, :])
        return y

class KinoDataset(data.Dataset):
    def __init__(self, df):
        self.df = df

    def __getitem__(self, idx):
        embedding = torch.tensor(self.df.iloc[idx]["embedding"], dtype=torch.float32)
        label = torch.tensor(self.df.iloc[idx]["label"], dtype=torch.long)
        return embedding, label

    def __len__(self):
        return len(self.df)

def parse_embedding(embedding_str):
    embedding_array = np.fromstring(embedding_str.strip("[]"), sep=" ")
    return torch.tensor(embedding_array, dtype=torch.float32)

df = pd.read_csv('datasets/end_data.csv')
df["embedding"] = df["embedding"].apply(parse_embedding)
df_train, df_test = train_test_split(df, test_size=0.2, random_state=52)

d_train = KinoDataset(df_train)
d_test = KinoDataset(df_test)
BATCH_SIZE = 32
train_data = data.DataLoader(d_train, batch_size=BATCH_SIZE, shuffle=True)
test_data = data.DataLoader(d_test, batch_size=BATCH_SIZE, shuffle=False)

num_classes = 6
model = KinoRNN(96, num_classes)
optimizer = optim.Adam(params=model.parameters(), lr=0.001, weight_decay=0.01)
loss_function = nn.CrossEntropyLoss()

epochs = 20
for epoch in range(epochs):
    model.train()
    train_loss = 0
    train_tqdm = tqdm(train_data, leave=True)
    for x_train, y_train in train_tqdm:
        x_train = x_train.unsqueeze(1)
        predict = model(x_train)
        loss = loss_function(predict, y_train.long())

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        train_loss += loss.item()
        train_tqdm.set_description(f"Epoch {epoch+1}/{epochs} - Loss: {loss.item():.4f}")

    avg_train_loss = train_loss / len(train_data)
    print(f"Epoch {epoch+1}: Average Train Loss: {avg_train_loss:.4f}")

    model.eval()
    val_loss = 0
    correct = 0
    total = 0
    with torch.no_grad():
        for x_val, y_val in test_data:
            x_val = x_val.unsqueeze(1)
            y_pred = model(x_val)
            loss = loss_function(y_pred, y_val.long())
            val_loss += loss.item()
            correct += (y_pred.argmax(dim=1) == y_val).sum().item()
            total += len(y_val)

    avg_val_loss = val_loss / len(test_data)
    accuracy = correct / total
    print(f"Validation Accuracy: {accuracy:.4f}")
    print(f"Epoch {epoch+1}: Average Validation Loss: {avg_val_loss:.4f}")

torch.save(model.state_dict(), 'model_rnn_words.tar')