import os
import pandas as pd
import numpy as np
from tqdm import tqdm
import torch
import torch.utils.data as data
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from data import prepare_data

class KinoRNN(nn.Module):
    def __init__(self, vocab_size, embedding_dim, hidden_size, num_layers, num_classes):
        super().__init__()
        self.hidden_size = hidden_size
        self.num_layers = num_layers

        self.embedding = nn.Embedding(vocab_size, embedding_dim)

        self.lstm = nn.LSTM(
            input_size=embedding_dim,
            hidden_size=self.hidden_size,
            num_layers=self.num_layers,
            batch_first=True,
            bidirectional=True)
        self.out = nn.Linear(self.hidden_size * 2, num_classes)

    def forward(self, x):
        device = x.device
        
        embedded = self.embedding(x)
        batch_size = x.size(0)
        
        h0 = torch.zeros(self.num_layers * 2, batch_size, self.hidden_size).to(device)
        c0 = torch.zeros(self.num_layers * 2, batch_size, self.hidden_size).to(device)
        
        lstm_out, (h, c) = self.lstm(embedded, (h0, c0))
        
        out = self.out(lstm_out[:, -1, :])
        return out

def train_model(model, train_loader, val_loader, epochs=20, device='cuda'):
    model = model.to(device)

    optimizer = optim.Adam(params=model.parameters(), lr=0.001, weight_decay=0.01)
    loss_function = nn.CrossEntropyLoss()  

    best_accuracy = 0.0
    patience = 3
    no_improve = 0

    for epoch in range(epochs):
        model.train()
        train_loss = 0
        train_total = 0
        train_correct = 0

        train_tqdm = tqdm(train_loader, desc=f'Epoch {epoch+1}/{epochs}')
        for batch in train_tqdm:
            input_ids = batch['input_ids'].to(device)
            labels = batch['label'].to(device)

            optimizer.zero_grad()
            predict = model(input_ids)
            loss = loss_function(predict, labels)

            loss.backward()
            optimizer.step()

            train_loss += loss.item()
            _, predicted = torch.max(predict.data, 1)
            train_total += labels.size(0)
            train_correct += (predicted == labels).sum().item()
        
            train_tqdm.set_description(f'Epoch {epoch+1}/{epochs} - Loss: {loss.item():.4f}')
        
        avg_train_loss = train_loss / len(train_loader)
        train_accuracy = train_correct / train_total

        model.eval()
        val_loss = 0
        val_correct = 0
        val_total = 0
        
        with torch.no_grad():
            for batch in val_loader:
                input_ids = batch['input_ids'].to(device)
                labels = batch['label'].to(device)
                
                outputs = model(input_ids)
                loss = loss_function(outputs, labels)
                
                val_loss += loss.item()
                _, predicted = torch.max(outputs.data, 1)
                val_total += labels.size(0)
                val_correct += (predicted == labels).sum().item()
        
        avg_val_loss = val_loss / len(val_loader)
        val_accuracy = val_correct / val_total
        
        print(f'Epoch {epoch+1}:')
        print(f'Training Loss: {avg_train_loss:.4f}, Training Accuracy: {train_accuracy:.4f}')
        print(f'Validation Loss: {avg_val_loss:.4f}, Validation Accuracy: {val_accuracy:.4f}')
    
    st = model.state_dict()
    torch.save('best_model_rnn.pt', st)

if __name__ == '__main__':

    data, vocab_size = prepare_data('datasets/end_data.csv')

    train_size = int(0.8 * len(data))
    val_size = len(data) - train_size
    train_dataset, val_dataset = torch.utils.data.random_split(data, [train_size, val_size])

    train_data = DataLoader(train_dataset, batch_size=32, shuffle=True)
    val_data = DataLoader(val_dataset, batch_size=32, shuffle=False)

    model = KinoRNN(
        vocab_size=vocab_size,
        embedding_dim=768,
        hidden_size=128,
        num_layers=1,
        num_classes=6)

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    train_model(model, train_data, val_data, epochs=20, device=device)

