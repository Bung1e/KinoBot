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
import random

class KinoRNN(nn.Module):
    def __init__(self, vocab_size, embedding_dim=128, hidden_size=256, num_layers=2, num_classes=6):
        super().__init__()
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        
        self.embedding = nn.Embedding(vocab_size, embedding_dim)
        nn.init.normal_(self.embedding.weight, mean=0, std=0.01)
        
        self.embed_dropout = nn.Dropout(0.1)
        self.embed_norm = nn.LayerNorm(embedding_dim)
        
        self.lstm = nn.LSTM(
            input_size=embedding_dim,
            hidden_size=hidden_size,
            num_layers=num_layers,
            batch_first=True,
            bidirectional=True,
            dropout=0.1 if num_layers > 1 else 0
        )
        
        self.attention = nn.Linear(hidden_size * 2, 1)
        
        self.dropout = nn.Dropout(0.1)
        self.norm = nn.LayerNorm(hidden_size * 2)
        self.fc1 = nn.Linear(hidden_size * 2, hidden_size)
        self.relu = nn.ReLU()
        self.fc2 = nn.Linear(hidden_size, num_classes)
        
        nn.init.xavier_normal_(self.fc1.weight)
        nn.init.xavier_normal_(self.fc2.weight)

    def forward(self, x):
        device = x.device
        
        embedded = self.embedding(x)
        embedded = self.embed_dropout(embedded)
        embedded = self.embed_norm(embedded)
        
        h0 = torch.zeros(self.num_layers * 2, x.size(0), self.hidden_size).to(device)
        c0 = torch.zeros(self.num_layers * 2, x.size(0), self.hidden_size).to(device)
        
        lstm_out, _ = self.lstm(embedded, (h0, c0))
        
        attention_weights = torch.softmax(self.attention(lstm_out), dim=1)
        out = torch.sum(attention_weights * lstm_out, dim=1)
        
        out = self.norm(out)
        out = self.dropout(out)
        out = self.fc1(out)
        out = self.relu(out)
        out = self.dropout(out)
        out = self.fc2(out)
        
        return out

def train_model(model, train_loader, val_loader, epochs=20, device='cuda'):
    model = model.to(device)
    
    class_weights = torch.tensor([1.2, 1.0, 1.2, 1.2, 1.1, 1.5])
    class_weights = class_weights / class_weights.sum()
    class_weights = class_weights.to(device)
    
    optimizer = optim.AdamW(
        model.parameters(),
        lr=0.001,
        weight_decay=0.01,
        betas=(0.9, 0.999)
    )
    
    scheduler = optim.lr_scheduler.OneCycleLR(
        optimizer,
        max_lr=0.001,
        epochs=epochs,
        steps_per_epoch=len(train_loader),
        pct_start=0.1,
        div_factor=10.0
    )
    
    loss_function = nn.CrossEntropyLoss(weight=class_weights)
    
    best_accuracy = 0.0
    patience = 5
    no_improve = 0

    for epoch in range(epochs):
        model.train()
        train_loss = 0
        train_total = 0
        train_correct = 0
        
        train_tqdm = tqdm(train_loader, desc=f'Epoch {epoch+1}/{epochs}')
        for i, batch in enumerate(train_tqdm):
            input_ids = batch['input_ids'].to(device)
            labels = batch['label'].to(device)
            
            if random.random() < 0.3:
                idx = torch.randperm(input_ids.size(0))
                input_ids = torch.cat([input_ids, input_ids[idx]], dim=0)
                labels = torch.cat([labels, labels[idx]], dim=0)
            
            outputs = model(input_ids)
            loss = loss_function(outputs, labels)
            
            optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            optimizer.step()
            scheduler.step()
            
            train_loss += loss.item()
            _, predicted = torch.max(outputs.data, 1)
            train_total += labels.size(0)
            train_correct += (predicted == labels).sum().item()
            
            train_tqdm.set_description(
                f'Epoch {epoch+1}/{epochs} - Loss: {loss.item():.4f}'
            )
        
        avg_train_loss = train_loss / len(train_loader)
        train_accuracy = train_correct / train_total
        
        model.eval()
        val_loss = 0
        val_correct = 0
        val_total = 0
        class_correct = torch.zeros(6).to(device)
        class_total = torch.zeros(6).to(device)
        
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
                
                for i in range(6):
                    mask = (labels == i)
                    class_correct[i] += (predicted[mask] == labels[mask]).sum().item()
                    class_total[i] += mask.sum().item()
        
        avg_val_loss = val_loss / len(val_loader)
        val_accuracy = val_correct / val_total
        
        print(f'\nEpoch {epoch+1}:')
        print(f'Training Loss: {avg_train_loss:.4f}, Training Accuracy: {train_accuracy:.4f}')
        print(f'Validation Loss: {avg_val_loss:.4f}, Validation Accuracy: {val_accuracy:.4f}')
        print(f'Learning rate: {optimizer.param_groups[0]["lr"]:.6f}')
        print('\nAccuracy by class:')
        for i in range(6):
            class_acc = class_correct[i] / class_total[i] if class_total[i] > 0 else 0
            print(f'Class {i}: {class_acc:.4f}')
        
        if val_accuracy > best_accuracy:
            best_accuracy = val_accuracy
            torch.save({
                'epoch': epoch,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'scheduler_state_dict': scheduler.state_dict(),
                'best_accuracy': best_accuracy,
            }, 'best_model_rnn.pt')
            no_improve = 0
        else:
            no_improve += 1
            if no_improve >= patience:
                print(f'Early stopping triggered after {epoch+1} epochs')
                break
    
    return best_accuracy

if __name__ == '__main__':
    data, vocab_size = prepare_data('datasets/end_data.csv')

    train_size = int(0.8 * len(data))
    val_size = len(data) - train_size
    train_dataset, val_dataset = torch.utils.data.random_split(data, [train_size, val_size])

    train_data = DataLoader(train_dataset, batch_size=64, shuffle=True)
    val_data = DataLoader(val_dataset, batch_size=64, shuffle=False)

    model = KinoRNN(
        vocab_size=vocab_size,
        embedding_dim=256,
        hidden_size=128,
        num_layers=2,
        num_classes=6
    )

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    train_model(model, train_data, val_data, epochs=20, device=device)

