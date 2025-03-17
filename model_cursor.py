import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from tqdm import tqdm
from tokenizer import prepare_dataset

class KinoRNN(nn.Module):
    def __init__(self, vocab_size, embedding_dim=768, hidden_size=256, num_layers=2, dropout=0.3, num_classes=6):
        super().__init__()
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        self.dropout_rate = dropout
        
        # Embedding layer
        self.embedding = nn.Embedding(vocab_size, embedding_dim)
        
        # Batch normalization
        self.batch_norm = nn.BatchNorm1d(embedding_dim)
        
        # LSTM layers
        self.lstm = nn.LSTM(
            input_size=embedding_dim,
            hidden_size=hidden_size,
            num_layers=num_layers,
            batch_first=True,
            dropout=dropout if num_layers > 1 else 0,
            bidirectional=True
        )
        
        # Dropout layer
        self.dropout = nn.Dropout(dropout)
        
        # Output layers
        self.fc1 = nn.Linear(hidden_size * 2, hidden_size)  # *2 for bidirectional
        self.relu = nn.ReLU()
        self.fc2 = nn.Linear(hidden_size, num_classes)

    def forward(self, x):
        # Get device of input tensor
        device = x.device
        
        # Forward pass through embedding
        embedded = self.embedding(x)
        
        # Reshape for batch norm
        batch_size, seq_len, embedding_dim = embedded.size()
        embedded = embedded.view(-1, embedding_dim)
        embedded = self.batch_norm(embedded)
        embedded = embedded.view(batch_size, seq_len, embedding_dim)
        
        # Initialize hidden state
        h0 = torch.zeros(self.num_layers * 2, batch_size, self.hidden_size).to(device)
        c0 = torch.zeros(self.num_layers * 2, batch_size, self.hidden_size).to(device)
        
        # LSTM forward pass
        lstm_out, (hidden, cell) = self.lstm(embedded, (h0, c0))
        
        # Get the output from the last time step
        out = lstm_out[:, -1, :]
        
        # Fully connected layers
        out = self.dropout(out)
        out = self.fc1(out)
        out = self.relu(out)
        out = self.dropout(out)
        out = self.fc2(out)
        
        return out

def train_model(model, train_loader, val_loader, epochs=20, device='cuda'):
    model = model.to(device)
    optimizer = optim.Adam(model.parameters(), lr=0.001, weight_decay=0.01)
    loss_function = nn.CrossEntropyLoss()
    
    best_accuracy = 0.0
    patience = 3
    no_improve = 0
    
    for epoch in range(epochs):
        # Training
        model.train()
        train_loss = 0
        train_correct = 0
        train_total = 0
        
        train_tqdm = tqdm(train_loader, desc=f'Epoch {epoch+1}/{epochs}')
        for batch in train_tqdm:
            input_ids = batch['input_ids'].to(device)
            labels = batch['label'].to(device)
            
            optimizer.zero_grad()
            outputs = model(input_ids)
            loss = loss_function(outputs, labels)
            
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            optimizer.step()
            
            train_loss += loss.item()
            _, predicted = torch.max(outputs.data, 1)
            train_total += labels.size(0)
            train_correct += (predicted == labels).sum().item()
            
            train_tqdm.set_description(f'Epoch {epoch+1}/{epochs} - Loss: {loss.item():.4f}')
        
        avg_train_loss = train_loss / len(train_loader)
        train_accuracy = train_correct / train_total
        
        # Validation
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
        
        # Early stopping
        if val_accuracy > best_accuracy:
            best_accuracy = val_accuracy
            torch.save({
                'epoch': epoch,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
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
    # Подготовка данных
    dataset, vocab_size = prepare_dataset('datasets/processed_data_sampled.csv')
    
    # Разделение на train и validation
    train_size = int(0.8 * len(dataset))
    val_size = len(dataset) - train_size
    train_dataset, val_dataset = torch.utils.data.random_split(dataset, [train_size, val_size])
    
    # Создание даталоадеров
    train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=32, shuffle=False)
    
    # Инициализация модели
    model = KinoRNN(
        vocab_size=vocab_size,
        embedding_dim=768,  # Размерность BERT эмбеддингов
        hidden_size=256,
        num_layers=2,
        dropout=0.3,
        num_classes=6
    )
    
    # Обучение модели
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    best_accuracy = train_model(model, train_loader, val_loader, epochs=20, device=device)
    print(f'Best validation accuracy: {best_accuracy:.4f}')