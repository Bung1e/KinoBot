import pandas as pd
import numpy as np
from transformers import AutoTokenizer
from collections import Counter
from tqdm import tqdm
import torch
from torch.utils.data import Dataset
import json
import os

class TextTokenizer:
    def __init__(self, max_length=128):
        self.tokenizer = AutoTokenizer.from_pretrained('bert-base-uncased')
        self.max_length = max_length
        
    def tokenize(self, text):
        # Токенизируем текст с помощью BERT токенизатора
        encoded = self.tokenizer(
            text,
            max_length=self.max_length,
            padding='max_length',
            truncation=True,
            return_tensors='pt'
        )
        return encoded['input_ids'].squeeze(0)

class KinoDataset(Dataset):
    def __init__(self, df, tokenizer):
        self.df = df
        self.tokenizer = tokenizer
        
    def __len__(self):
        return len(self.df)
    
    def __getitem__(self, idx):
        text = self.df.iloc[idx]['text']
        label = self.df.iloc[idx]['label']
        
        # Токенизируем текст
        tokens = self.tokenizer.tokenize(text)
        
        return {
            'input_ids': tokens,
            'label': torch.tensor(label, dtype=torch.long)
        }

def prepare_dataset(csv_path, max_length=128):
    """
    Подготовка датасета с новой токенизацией
    """
    print("Loading dataset...")
    df = pd.read_csv(csv_path)
    
    # Создаем токенизатор
    tokenizer = TextTokenizer(max_length=max_length)
    
    # Создаем датасет
    dataset = KinoDataset(df, tokenizer)
    
    # Получаем размер словаря
    vocab_size = tokenizer.tokenizer.vocab_size
    
    print(f"Dataset prepared. Vocabulary size: {vocab_size}")
    return dataset, vocab_size

if __name__ == '__main__':
    # Пример использования
    dataset, vocab_size = prepare_dataset('datasets/processed_data_sampled.csv')
    print(f"Vocabulary size: {vocab_size}")
    print(f"Dataset size: {len(dataset)}")
    
    # Проверяем первый элемент
    sample = dataset[0]
    print(f"Sample input shape: {sample['input_ids'].shape}")
    print(f"Sample label: {sample['label']}") 