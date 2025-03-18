import pandas as pd
import numpy as np
from transformers import AutoTokenizer
import torch
from torch.utils.data import Dataset
import json
import os
import random

class TextTokenizer():
    def __init__(self, max_length=128):
        self.tokenizer = AutoTokenizer.from_pretrained('bert-base-uncased')
        self.max_length = max_length

    def tokenize(self, text):
        if len(text) < 20:
            text = text + " " + text
        
        encoded = self.tokenizer(
            text,
            max_length=self.max_length,
            padding='max_length',
            truncation=True,
            return_tensors='pt'
        )
        return encoded["input_ids"].squeeze(0)

class KinoDataset(Dataset):
    def __init__(self, df, tokenizer):
        self.df = df
        self.tokenizer = tokenizer

    def __getitem__(self, idx):
        text = self.df.iloc[idx]['text']
        label = self.df.iloc[idx]['label']
        
        if label == 5 and random.random() < 0.5:
            words = text.split()
            if len(words) > 3:
                i, j = random.sample(range(len(words)), 2)
                words[i], words[j] = words[j], words[i]
                text = ' '.join(words)
        
        tokens = self.tokenizer.tokenize(text)
        return {
            'input_ids': tokens,
            'label': torch.tensor(label, dtype=torch.long)
        }

    def __len__(self):
        return len(self.df)

def prepare_data(csv_path, max_length=128):
    df = pd.read_csv(csv_path)

    tokenizer = TextTokenizer(max_length=max_length)
    data = KinoDataset(df, tokenizer)
    vocab_size = tokenizer.tokenizer.vocab_size
    
    return data, vocab_size

if __name__ == '__main__':
    data, vocab_size = prepare_data('datasets/end_data.csv')
    print(f"Dataset size: {len(data)}")
    sample = data[0]
    print(f"Sample input shape: {sample['input_ids'].shape}")
    print(f"Sample label: {sample['label']}")






