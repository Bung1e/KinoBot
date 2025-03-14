import pandas as pd
import os
import numpy as np
import re
# import matplotlib.pyplot as plt
from tqdm import tqdm
import torch
from torch.utils.data import BatchSampler, SequentialSampler
import torch.utils.data as data
import torchvision
from torchvision import models
import torchvision.transforms.v2 as tfs_v2
import torch.nn as nn
import torch.optim as optim
import spacy
from multiprocessing import Pool
from sklearn.model_selection import train_test_split

df = pd.read_csv('datasets/cleaned_data.csv')
nlp = spacy.load("en_core_web_sm")


def preprocess_text(text):
    doc = nlp(text.lower())
    tokens = [token.lemma_ for token in doc if not token.is_stop and token.is_alpha]
    return tokens


def chunk_data(df, chunk_size):
    chunks = [df.iloc[i:i + chunk_size] for i in range(0, len(df), chunk_size)]
    return chunks

def process_chunk(chunk):
    chunk['tokens'] = chunk['text'].apply(preprocess_text)
    chunk['embedding'] = chunk['tokens'].apply(get_embedding) 
    return chunk[['text', 'tokens', 'embedding']]

def parallel_processing(df, chunk_size):
    chunks = chunk_data(df, chunk_size)  
    with Pool(processes=4) as pool:  
        result = pool.map(process_chunk, chunks)  
    return pd.concat(result, ignore_index=True)  

def sample_class_data(group, samples_per_class):
    num_samples = min(samples_per_class, len(group))
    return group.sample(n=num_samples, random_state=42)

def get_embedding(text):
    try:
        doc = nlp(" ".join(text)) 
        if doc.vector is not None and doc.vector.any():
            return doc.vector
        else:
            return np.zeros(96)
    except Exception as e:
        print(f"Ошибка при обработке текста: {e}")
        return np.zeros(96)

if __name__ == '__main__':
    df = pd.read_csv('datasets/cleaned_data.csv')
    min_class_count = df['label'].value_counts().min()
    desired_samples_per_class = 150000 // len(df['label'].value_counts())

    df_sampled = df.groupby('label').apply(lambda x: sample_class_data(x, desired_samples_per_class))
    df_sampled = df_sampled.reset_index(drop=True) #delete multi indexes after group_by
    print(df_sampled['label'].value_counts())

    df_processed = parallel_processing(df_sampled, chunk_size=50000)
    df_processed = df_processed[df_processed['embedding'].apply(lambda x: np.any(x != 0))]
    df_processed.to_csv("processed_data_sampled.csv", index=False)
    np.save("embeddings_sampled.npy", np.vstack(df_processed['embedding'].values))
