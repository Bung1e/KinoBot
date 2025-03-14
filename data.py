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


df = pd.read_csv('datasets/cleaned_data.csv')
nlp = spacy.load("en_core_web_sm")


def preprocess_text(text):
    doc = nlp(text.lower())
    tokens = [token.lemma_ for token in doc if not token.is_stop and token.is_alpha]
    return tokens

def get_embedding(tokens):
    doc = nlp(" ".join(tokens))
    return doc.vector

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

if __name__ == '__main__':

    df_processed = parallel_processing(df, chunk_size=10000) 

    df_processed.to_csv("processed_data.csv", index=False)

    np.save("embeddings.npy", np.vstack(df_processed['embedding'].values)) 
