import os
from config import BOT_TOKEN
from aiogram import Bot, Dispatcher, types
from aiogram.types import Message
from aiogram.filters import Command
from config import headers
import asyncio
import torch 
import logging
from transformers import pipeline
import requests
from model import TextTokenizer, KinoRNN

logging.basicConfig(level=logging.INFO)

# TOKEN = os.getenv("BOT_TOKEN")
# bot = Bot(token=TOKEN)
# dp = Dispatcher()

emotion_labels = {
    0: 'sad',
    1: 'joi',
    2: 'love',
    3: 'angry',
    4: 'fear',
    5: 'surprise'
}

def load_model(model_path, vocab_size):
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    model = KinoRNN(
        vocab_size=vocab_size,
        embedding_dim=128,
        hidden_size=256,
        num_layers=1,
        num_classes=6
    )
    
    checkpoint = torch.load(model_path, map_location=device)
    model.load_state_dict(checkpoint['model_state_dict'])
    model.to(device)
    model.eval()
    
    return model, device

def predict_emotion(text, model, tokenizer, device):
    tokens = tokenizer.tokenize(text)
    tokens = tokens.unsqueeze(0).to(device)
    
    with torch.no_grad():
        outputs = model(tokens)
        _, predicted = torch.max(outputs, 1)
        predicted_class = predicted.item()
    
    emotion = emotion_labels[predicted_class]
    
    probabilities = torch.nn.functional.softmax(outputs, dim=1)[0]
    confidence = probabilities[predicted_class].item() * 100
    
    return emotion, confidence, probabilities.cpu().numpy()

def main():
    model_path = 'model/best_model_simple_rnn.pt'
    
    tokenizer = TextTokenizer(max_length=128)
    vocab_size = tokenizer.tokenizer.vocab_size
    
    model, device = load_model(model_path, vocab_size)
    print(f"model loaded successfully {device}")
    
    while True:
        user_input = input("\ntext: ")
        
        if user_input.lower() == 'q':
            break
        
        emotion, confidence, probabilities = predict_emotion(user_input, model, tokenizer, device)
        
        print(f"\nEmotion: {emotion.upper()} (confidence: {confidence:.2f}%)")
        
        print("\nProbabilities:")
        for i, prob in enumerate(probabilities):
            print(f"{emotion_labels[i]}: {prob*100:.2f}%")

if __name__ == "__main__":
    main()
    