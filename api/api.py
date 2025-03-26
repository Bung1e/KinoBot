from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from model.model import KinoRNN
from model.data import TextTokenizer
import torch  
import numpy as np
from config_api import TMDB_API_KEY
import requests

app = FastAPI()
class TextRequest(BaseModel):
    text: str

class PredictionResponse(BaseModel):
    movies: list[dict]

emotion_labels = {
    0: 'sad',
    1: 'joy',
    2: 'love',
    3: 'angry',
    4: 'fear',
    5: 'surprise'
}

MOOD_TO_GENRE = {
    'sad': [18, 35], 
    'joy': [35, 10749], 
    'love': [10749, 18],
    'angry': [28, 53],
    'fear': [27, 53],
    'surprise': [878, 9648] 
}

def get_movies_by_mood(mood, api_key):
    genres = MOOD_TO_GENRE.get(mood)
    url = "https://api.themoviedb.org/3/discover/movie"
    params = {
        "api_key": TMDB_API_KEY,
        "with_genres": f"{genres[0]}|{genres[1]}",
        "sort_by": "popularity.desc",
        "language": "en-EN",
        "page": 1
    }
    
    response = requests.get(url, params=params)
    movies = response.json().get('results', [])
    
    return movies[:3]

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

model_path = 'model/best_model_simple_rnn.pt'
tokenizer = TextTokenizer(max_length=128)
vocab_size = tokenizer.tokenizer.vocab_size

model, device = load_model(model_path, vocab_size)

@app.post("/films", response_model=PredictionResponse)
async def films(input_data: TextRequest):
        tokens = tokenizer.tokenize(input_data.text)
        tokens = tokens.unsqueeze(0).to(device)

        with torch.no_grad():
            outputs = model(tokens)
            probabilities = torch.nn.functional.softmax(outputs, dim=1)[0].cpu().numpy()

        top_indices = probabilities.argsort()[-2:][::-1]
        top_emotions = [emotion_labels[idx] for idx in top_indices]
        print(top_emotions)

        movies = get_movies_by_mood(top_emotions[0], api_key=TMDB_API_KEY)

        return {"movies": movies}

if __name__ == "__main__":
    import uvicorn
    uvicorn.run("api:app", host="0.0.0.0", port=8000, reload=True)