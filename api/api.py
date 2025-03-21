from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from model.model import KinoRNN
from model.data import TextTokenizer
import torch  
import numpy as np

app = FastAPI()
class TextRequest(BaseModel):
    text: str

class PredictionResponse(BaseModel):
    mood: dict

emotion_labels = {
    0: 'sad',
    1: 'joy',
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

model_path = 'model/best_model_simple_rnn.pt'
tokenizer = TextTokenizer(max_length=128)
vocab_size = tokenizer.tokenizer.vocab_size

model, device = load_model(model_path, vocab_size)

@app.post("/predict", response_model=PredictionResponse)
async def predict(input_data: TextRequest):
        tokens = tokenizer.tokenize(input_data.text)
        tokens = tokens.unsqueeze(0).to(device)

        with torch.no_grad():
            outputs = model(tokens)
            probabilities = torch.nn.functional.softmax(outputs, dim=1)[0].cpu().numpy()

        top_indices = probabilities.argsort()[-2:][::-1]
        top_emotions = {emotion_labels[idx]: float(round(probabilities[idx] * 100, 2)) for idx in top_indices}

        return {"mood": top_emotions}

if __name__ == "__main__":
    import uvicorn
    uvicorn.run("api:app", host="0.0.0.0", port=8000, reload=True)