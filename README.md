# Emotion-Based Movie Recommender Bot

A Telegram bot that analyzes user's emotional state from text and recommends movies that match their mood.

## Features

- Text-based emotion analysis
- Movie recommendations based on emotions
- TMDB API integration for movie information
- REST API for emotion analysis
- Docker containerization

## Technologies

- Python 3.12
- PyTorch
- FastAPI
- aiogram 3.x
- Docker
- TMDB API

## Model and Training

### Model Architecture

The model is a RNN (Recurrent Neural Network) with LSTM layers, trained on the dair-ai/emotion dataset. Key features:

- Input layer: tokenized text
- Embedding layer: 128 dimensions
- LSTM layer: 256 hidden units
- Output layer: 6 emotions (sad, joy, love, angry, fear, surprise)

### Training Process

1. Data Preparation:
   - Text tokenization
   - Vocabulary creation
   - Sentence vectorization

2. Architecture:
```python
class KinoRNN(nn.Module):
    def __init__(self, vocab_size, embedding_dim, hidden_size, num_layers, num_classes):
        super(KinoRNN, self).__init__()
        self.embedding = nn.Embedding(vocab_size, embedding_dim)
        self.lstm = nn.LSTM(embedding_dim, hidden_size, num_layers, batch_first=True)
        self.fc = nn.Linear(hidden_size, num_classes)
```

3. Hyperparameters:
   - Batch size: 32
   - Vocabulary size: 10000
   - Embedding dimension: 128
   - Hidden layer size: 256
   - Number of LSTM layers: 1
   - Learning rate: 0.001
   - Optimizer: Adam

4. Metrics:
   - Validation accuracy: ~85%
   - Loss: categorical_crossentropy

### Dataset

The model was trained on the dair-ai/emotion dataset from Hugging Face, which contains:
- 20k text examples
- 6 emotional categories
- Clean, balanced dataset for emotion classification

## Installation and Setup

### Prerequisites

- Docker and Docker Compose
- TMDB API key
- Telegram Bot Token

### Setup

1. Clone the repository:
```bash
git clone https://github.com/bung1e/KinoBot.git
cd KinoBot
```

2. Create a `.env` file in the root directory:
```env
BOT_TOKEN=your_telegram_bot_token
API_URL=http://localhost:8000
TMDB_API_KEY=your_tmdb_api_key
```


## Usage

1. Find the bot in Telegram by its username
2. Send `/start` command to begin
3. Use `/films` command and write text describing your mood
4. Get movie recommendations matching your emotional state

## API Endpoints

- `POST /films` - analyze text and get movie recommendations
- `GET /docs` - Swagger API documentation

## Project Structure

```
.
├── api/
│   └── api.py          # FastAPI application
├── bot/
│   └── bot.py          # Telegram bot
├── model/
│   ├── model.py        # Emotion analysis model
│   └── data.py         # Data processing
├── docker-compose.yml  # Docker Compose configuration
├── Dockerfile         # Dockerfile for image building
├── requirements.txt   # Project dependencies
└── .env              # Environment variables
```

## Development

For local development:

1. Create a virtual environment:
```bash
python -m venv venv
source venv/bin/activate  # for Linux/Mac
venv\Scripts\activate     # for Windows
```

2. Install dependencies:
```bash
pip install -r requirements.txt
```

3. Run API locally:
```bash
uvicorn api.api:app --reload
```

4. Run bot:
```bash
python bot/bot.py
```
