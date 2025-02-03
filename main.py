import os
from config import BOT_TOKEN
from aiogram import Bot, Dispatcher, types
from aiogram.types import Message
from aiogram.filters import Command
from config import headers
import asyncio
from dotenv import load_dotenv
import os
import logging
from transformers import pipeline
import requests

load_dotenv()

logging.basicConfig(level=logging.INFO)

TOKEN = os.getenv("BOT_TOKEN")
bot = Bot(token=TOKEN)
dp = Dispatcher()

# classifier = pipeline("zero-shot-classification", model="facebook/bart-large-mnli")
classifier = pipeline("zero-shot-classification", model="MoritzLaurer/DeBERTa-v3-base-mnli-fever-anli")
emotion_classifier = pipeline("text-classification", model="j-hartmann/emotion-english-distilroberta-base")

def analyze_sentiment(message):
    candidate_labels = ['comedy', 'drama', 'action', 'adventure', 'crime', 'family', 'fantasy', 'history', 'thriller', 'horror', 'music', 'mystery','tv Movie', 'war', 'science fiction', 'romantic']  #deleted western
    result = classifier(message, candidate_labels)
    # emotion_result = emotion_classifier(message)
    answer = result['labels'][:2]
    print(answer)
    return answer


def get_data(genres):
    genre_ids = {
        'action': 28,
        'adventure': 12,
        'animation': 16,
        'comedy': 35,
        'crime': 80,
        'documentary': 99,
        'drama ': 18,
        'family': 10751,
        'fantasy': 14,
        'history': 36,
        'horror': 27,
        'music': 10402,
        'mystery': 9648,
        'romantic': 10749,
        'science fiction': 878,
        'tv Movie': 10770,
        'thriller': 53,
        'war': 10752,
        'western': 37
    }

    searchig_genres = [genre_ids[genre] for genre in genres]
    url = f'https://api.themoviedb.org/3/discover/movie?page=1&sort_by=popularity.desc&with_genres={searchig_genres[0]}%2C%20{searchig_genres[1]}'
    response = requests.get(url, headers=headers)
    data = response.json()
    print(searchig_genres)
    titles = [movie['original_title'] for movie in data['results'][:3]]
    return titles

@dp.message(Command("start"))
async def start_handler(message: Message):
    await message.answer("Hello, i will help you with film recommendation")

@dp.message()
async def echo_handler(message: Message):
    genres = analyze_sentiment(message.text)
    response = get_data(genres)
    await message.reply(", ".join(response))

async def main():
    await dp.start_polling(bot)

if __name__ == "__main__":
    asyncio.run(main())