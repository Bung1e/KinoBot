import os
from config import BOT_TOKEN
from aiogram import Bot, Dispatcher, types
from aiogram.types import Message
from aiogram.filters import Command
from config import BOT_TOKEN
import asyncio
from dotenv import load_dotenv
import os
import logging
from transformers import pipeline

load_dotenv()

logging.basicConfig(level=logging.INFO)

TOKEN = os.getenv("BOT_TOKEN")
bot = Bot(token=TOKEN)
dp = Dispatcher()

# classifier = pipeline("zero-shot-classification", model="facebook/bart-large-mnli")
classifier = pipeline("zero-shot-classification", model="MoritzLaurer/DeBERTa-v3-base-mnli-fever-anli")
emotion_classifier = pipeline("text-classification", model="j-hartmann/emotion-english-distilroberta-base")

def analyze_sentiment(message):
    candidate_labels = ['comedy', 'drama', 'action', 'romantic', 'thriller', 'happy', 'sad', 'exciting', 'relaxing']
    result = classifier(message, candidate_labels)
    emotion_result = emotion_classifier(message)
    answer = result['labels'][:3]
    return answer

@dp.message(Command("start"))
async def start_handler(message: Message):
    await message.answer("Hello, i will help you with film recommendation")

@dp.message()
async def echo_handler(message: Message):
    response = analyze_sentiment(message.text)
    await message.reply(", ".join(response))

async def main():
    await dp.start_polling(bot)

if __name__ == "__main__":
    asyncio.run(main())