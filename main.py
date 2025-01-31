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

sentiment_analyzer = pipeline(model="r1char9/rubert-base-cased-russian-sentiment")

def analyze_sentiment(message):
    result = sentiment_analyzer(message)
    return result[0]['label']

@dp.message(Command("start"))
async def start_handler(message: Message):
    await message.answer("Привет! Я помогу тебе подобрать фильм. Напиши, что ты ищешь!")

@dp.message()
async def echo_handler(message: Message):
    sentiment = analyze_sentiment(message.text)
    if sentiment == 'positive':
        response = 'Позитивное сообщение'
    else:
        response = 'Негативное сообщение'
    await message.reply(response)

async def main():
    await dp.start_polling(bot)

if __name__ == "__main__":
    asyncio.run(main())