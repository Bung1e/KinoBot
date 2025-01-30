import os
from config import BOT_TOKEN
from aiogram import Bot, Dispatcher, types
from aiogram.types import Message
from aiogram.filters import Command
from config import BOT_TOKEN
import asyncio
from dotenv import load_dotenv
import os

load_dotenv()

TOKEN = os.getenv("BOT_TOKEN")
bot = Bot(token=TOKEN)
dp = Dispatcher()

@dp.message(Command("start"))
async def start_handler(message: Message):
    await message.answer("Привет! Я помогу тебе подобрать фильм. Напиши, что ты ищешь!")

@dp.message()
async def echo_handler(message: Message):
    await message.answer(f"Ты написал: {message.text}")

async def main():
    await dp.start_polling(bot)

if __name__ == "__main__":
    asyncio.run(main())