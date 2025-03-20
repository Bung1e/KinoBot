from aiogram import Bot, Dispatcher, types
from aiogram.types import Message
from aiogram.filters import CommandStart, Command
import asyncio
from config import BOT_TOKEN, API_URL
import requests

bot = Bot(token=BOT_TOKEN)
dp = Dispatcher()

@dp.message(CommandStart())
async def cmd_start(message: Message):
    await message.answer("Hello!")

@dp.message(Command('predict'))
async def analyze_message(message: Message):
    response = requests.post(f"{API_URL}/predict", json={"text": message.text})
    recommendation = response.json()
    await message.reply(recommendation['mood'])

async def main():
    await dp.start_polling(bot)

if __name__ == "__main__":
    asyncio.run(main())
