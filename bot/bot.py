from aiogram import Bot, Dispatcher, types
from aiogram.types import Message
from aiogram.filters import CommandStart, Command
import asyncio
import json
import aiohttp
import os

API_URL = os.getenv('API_URL')
bot = Bot(token=os.getenv('BOT_TOKEN'))
dp = Dispatcher()

@dp.message(CommandStart())
async def cmd_start(message: Message):
    
    await message.answer("Hello!")

@dp.message(Command('films'))
async def analyze_message(message: Message):
    user_text = message.text.replace("/films", "").strip()
    async with aiohttp.ClientSession() as session:
        async with session.post(f"{API_URL}/films", json={"text": user_text}) as response:
            if response.status == 200:
                try:
                    recommendation = await response.json()
                    if "movies" in recommendation and recommendation["movies"]:
                        movies_text = "Recommended films:\n\n"
                        for movie in recommendation["movies"]:
                            movies_text += f"🎬 {movie.get('title', 'unknown name')}\n" \
                                           f"Discription: {movie.get('overview', 'no discription')[:100]}...\n" \
                                           f"Rating: {movie.get('vote_average', 'no average votes')}/10\n\n"
                        
                        await message.reply(movies_text)
                except json.JSONDecodeError as e:
                    error_text = await response.text()
                    await message.reply(f"Anlysing error")
                    print(f"JSON error: {str(e)}, API response: {error_text[:100]}")
    

async def main():
    await dp.start_polling(bot)

if __name__ == "__main__":
    asyncio.run(main())
