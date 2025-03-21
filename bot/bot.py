from aiogram import Bot, Dispatcher, types
from aiogram.types import Message
from aiogram.filters import CommandStart, Command
import asyncio
import json
import aiohttp
from config import BOT_TOKEN, API_URL

bot = Bot(token=BOT_TOKEN)
dp = Dispatcher()

@dp.message(CommandStart())
async def cmd_start(message: Message):
    
    await message.answer("Hello!")

@dp.message(Command('predict'))
async def analyze_message(message: Message):
    user_text = message.text.replace("/predict", "").strip()
    async with aiohttp.ClientSession() as session:
        async with session.post(f"{API_URL}/predict", json={"text": user_text}) as response:
            if response.status == 200:
                try:
                    recommendation = await response.json()
                    if "mood" in recommendation and isinstance(recommendation["mood"], dict):                            
                        mood_text = "\n".join([f"{emotion}: {prob}%" for emotion, prob in recommendation["mood"].items()])
                        await message.reply(f"Предсказанное настроение:\n{mood_text}")
                    else:
                        await message.reply("Не удалось определить настроение текста.")
                except json.JSONDecodeError as e:
                    error_text = await response.text()
                    await message.reply(f"Ошибка анализа. API вернул некорректный ответ.")
                    print(f"JSON ошибка: {str(e)}, Ответ API: {error_text[:100]}")
    
    

@dp.message()
async def process_message(message: Message):
    await message.reply("Используйте команду /predict [текст] для анализа текста.")

async def main():
    await dp.start_polling(bot)

if __name__ == "__main__":
    print(f"Бот запущен и подключается к API по адресу: {API_URL}")
    asyncio.run(main())
