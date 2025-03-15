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


if __name__ == "__main__":
    asyncio.run(main())