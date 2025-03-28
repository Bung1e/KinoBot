# Emotion-Based Movie Recommender Bot

Telegram бот, который анализирует эмоциональное состояние пользователя по тексту и рекомендует фильмы, соответствующие настроению.

## Возможности

- Анализ эмоционального состояния по тексту
- Рекомендации фильмов на основе эмоций
- Интеграция с TMDB API для получения информации о фильмах
- REST API для анализа эмоций
- Docker контейнеризация

## Технологии

- Python 3.12
- PyTorch
- FastAPI
- aiogram 3.x
- Docker
- TMDB API

## Модель и обучение

### Архитектура модели

Модель представляет собой RNN (Recurrent Neural Network) с LSTM слоями, обученную на датасете GoEmotions. Основные характеристики:

- Входной слой: токенизированный текст
- Embedding слой: 128 размерности
- LSTM слой: 256 скрытых единиц
- Выходной слой: 6 эмоций (sad, joy, love, angry, fear, surprise)

### Процесс обучения

1. Подготовка данных:
   - Токенизация текста
   - Создание словаря
   - Векторизация предложений

2. Архитектура:
```python
class KinoRNN(nn.Module):
    def __init__(self, vocab_size, embedding_dim, hidden_size, num_layers, num_classes):
        super(KinoRNN, self).__init__()
        self.embedding = nn.Embedding(vocab_size, embedding_dim)
        self.lstm = nn.LSTM(embedding_dim, hidden_size, num_layers, batch_first=True)
        self.fc = nn.Linear(hidden_size, num_classes)
```

3. Гиперпараметры:
   - Размер батча: 32
   - Размер словаря: 10000
   - Размер эмбеддингов: 128
   - Размер скрытого слоя: 256
   - Количество слоев LSTM: 1
   - Learning rate: 0.001
   - Optimizer: Adam

4. Метрики:
   - Точность на валидационном наборе: ~85%
   - Loss: categorical_crossentropy

### Датасет

Использовался датасет GoEmotions, который содержит:
- 58k текстовых примеров
- 27 эмоциональных категорий
- Анонимизированные данные из Reddit

## Установка и запуск

### Предварительные требования

- Docker и Docker Compose
- TMDB API ключ
- Telegram Bot Token

### Настройка

1. Клонируйте репозиторий:
```bash
git clone https://github.com/yourusername/emotion-movie-bot.git
cd emotion-movie-bot
```

2. Создайте файл `.env` в корневой директории проекта:
```env
BOT_TOKEN=your_telegram_bot_token
API_URL=http://api:8000
TMDB_API_KEY=your_tmdb_api_key
```

### Запуск

1. Соберите и запустите Docker контейнеры:
```bash
docker-compose up --build -d
```

2. Проверьте статус контейнеров:
```bash
docker-compose ps
```

3. Проверьте логи:
```bash
docker-compose logs -f
```

## Использование

1. Найдите бота в Telegram по его username
2. Отправьте команду `/start` для начала работы
3. Используйте команду `/films` и напишите текст, описывающий ваше настроение
4. Получите рекомендации фильмов, соответствующих вашему настроению

## API Endpoints

- `POST /films` - анализ текста и получение рекомендаций фильмов
- `GET /docs` - Swagger документация API

## Структура проекта

```
.
├── api/
│   └── api.py          # FastAPI приложение
├── bot/
│   └── bot.py          # Telegram бот
├── model/
│   ├── model.py        # Модель для анализа эмоций
│   └── data.py         # Обработка данных
├── docker-compose.yml  # Docker Compose конфигурация
├── Dockerfile         # Dockerfile для сборки образа
├── requirements.txt   # Зависимости проекта
└── .env              # Переменные окружения
```

## Разработка

Для локальной разработки:

1. Создайте виртуальное окружение:
```bash
python -m venv venv
source venv/bin/activate  # для Linux/Mac
venv\Scripts\activate     # для Windows
```

2. Установите зависимости:
```bash
pip install -r requirements.txt
```

3. Запустите API локально:
```bash
uvicorn api.api:app --reload
```

4. Запустите бота:
```bash
python bot/bot.py
```

## Лицензия

MIT

## Автор

[Ваше имя]

## Поддержка

Если у вас есть вопросы или проблемы, создайте issue в репозитории проекта.