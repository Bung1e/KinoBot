from fastapi import FastAPI
from model import KinoRNN

app = FastAPI()
mood = "good"

@app.post("/predict")
async def predict():
    return{
        "mood": f"your mood is {mood}",
    }

if __name__ == "__main__":
    import uvicorn
    uvicorn.run("main:app", host="0.0.0.0", port=8000, reload=True)