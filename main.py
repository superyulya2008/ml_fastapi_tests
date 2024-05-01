from fastapi import FastAPI
from pydantic import BaseModel
from transformers import pipeline


class Item(BaseModel):
    text: str  # уменьшила количество пробелов


app = FastAPI()
classifier = pipeline("sentiment-analysis")  # Уменьшила количество пробелов


@app.get("/")
def root():
    return {"message": "Hello World"}


@app.post("/predict/")
def predict(item: Item):
    return classifier(item.text)[0]
