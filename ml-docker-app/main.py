from fastapi import FastAPI, File
from pydantic import BaseModel
import joblib
from xgboost import Booster, XGBClassifier
import numpy as np
import pickle
import warnings

import base64
from PIL import Image
import io

warnings.simplefilter(action='ignore', category=DeprecationWarning)

app = FastAPI()

class PredictionResponse(BaseModel):
    prediction: float

class ImageRequest(BaseModel):
    image: str

def load_model():
    global xgb_tuned_carregado

    # If the model was saved using Booster.save_model (recommended for XGBoost)
    model = Booster()
    model.load_model("xgb_tuned.json")

    # Convert the Booster to an XGBClassifier
    xgb_tuned_carregado = XGBClassifier()
    xgb_tuned_carregado._Booster = model
    xgb_tuned_carregado.n_classes_ = 10  # Adjust based on the dataset used

    print("Model loaded successfully using Booster")

@app.on_event("startup")
async def startup_event():
    load_model()

@app.post("/predict", response_model=PredictionResponse)
async def predict(request: ImageRequest):
    img_bytes = base64.b64decode(request.image)
    img = Image.open(io.BytesIO(img_bytes))
    img = img.resize((8,8))
    img_array = np.array(img)

    img_array = np.dot(img_array[...,:3], [0.2989, 0.5870, 0.1140])

    img_array = img_array.reshape(1, -1)

    prediction = xgb_tuned_carregado.predict(img_array)

    return {"prediction": prediction}
