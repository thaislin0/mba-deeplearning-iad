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

# def preprocess_image(image):
#     """
#     Preprocess the input image to match the format expected by the model.
#     - Convert to grayscale
#     - Resize to 8x8 pixels
#     - Invert colors to match the original digits dataset
#     - Flatten into a 64-element array
#     """
#     image = image.convert('L')  # Convert to grayscale
#     image = image.resize((8, 8))  # Resize to 8x8 pixels
#     image = np.array(image)  # Convert to NumPy array
#     image = 16 - (image // 16)  # Invert colors
#     image = image.flatten()  # Flatten the image to a 64-element array
#     return image

# @app.route('/')
# def index():
#     return render_template('index.html')

# @app.route('/predict', methods=['POST'])
# def predict():
#     if 'file' not in request.files:
#         return jsonify({'error': 'No file provided'}), 400

#     file = request.files['file']
#     image = Image.open(io.BytesIO(file.read()))
#     processed_image = preprocess_image(image)
    
#     # # Create a DataFrame with the correct column names
#     column_names = [str(i) for i in range(64)]  # Column names as strings "0", "1", ..., "63"
#     processed_image_df = pd.DataFrame([processed_image], columns=column_names)
    
#     # Make prediction
#     prediction = model.predict(processed_image_df)
    
#     return jsonify({'prediction': int(prediction[0])})

# if __name__ == '__main__':
#     app.run(host='0.0.0.0', port=8000)
