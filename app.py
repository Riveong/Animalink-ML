from fastapi import FastAPI, File, UploadFile, HTTPException
from fastapi.responses import JSONResponse
import tensorflow as tf
from tensorflow import keras
from PIL import Image
import numpy as np

app = FastAPI()
model_path = 'animalink.h5'
model = keras.models.load_model(model_path)

dic = {0 : 'Anjing ajag',  
       1 : 'Merak biru',
       2 : 'Merak hijau', 
       3 : 'Sanca hijau',
       4 : 'Anjing Shiba',
       5 : 'Tenggiling',
       6 : 'Turkish angora',
       7 : 'Ikan koi',
       8 : 'Jalak bali',
       9 : 'Sanca bola'}

extinct_animals = {"Tenggiling", "Jalak bali", "Anjing ajag", "Merak hijau", "Sanca hijau"}

def preprocess_image(image):
    img = Image.open(image.file).convert("RGB")
    img = img.resize((150, 150)) 
    img_array = np.asarray(img)
    img_array = img_to_array(img)
    img_array = img_array / 255.0
    return img_array

@app.post("/predict")
async def predict(file: UploadFile = File(...)):
    try:
        image_array = preprocess_image(file)
        predictions = model.predict(image_array)
        class_index = np.argmax(predictions[0])
        predicted_animal = animal_classes.get(class_index, "Unknown Animal")

        if predicted_animal in extinct_animals:
            message = f"Warning! {predicted_animal} is an extinct animal. Selling them is prohibited."
        else:
            message = f"The predicted animal is {predicted_animal}. You can sell them."

        return JSONResponse(content={"predicted_animal": predicted_animal, "message": message})
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))