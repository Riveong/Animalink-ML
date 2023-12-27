import time
import uvicorn
from fastapi import FastAPI, File, UploadFile, HTTPException
from fastapi.responses import JSONResponse
import tensorflow as tf
from tensorflow import keras
from PIL import Image
import numpy as np
from tensorflow.keras.preprocessing.image import img_to_array

app = FastAPI()

model = tf.keras.models.load_model("model/animalink_model.h5")

animal_classes = {0 : 'Anjing Ajag',  
       1 : 'Merak Biru',
       2 : 'Merak Hijau', 
       3 : 'Sanca Hijau',
       4 : 'Anjing Shiba',
       5 : 'Tenggiling',
       6 : 'Kucing Angora',
       7 : 'Ikan Koi', 
       8 : 'Jalak Bali',
       9 : 'Sanca Bola'}

extinct_animals = {"Tenggiling", "Jalak Bali", "Anjing Ajag", "Merak Hijau", "Sanca Hijau"}

def preprocess_image(image):
    img = Image.open(image.file).convert("RGB")
    img = img.resize((224, 224)) 
    img_array = np.asarray(img)
    img_array = img_to_array(img)
    img_array = img_array / 255.0
    img_array = np.expand_dims(img_array, axis=0) 
    return img_array

@app.post("/predict")
async def predict(file: UploadFile = File(...)):
    start_time = time.time()
    try:
        image_array = preprocess_image(file)
        predictions = model.predict(image_array)
        class_index = np.argmax(predictions[0])
        confidence = predictions[0][class_index] * 100
        predicted_animal = animal_classes.get(class_index, "Unknown Animal")

        if confidence < 70:
            status = "Fail"
            status_extinct = "unknown"
            message = f"Prediction confidence is under 70 percent for desire animal. Please verify the result." 
            predict = f"{round(confidence, 2)}"

        elif predicted_animal in extinct_animals:
            status = "Success"
            status_extinct = "extinct"
            message = f"Warning! {predicted_animal} is an extinct animal. Selling them is prohibited."
            predict = f"{round(confidence, 2)}"
        else:
            status = "Success"
            status_extinct = "not extinct"
            message = f"The predicted animal is {predicted_animal}. This animal can legally be sold."
            predict = f"{round(confidence, 2)}"

        end_time = time.time()
        benchmark = end_time - start_time
        return JSONResponse(content={"status": status, "predicted_animal": predicted_animal, "animal_status": status_extinct, "message": message, "model_confidence": predict, "benchmark": benchmark})
    except Exception as e:
        end_time = time.time()
        benchmark = end_time - start_time
        status = "Fail"
        status_extinct = "unknown"
        message = f"An error occurred: {str(e)}"
        predict = "N/A"

        return JSONResponse(content={"status": status, "predicted_animal": "Unknown Animal", "animal_status": status_extinct, "message": message, "model_confidence": predict, "benchmark": benchmark}, status_code=500)

if __name__ == "__main__":
    uvicorn.run(app, host='0.0.0.0',port=8080)
