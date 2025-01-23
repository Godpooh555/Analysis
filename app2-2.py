import gradio as gr
import tensorflow as tf
from PIL import Image
import numpy as np
import os
import requests

model_05_path = "model_05.keras"

def download_model(model_url, model_path):
    if not os.path.exists(model_path):
        print(f"Downloading model from {model_url}...")
        response = requests.get(model_url)
        with open(model_path, 'wb') as f:
            f.write(response.content)
        print(f"Model downloaded and saved to {model_path}")
    else:
        print(f"Model found at {model_path}, no download needed.")

model_url = "https://drive.google.com/uc?export=download&id=1cc_KscsoMtedAqDAbDX0h4FkEXUZPFrv"

download_model(model_url, model_05_path)

print("Loading model...")
try:
    model_05 = tf.keras.models.load_model(model_05_path)
    print("Model loaded successfully.")
except Exception as e:
    print(f"Error loading model: {e}")

def predict_image(image):
    image = image.resize((224, 224)) 
    image_array = np.array(image) / 255.0 
    image_array = np.expand_dims(image_array, axis=0) 

    try:
        prediction_05 = model_05.predict(image_array)

        if prediction_05[0][0] > 0.5:
            return "Stroke Detected"
        else:
            return "Stroke Not Detected"
    except Exception as e:
        print(f"Error during prediction: {e}")
        return "Error in prediction"

interface = gr.Interface(
    fn=predict_image,
    inputs=gr.Image(type="pil"),
    outputs="text",
    title="Ischemic Stroke Detection",
    description="Upload an MRI image to detect if ischemic stroke is present."
)

print("Launching interface...")
interface.launch()
