# !pip install torch torchvision transformers flask flask-ngrok pyngrok

from flask import Flask, request, jsonify
from flask_ngrok import run_with_ngrok
from PIL import Image
import torch
from torchvision import transforms
from transformers import ViTForImageClassification, ViTFeatureExtractor
from pyngrok import ngrok
import threading
import time

import json

with open('config.json', 'r') as file:
    config = json.load(file)


device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# Set up ngrok authentication
NGROK_AUTH_TOKEN = config['NGROK_AUTH_TOKEN']
ngrok.set_auth_token(NGROK_AUTH_TOKEN)

# Load the model and feature extractor
model_name = "m-faraz-ali/Vit_Classification_Pneumonia"
model = ViTForImageClassification.from_pretrained(model_name)
feature_extractor = ViTFeatureExtractor.from_pretrained(model_name)

# Function to predict on a single image
def predict_tb(image):
    inputs = feature_extractor(images=image, return_tensors="pt")
    inputs = {k: v.to(device) for k, v in inputs.items()}

    # Make prediction
    with torch.no_grad():
        outputs = model(**inputs)
    logits = outputs.logits
    preds = torch.nn.functional.softmax(logits, dim=1)
    predicted_class = preds.argmax(dim=1).item()

    return predicted_class, preds[0][predicted_class].item()

app = Flask(__name__)
run_with_ngrok(app)


@app.route('/')
def hello():
    return 'Hello, World!'

# Prediction endpoint
@app.route('/predict', methods=['POST'])
def predict():
    try:
      if 'file' not in request.files:
          return jsonify({'error': 'No file part'})

      file = request.files['file']
      if file.filename == '':
          return jsonify({'error': 'No selected file'})
      image = Image.open(file).convert("RGB")
      predicted_class, confidence = predict_tb(image)
      label_map = {0: 'No Pneumonia', 1: 'Pneumonia'}  # Adjust according to your label mapping
      prediction = label_map[predicted_class]

      return jsonify({'label': prediction, 'confidence': confidence})
    except Exception as e:
        return jsonify({'error': str(e)}), 500
# Run the app
def run_flask():
    app.run()

def start_ngrok():
    time.sleep(2)  # Give the Flask server some time to start
    public_url = ngrok.connect(5000)
    print('Public URL:', public_url)
    return public_url

def main():
    # Start Flask server in a new thread
    flask_thread = threading.Thread(target=run_flask)
    flask_thread.start()

    # Start ngrok tunnel
    public_url = start_ngrok()

    try:
        flask_thread.join()
    except KeyboardInterrupt:
        print("Shutting down...")
    finally:
        ngrok.disconnect(public_url)
        ngrok.kill()

if __name__ == "__main__":
    main()
