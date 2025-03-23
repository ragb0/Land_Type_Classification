#!/usr/bin/env python
# coding: utf-8

# In[1]:


import torch
import streamlit as st
import torchvision.transforms as transforms
from torchvision import models
from flask import Flask, request, jsonify
from PIL import Image
import io

# In[3]:
class_labels = {
    0: "Forest",
    1: "Desert",
    2: "Water",
    3: "Urban Area",
    4: "Farmland",
    5: "Mountain",
    6: "Grassland",
    7: "Glacier",
    8: "Wetlands",
    9: "Highway"
}


# Initialize Flask app
app = Flask(__name__)


# In[4]:


# Load the pretrained ResNet-50 model
model = models.resnet50(pretrained=False)  # Use False since we load a custom model
num_ftrs = model.fc.in_features
model.fc = torch.nn.Linear(num_ftrs, 14)  # Modify based on your number of classes
model.load_state_dict(torch.load("ResNet-50_model.pth", map_location=torch.device("cpu")))
model.eval()  # Set model to evaluation mode


# In[5]:


# Define image transformations (match the ones used during training)
transform = transforms.Compose([
    transforms.Resize((224, 224)),  # ResNet-50 requires 224x224 images
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
])


# In[6]:
@app.route('/')
def home():
    return "Welcome to the Land Type Classification API! Use /predict to send an image."


@app.route('/predict', methods=['POST'])
def predict():
    try:
        if 'file' not in request.files:
            return jsonify({'error': "No file uploaded"}), 400

        file = request.files['file']
        image = Image.open(io.BytesIO(file.read())).convert("RGB")
        image = transform(image).unsqueeze(0)

        with torch.no_grad():
            output = model(image)
            probabilities = torch.nn.functional.softmax(output, dim=1)[0]
            predicted_class = torch.argmax(probabilities).item()
            confidence = round(probabilities[predicted_class].item() * 100, 2)

        return jsonify({
            'prediction': predicted_class,
            'label': class_labels.get(predicted_class, "Unknown"),  # ✅ Returns class name
            'confidence': confidence
        })

    except Exception as e:
        return jsonify({'error': str(e)})
# In[7]:


# Run the Flask app
if __name__ == '__main__':
    app.run(debug=True)


# In[ ]:




