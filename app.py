import os
import torch
from torchvision import models, transforms
from PIL import Image
from flask import Flask, request, render_template, jsonify

app = Flask(__name__)

# Define class names
class_names = [
    'AnnualCrop', 'Forest', 'HerbaceousVegetation', 'Highway', 'Industrial', 
    'Pasture', 'PermanentCrop', 'Residential', 'River', 'SeaLake'
]

# Load the model
MODEL_PATH = "D:\\Deployment\\ResNet-50_model.pth"

def load_model(model_path):
    if not os.path.isfile(model_path):
        raise FileNotFoundError(f"Model file '{model_path}' not found.")

    model = models.resnet50(weights=None)
    model.fc = torch.nn.Linear(model.fc.in_features, len(class_names))
    
    try:
        model.load_state_dict(torch.load(model_path, map_location=torch.device('cpu')))
        model.eval()
        print(f'Model successfully loaded from {model_path}')
        return model
    except Exception as e:
        print(f"Error loading model: {e}")
        return None

model = load_model(MODEL_PATH)

# Define the image transformation
transform = transforms.Compose([
    transforms.Resize((64, 64)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5])
])

# Prediction function
def predict_image(image_path):
    # Open and preprocess the image
    image = Image.open(image_path).convert('RGB')
    image = transform(image).unsqueeze(0)  # Add batch dimension
    
    # Make prediction
    with torch.no_grad():
        outputs = model(image)
        _, preds = torch.max(outputs, 1)
        probs = torch.nn.functional.softmax(outputs, dim=1)
    
    # Get the predicted class and confidence
    predicted_class = class_names[preds.item()]
    confidence = probs[0][preds.item()].item()  # Ensure confidence is a float
    
    return predicted_class, confidence

# Flask routes
@app.route('/')
def home():
    return render_template('index.html')

@app.route('/predict', methods=['POST'])
def predict():
    if 'file' not in request.files:
        return jsonify({'error': 'No file uploaded'}), 400
    
    file = request.files['file']
    if file.filename == '':
        return jsonify({'error': 'No file selected'}), 400
    
    # Save the uploaded file temporarily
    image_path = "temp_image.jpg"
    file.save(image_path)
    
    # Make prediction
    predicted_class, confidence = predict_image(image_path)
    
    # Remove the temporary file
    os.remove(image_path)
    
    return jsonify({
        'predicted_class': predicted_class,
        'confidence': float(confidence)  # Ensure confidence is a float
    })

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5000, debug=True)
