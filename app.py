import streamlit as st
import requests
from PIL import Image
import io
import base64

# Function to set the background using an image
def set_background(image_path=None):
    if image_path:
        with open(image_path, "rb") as img_file:
            base64_image = base64.b64encode(img_file.read()).decode()
        bg_style = f"""
        <style>
        .stApp {{
            background-image: url("data:image/jpeg;base64,{base64_image}");
            background-size: auto 50%;
            background-position: center bottom 20px ;
            background-repeat: no-repeat;
        }}
        </style>
        """
    else:
        bg_style = """
        <style>
        .stApp {
            background: none;
        }
        </style>
        """
    st.markdown(bg_style, unsafe_allow_html=True)

# Define class labels
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

st.markdown("<h1 style='color:white;'>🌍 Land Types</h1>", unsafe_allow_html=True)

# File uploader
uploaded_file = st.file_uploader("📂 Upload an image", type=["jpg", "png", "jpeg"])

if uploaded_file is not None:
    # Remove background when an image is uploaded
    set_background(None)

    # Display uploaded image
    image = Image.open(uploaded_file)
    st.image(image, caption="🖼 Uploaded Image", use_container_width=True)

    # Convert image to bytes
    img_bytes = io.BytesIO()
    image.save(img_bytes, format="PNG")
    img_bytes = img_bytes.getvalue()

    # Send request to API
    response = requests.post("http://127.0.0.1:5000/predict", files={"file": img_bytes})

    if response.status_code == 200:
        data = response.json()
        prediction = data.get("prediction", "Unknown")
        confidence = data.get("confidence", None)

        label_name = class_labels.get(prediction, "Unknown")

        # Display result
        st.markdown(f"**Prediction:** <span style='color:green;'>{label_name}</span>", unsafe_allow_html=True)
        if confidence is not None:
            st.markdown(f"**Confidence:** <span style='color:blue;'>{confidence:.2f}%</span>", unsafe_allow_html=True)
        else:
            st.error("❌ Confidence score not returned by API")
    else:
        st.error("❌ Failed to get prediction. Please try again.")
else:
    # Set background image if no image is uploaded
    set_background("background_image.jpeg")  # Replace with your background image path
