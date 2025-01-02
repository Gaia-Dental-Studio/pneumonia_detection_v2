import streamlit as st
import requests
from PIL import Image
import io

# Flask backend URL
BACKEND_URL = "http://127.0.0.1:5000/predict"

st.title("Pneumonia Detection")
st.write("Upload a chest X-ray image for classification.")

# File uploader
uploaded_file = st.file_uploader("Choose an image...", type=["jpg", "jpeg", "png"])

if uploaded_file is not None:
    # Display the uploaded image
    image = Image.open(uploaded_file)
    st.image(image, caption="Uploaded Image", use_column_width=True)

    # Convert the image to bytes for sending to the backend
    image_bytes = io.BytesIO()
    image.save(image_bytes, format=image.format)
    image_bytes.seek(0)

    # Send the file to the Flask backend
    files = {"file": ("image.jpg", image_bytes, f"image/{image.format.lower()}")}
    response = requests.post(BACKEND_URL, files=files)

    # Display the prediction results
    if response.status_code == 200:
        result = response.json()
        st.success(f"Predicted Class: {result['Predicted Class']}")
        st.write("### Probability Scores")
        for class_name, probability in result["Probability Scores"].items():
            st.write(f"{class_name}: {probability:.2f}")
    else:
        st.error(f"Error: {response.json().get('error', 'Unknown error')}")
