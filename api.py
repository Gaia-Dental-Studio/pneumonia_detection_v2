from flask import Flask, request, jsonify
import numpy as np
import tensorflow as tf
from tensorflow.keras.preprocessing import image
import os

app = Flask(__name__)

# Load the trained model
model = tf.keras.models.load_model('./model/CNN_model-v2.h5')
model.compile(
    loss='categorical_crossentropy',
    optimizer=tf.optimizers.SGD(learning_rate=0.001),
    metrics=['accuracy']
)

# Define class labels
class_labels = ['NORMAL', 'PNEUMONIA']

def preprocess_and_predict(img_path):
    """
    Predict the class of an input image using the trained model.

    Args:
        img_path (str): Path to the input image.

    Returns:
        dict: Predicted class with probability scores for each class.
    """
    # Load and preprocess the image
    img = image.load_img(img_path, target_size=(224, 224))  # Resize image to model's input size
    img_array = image.img_to_array(img)  # Convert to array
    img_array = np.expand_dims(img_array, axis=0)  # Add batch dimension
    img_array = tf.keras.applications.mobilenet_v2.preprocess_input(img_array)  # Preprocess as in training

    # Perform prediction
    predictions = model.predict(img_array)
    predicted_class = class_labels[np.argmax(predictions)]  # Get class with highest probability
    predicted_probs = {class_labels[i]: float(predictions[0][i]) for i in range(len(class_labels))}

    return {
        "Predicted Class": predicted_class,
        "Probability Scores": predicted_probs
    }

@app.route('/predict', methods=['POST'])
def predict():
    # Check if an image file is uploaded
    if 'file' not in request.files:
        return jsonify({"error": "No file provided"}), 400

    file = request.files['file']
    temp_path = "./temp_image.jpg"
    file.save(temp_path)

    try:
        # Perform prediction
        result = preprocess_and_predict(temp_path)
        os.remove(temp_path)  # Remove the temporary image file
        return jsonify(result)
    except Exception as e:
        os.remove(temp_path)
        return jsonify({"error": str(e)}), 500

if __name__ == "__main__":
    app.run(debug=True)
