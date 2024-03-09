from flask import Flask, request, jsonify
from flask_cors import CORS
import cv2
import numpy as np
from tensorflow import keras

app = Flask(__name__)
CORS(app)  # Enable CORS for all routes, adjust the configuration as needed

# Load the pre-trained Keras model


# Class names corresponding to the model output
class_names = ["Mild Dementia", "Moderate Dementia", "Non Dementia", "Very Mild Dementia"]

# Variable to store the uploaded image
uploaded_image = None

@app.route('/upload', methods=['POST', 'GET'])
def upload_file():
    global uploaded_image
    if 'image' not in request.files:
        return jsonify({'error': 'No file part'}), 400

    file = request.files['image']
    if file.filename == '':
        return jsonify({'error': 'No selected file'}), 400

    try:
        # Read and store the image array
        uploaded_image = cv2.imdecode(np.frombuffer(file.read(), np.uint8), cv2.IMREAD_COLOR)
        # Resize the image
        uploaded_image = cv2.resize(uploaded_image, (176, 176))
        return jsonify({'message': 'File uploaded successfully'}), 200

    except Exception as e:
        print(f"Error reading the image: {e}")
        return jsonify({'error': 'Error reading the image'}), 500

@app.route('/predict', methods=['GET', 'POST'])
def predict():
    global uploaded_image
    model = keras.models.load_model('./model/AlzDisConvModel_hyperTuned.h5')
    try:
        if uploaded_image is None:
            return jsonify({'error': 'No uploaded image'}), 400

        # Make predictions using the loaded Keras model
        prediction_probs = model.predict(np.expand_dims(uploaded_image, axis=0))
        predicted_class_index = np.argmax(prediction_probs)
        predicted_class_name = class_names[predicted_class_index]

        return jsonify({'prediction': predicted_class_name, 'confidence': float(prediction_probs[0][predicted_class_index])}), 200

    except Exception as e:
        print(f"Error predicting: {e}")
        return jsonify({'error': 'Error predicting'}), 500
    finally:
        # Reset the uploaded_image variable after making predictions
        uploaded_image = None

if __name__ == '__main__':
    app.run(debug=True)
