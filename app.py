from flask import Flask, jsonify, request
from keras.models import load_model
import numpy as np
from PIL import Image

app = Flask(__name__)

# Load the pre-trained Keras model
model = load_model('HistoPath_Can.h5')

# Define the route for the prediction API
@app.route('/predict', methods=['POST'])
def predict():
    # Get the uploaded image file
    file = request.files['file']

    # Load the image and preprocess it for the model
    img = Image.open(file)
    img = img.resize((224, 224)) # Resize to match model input shape
    img = np.array(img)
    img = img / 255.0 # Normalize pixel values
    img = np.expand_dims(img, axis=0) # Add batch dimension

    # Make the prediction using the model
    prediction = model.predict(img)[0]
    class_idx = np.argmax(prediction)
    class_prob = prediction[class_idx]

    # Return the prediction as a JSON object
    return jsonify({
        'class_idx': int(class_idx),
        'class_prob': float(class_prob)
    })

if __name__ == '__main__':
    app.run(debug=True)
