from flask import Flask, jsonify, request
from keras.models import load_model
import numpy as np
from PIL import Image
from cachelib import SimpleCache


app = Flask(__name__)

# Load the pre-trained Keras model
model = load_model('HistoPath_Can.h5')

# Create a cache
cache = SimpleCache()

# Define the route for the prediction API
@app.route('/predict', methods=['POST'])
def predict():
    # Get the uploaded image file
    file = request.files['file']

    # Check if the image is already in the cache
    key = 'image-' + file.filename
    prediction = cache.get(key)

    # If the image is not in the cache, make the prediction and store it in the cache
    if prediction is None:
        img = Image.open(file)
        img = img.resize((224, 224)) # Resize to match model input shape
        img = np.array(img)
        img = img / 255.0 # Normalize pixel values
        img = np.expand_dims(img, axis=0) # Add batch dimension

        # Make the prediction using the model
        prediction = model.predict(img)[0]
        class_idx = np.argmax(prediction)
        class_prob = prediction[class_idx]

        # Store the prediction in the cache
        cache.set(key, prediction)

    # Return the prediction as a JSON object
    return jsonify({
        'class_idx': int(class_idx),
        'class_prob': float(class_prob)
    })

if __name__ == '__main__':
    app.run(debug=True)
