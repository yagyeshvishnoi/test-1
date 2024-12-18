from flask import Flask, request, jsonify
from PIL import Image
import torch
from func import *
import psutil  # Import psutil for memory tracking
import os

app = Flask(__name__)

# Function to get the current memory usage
def get_memory_usage():
    """Returns the memory usage of the current process in MB."""
    process = psutil.Process(os.getpid())
    memory_usage = process.memory_info().rss / (1024 * 1024)  # Convert to MB
    return memory_usage

# Define route for image upload and prediction
@app.route('/predict', methods=['POST'])
def predict():
    try:
        # Check if the image is provided in the form-data
        if 'img' not in request.files:
            return jsonify({"error": "No file part"}), 400

        # Access the file
        file = request.files['img']
        if file.filename == '':
            return jsonify({"error": "No selected file"}), 400

        # Log memory usage before processing
        initial_memory = get_memory_usage()

        # Open the image using PIL
        img = Image.open(file.stream)

        # Preprocess the image
        img_tensor = transforms.ToTensor()(img)  # Transform to tensor
        img_tensor = img_tensor.unsqueeze(0)    # Add batch dimension

        # Perform prediction
        predicted_label = predict_image(img_tensor.squeeze(0), model)

        # Log memory usage after processing
        final_memory = get_memory_usage()

        # Return the result in JSON format, including memory usage
        return jsonify({
            "prediction": predicted_label,
            "memory_usage": {
                "initial": f"{initial_memory:.2f} MB",
                "final": f"{final_memory:.2f} MB",
                "difference": f"{final_memory - initial_memory:.2f} MB"
            }
        })

    except Exception as e:
        return jsonify({"error": str(e)}), 500


if __name__ == '__main__':
    app.run(debug=True)
