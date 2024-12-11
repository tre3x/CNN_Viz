from flask import Flask, request, jsonify
from flask_cors import CORS
from lenet import LeNetModel
from vgg import VGG16Model
from utils import preprocess_for_lenet, preprocess_for_vgg16
import os

app = Flask(__name__)
CORS(app)  # Enable CORS for frontend-backend communication

# Initialize the model
model = LeNetModel()
#model = VGG16Model()

@app.route('/upload', methods=['POST'])
def upload_image():
    if 'image' not in request.files:
        return jsonify({"error": "No image part"}), 400

    file = request.files['image']
    if file.filename == '':
        return jsonify({"error": "No selected file"}), 400

    # Save the image temporarily
    temp_path = "./data/temp_image.jpg"
    file.save(temp_path)

    # Preprocess the image and run the model
    input_tensor = preprocess_for_lenet(temp_path)
    #input_tensor =  preprocess_for_vgg16(temp_path)
    final_output = model.forward(input_tensor)
    layer_output = model.intermediate_outputs

    # Convert activations to lists for JSON compatibility
    export_data = [output.tolist() for output in layer_output]

    # Clean up the temporary file
    os.remove(temp_path)

    return jsonify(export_data)

if __name__ == '__main__':
    app.run(debug=True, host='0.0.0.0', port=5000)
