from flask import Flask, request, jsonify, render_template
import base64
from io import BytesIO
from PIL import Image
import torch
import config
from tokenizer_utils import get_tokenizer
from dataset_utils import get_image_transforms
from model import OCRModel
from inference import load_model_for_inference, predict

app = Flask(__name__)

# Initialize components
device = torch.device(config.DEVICE)
tokenizer = get_tokenizer()
image_transform = get_image_transforms()
model = load_model_for_inference(
    model_path=f"{config.MODEL_SAVE_DIR}/{config.BEST_MODEL_NAME}",
    device=device,
    tokenizer=tokenizer
)

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/api/latex_ocr', methods=['POST'])
def latex_ocr():
    data = request.get_json()
    if not data or 'image' not in data:
        return jsonify({'error': 'No image data provided'}), 400

    image_data = data['image']
    try:
        # Remove header if present
        if ',' in image_data:
            image_data = image_data.split(',')[1]
        image_bytes = base64.b64decode(image_data)
        image = Image.open(BytesIO(image_bytes)).convert('RGB')
    except Exception as e:
        return jsonify({'error': f'Invalid image data: {str(e)}'}), 400

    try:
        latex_code = predict(model, image, tokenizer, device, image_transform)
        return jsonify({'latex': latex_code})
    except Exception as e:
        return jsonify({'error': f'Prediction failed: {str(e)}'}), 500

if __name__ == '__main__':
    app.run(debug=True)
