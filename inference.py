# inference.py
"""
Script for loading a trained model and performing inference on new images.
"""
import torch
from PIL import Image
import os
import argparse

import config
from tokenizer_utils import get_tokenizer, LaTeXTokenizer
from dataset_utils import get_image_transforms # For preprocessing new images
from model import OCRModel

def load_model_for_inference(model_path, device, tokenizer):
    if not os.path.exists(model_path):
        print(f"Model checkpoint not found at {model_path}")
        return None

    print(f"Loading model from {model_path} for inference...")
    # Try to load config from checkpoint if available, otherwise use current config
    try:
        checkpoint = torch.load(model_path, map_location=device)
        chkpt_config = checkpoint.get('config', {}) # Get saved config if exists
        
        # Prioritize tokenizer's dynamic values for vocab_size and pad_id
        vocab_size = config.DYNAMIC_VOCAB_SIZE if config.DYNAMIC_VOCAB_SIZE else chkpt_config.get('DYNAMIC_VOCAB_SIZE', config.VOCAB_SIZE)
        pad_token_id = config.DYNAMIC_PAD_TOKEN_ID if config.DYNAMIC_PAD_TOKEN_ID is not None else chkpt_config.get('DYNAMIC_PAD_TOKEN_ID', config.PAD_TOKEN_ID)
        
        # Model parameters should ideally be stored or inferred if not in global config
        model = OCRModel(
            vision_encoder_name=chkpt_config.get('VISION_ENCODER_NAME', config.VISION_ENCODER_NAME),
            decoder_dim=chkpt_config.get('DECODER_DIM', config.DECODER_DIM),
            decoder_depth=chkpt_config.get('DECODER_DEPTH', config.DECODER_DEPTH),
            decoder_heads=chkpt_config.get('DECODER_HEADS', config.DECODER_HEADS),
            vocab_size=vocab_size,
            max_seq_len=chkpt_config.get('MAX_SEQ_LEN', config.MAX_SEQ_LEN),
            pad_token_id=pad_token_id
        ).to(device)
        model.load_state_dict(checkpoint['model_state_dict'])
        model.eval()
        print("Model loaded successfully.")
        return model
    except Exception as e:
        print(f"Error loading model: {e}")
        print("Ensure the model architecture in `config.py` matches the saved model, or that checkpoint['config'] is valid.")
        return None


def predict(model: OCRModel, image_path_or_pil, tokenizer: LaTeXTokenizer, device, image_transform):
    if not model or not tokenizer or not tokenizer.tokenizer:
        print("Model or tokenizer not available.")
        return None

    try:
        if isinstance(image_path_or_pil, str):
            image = Image.open(image_path_or_pil).convert('RGB')
        elif isinstance(image_path_or_pil, Image.Image):
            image = image_path_or_pil.convert('RGB')
        else:
            raise ValueError("Input must be an image path or a PIL Image object.")
    except Exception as e:
        print(f"Error loading image {image_path_or_pil}: {e}")
        return None

    pixel_values = image_transform(image).unsqueeze(0).to(device) # Add batch dimension

    with torch.no_grad():
        # Get SOS token ID from the loaded tokenizer via config
        sos_token_id = config.DYNAMIC_SOS_TOKEN_ID if config.DYNAMIC_SOS_TOKEN_ID is not None else config.SOS_TOKEN_ID

        start_tokens = torch.ones((1, 1), dtype=torch.long, device=device) * sos_token_id
        
        # Encode the image to obtain context
        # encoded_features = model.encoder(pixel_values)
        
        generated_ids = model(
            pixel_values,
            generate=True,
            start_tokens=start_tokens,
            seq_len=config.GENERATION_MAX_LEN, # Max length for generation
            temperature=config.GENERATION_TEMPERATURE,
            filter_thres=config.GENERATION_FILTER_THRES, # Nucleus sampling
            eos_token=config.DYNAMIC_EOS_TOKEN_ID if config.DYNAMIC_EOS_TOKEN_ID is not None else config.EOS_TOKEN_ID # Pass EOS token for stopping
        )
        # generated_ids = model.autoregressive_wrapper.generate(
        #     prompts=start_tokens,
        #     context=encoded_features,
        #     seq_len=config.GENERATION_MAX_LEN,
        #     temperature=config.GENERATION_TEMPERATURE,
        #     filter_thres=config.GENERATION_FILTER_THRES,
        #     eos_token=config.DYNAMIC_EOS_TOKEN_ID if config.DYNAMIC_EOS_TOKEN_ID is not None else config.EOS_TOKEN_ID
        # )
    # generated_ids shape: (batch_size, seq_len)
    
    # Decode the generated IDs
    # The generate method of AutoregressiveWrapper usually outputs tokens including SOS and potentially EOS.
    # skip_special_tokens=True in batch_decode will handle removing SOS, EOS, PAD.
    decoded_text = tokenizer.decode(
      generated_ids.cpu().tolist(),
      skip_special_tokens=True
    ).replace("Ġ", " ").replace("\\ ", "\\")
    
    return decoded_text[0] # Return the first (and only) generated string

def main():
    parser = argparse.ArgumentParser(description="Perform LaTeX OCR on an image.")
    parser.add_argument("image_path", type=str, help="Path to the input image file.")
    parser.add_argument("--model_path", type=str,
                        default=os.path.join(config.MODEL_SAVE_DIR, config.BEST_MODEL_NAME),
                        help="Path to the trained model checkpoint.")
    args = parser.parse_args()

    print("--- LaTeX OCR Inference ---")
    device = torch.device(config.DEVICE)
    print(f"Using device: {device}")

    # 1. Initialize Tokenizer
    print("Initializing tokenizer...")
    latex_tokenizer = get_tokenizer()
    if not latex_tokenizer or not latex_tokenizer.tokenizer:
        print("Failed to initialize tokenizer. Exiting.")
        return
    if config.DYNAMIC_VOCAB_SIZE is None: # Ensure tokenizer has updated config
        print("CRITICAL: Tokenizer did not set DYNAMIC_VOCAB_SIZE. Check tokenizer loading.")
        return

    # 2. Load Model
    model = load_model_for_inference(args.model_path, device, latex_tokenizer)
    if not model:
        print("Failed to load model. Exiting.")
        return

    # 3. Get Image Transforms
    image_transform = get_image_transforms()

    # 4. Predict
    print(f"\nPredicting LaTeX for image: {args.image_path}")
    predicted_latex = predict(model, args.image_path, latex_tokenizer, device, image_transform)

    if predicted_latex is not None:
        print("\n--- Predicted LaTeX ---")
        print(predicted_latex)
        print("-----------------------\n")
    else:
        print("Prediction failed.")

if __name__ == '__main__':
    main()