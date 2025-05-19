# config.py
"""
Configuration settings for the LaTeX OCR model.
"""

import torch

# --- Dataset Configuration ---
# TODO: Replace with your actual Hugging Face dataset name and any specific configuration/split.
DATASET_NAME = "unsloth/LaTeX_OCR" # e.g., "harvard-npl/im2latex-100k" (check availability and format)
DATASET_CONFIG = None # e.g., "default" or specific subset name if applicable
IMAGE_COLUMN = "image" # Column name in the dataset for the image (or image path)
FORMULA_COLUMN = "text" # Column name in the dataset for the LaTeX string
TRAIN_SPLIT = "train" # e.g., "train", "train_val[:80%]"
VALIDATION_SPLIT = "test" # e.g., "validation", "train_val[80%:]"
TEST_SPLIT = None # e.g., "test"

# --- Tokenizer Configuration ---
TOKENIZER_PATH = "./latex_tokenizer.json" # Path to save/load the trained tokenizer
VOCAB_SIZE = 5000 # Target vocabulary size (if training a new tokenizer)
# Special tokens - ensure your tokenizer uses these or similar ones
PAD_TOKEN = "<PAD>"
SOS_TOKEN = "<SOS>"
EOS_TOKEN = "<EOS>"
UNK_TOKEN = "<UNK>"
PAD_TOKEN_ID = 0
SOS_TOKEN_ID = 1
EOS_TOKEN_ID = 2
UNK_TOKEN_ID = 3


# --- Image Preprocessing ---
IMAGE_SIZE = (224, 224) # Target image size (height, width)
IMAGE_MEAN = [0.485, 0.456, 0.406] # ImageNet mean
IMAGE_STD = [0.229, 0.224, 0.225] # ImageNet std

# --- Model Configuration ---
# Vision Encoder options: 'resnet50v2', 'vit_small_r26_s32_224', 'vit_base_patch16_224', etc.
# VISION_ENCODER_NAME = 'vit_small_r26_s32_224' # or 'resnet50'
VISION_ENCODER_NAME = 'resnet50' # or 'resnet50'
VISION_ENCODER_PRETRAINED = True

# Decoder (x_transformers)
DECODER_DIM = 512 # Dimension of the decoder
DECODER_DEPTH = 6 # Number of decoder layers
DECODER_HEADS = 8 # Number of attention heads
MAX_SEQ_LEN = 256 # Maximum length of the LaTeX sequence (including SOS/EOS)

# --- Training Configuration ---
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
BATCH_SIZE = 16 # Adjust based on your GPU memory
NUM_EPOCHS = 20
LEARNING_RATE = 3e-4 # Initial learning rate
WEIGHT_DECAY = 1e-2
GRAD_CLIP_NORM = 1.0 # Max norm for gradient clipping, 0 for no clipping

# --- Inference Configuration ---
GENERATION_MAX_LEN = MAX_SEQ_LEN
GENERATION_TEMPERATURE = 0.7 # For sampling
GENERATION_FILTER_THRES = 0.9 # For nucleus sampling (top_p)

# --- Paths ---
MODEL_SAVE_DIR = "./trained_models"
BEST_MODEL_NAME = "best_latex_ocr_model.pth"
LATEST_MODEL_NAME = "latest_latex_ocr_model.pth"

# Ensure these match your tokenizer after it's built/loaded
# These will be updated by the tokenizer utility if a new one is trained
# or when loaded from file.
DYNAMIC_VOCAB_SIZE = None # Will be set by tokenizer
DYNAMIC_PAD_TOKEN_ID = None # Will be set by tokenizer
DYNAMIC_SOS_TOKEN_ID = None # Will be set by tokenizer
DYNAMIC_EOS_TOKEN_ID = None # Will be set by tokenizer