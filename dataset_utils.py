# dataset_utils.py
"""
Dataset loading, preprocessing, and DataLoader creation.
"""
import torch
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
from datasets import load_dataset
from PIL import Image
import requests # For loading images from URLs if dataset contains URLs
from io import BytesIO

import config # Assuming config.py is in the same directory
from tokenizer_utils import LaTeXTokenizer

def get_image_transforms():
    return transforms.Compose([
        transforms.Resize(config.IMAGE_SIZE),
        transforms.ToTensor(),
        transforms.Normalize(mean=config.IMAGE_MEAN, std=config.IMAGE_STD)
    ])

class LaTeXDataset(Dataset):
    def __init__(self, hf_dataset, tokenizer: LaTeXTokenizer, image_transform, split_name="train"):
        self.hf_dataset = hf_dataset
        self.tokenizer = tokenizer
        self.image_transform = image_transform
        self.split_name = split_name

        if not self.tokenizer or not self.tokenizer.tokenizer:
             raise ValueError("Tokenizer not initialized properly.")
        if config.DYNAMIC_PAD_TOKEN_ID is None: # Check if tokenizer updated config
            print("Warning: DYNAMIC_PAD_TOKEN_ID not set in config from tokenizer. Ensure tokenizer is loaded.")


    def __len__(self):
        return len(self.hf_dataset)

    def __getitem__(self, idx):
        item = self.hf_dataset[idx]

        # Load image
        # This part needs to be adapted based on how images are stored in your HF dataset
        # (e.g., path, PIL Image object, URL, raw bytes)
        img_data = item[config.IMAGE_COLUMN]
        try:
            if isinstance(img_data, str): # Assuming it's a path or URL
                if img_data.startswith('http'):
                    response = requests.get(img_data)
                    response.raise_for_status()
                    image = Image.open(BytesIO(response.content)).convert('RGB')
                else: # Assuming local path
                    image = Image.open(img_data).convert('RGB')
            elif isinstance(img_data, Image.Image):
                image = img_data.convert('RGB')
            elif isinstance(img_data, dict) and 'bytes' in img_data and img_data['bytes']: # HF datasets image feature
                 image = Image.open(BytesIO(img_data['bytes'])).convert('RGB')
            else: # Fallback or specific handling for your dataset's image format
                # If your dataset directly provides PIL images under 'image' key for example:
                # image = item['image'].convert('RGB')
                raise ValueError(f"Unsupported image data type: {type(img_data)}. Please check IMAGE_COLUMN and dataset structure.")
        except Exception as e:
            print(f"Error loading image for item {idx} from source '{img_data}': {e}")
            # Return a dummy image and skip this item in collate_fn or handle appropriately
            image = Image.new('RGB', config.IMAGE_SIZE, (128, 128, 128)) # Dummy grey image


        pixel_values = self.image_transform(image)

        # Tokenize formula
        formula_str = item[config.FORMULA_COLUMN]
        tokenized_output = self.tokenizer.encode(
            formula_str,
            add_special_tokens=True, # Tokenizer's post-processor handles this
            max_length=config.MAX_SEQ_LEN,
            padding="max_length",
            truncation=True,
            return_tensors="pt" # Return PyTorch tensors
        )
        # input_ids will already have SOS/EOS if post_processor is set correctly
        labels = tokenized_output['input_ids'].squeeze(0) # Remove batch dim

        return {
            "pixel_values": pixel_values,
            "labels": labels,
            "attention_mask": tokenized_output['attention_mask'].squeeze(0) # For decoder if needed, AutoregressiveWrapper handles padding
        }

def get_dataloaders(tokenizer: LaTeXTokenizer, batch_size=config.BATCH_SIZE):
    image_transform = get_image_transforms()

    # Load raw Hugging Face dataset
    # Handle potential errors during dataset loading
    try:
        train_hf_dataset = load_dataset(config.DATASET_NAME, name=config.DATASET_CONFIG, split=config.TRAIN_SPLIT)
        val_hf_dataset = load_dataset(config.DATASET_NAME, name=config.DATASET_CONFIG, split=config.VALIDATION_SPLIT)
    except Exception as e:
        print(f"Failed to load dataset '{config.DATASET_NAME}' with config '{config.DATASET_CONFIG}'. Error: {e}")
        print("Please check your Hugging Face dataset name, configuration, and network access.")
        print("You might need to log in using `huggingface-cli login`.")
        # Return None or empty DataLoaders if dataset loading fails
        return None, None


    train_dataset = LaTeXDataset(train_hf_dataset, tokenizer, image_transform, "train")
    val_dataset = LaTeXDataset(val_hf_dataset, tokenizer, image_transform, "validation")

    # Collate function is implicitly handled by default DataLoader if items are tensors
    # For more complex batching, define a custom collate_fn
    def collate_fn(batch):
        pixel_values = torch.stack([item['pixel_values'] for item in batch])
        labels = torch.stack([item['labels'] for item in batch])
        attention_masks = torch.stack([item['attention_mask'] for item in batch])
        return {"pixel_values": pixel_values, "labels": labels, "attention_mask": attention_masks}


    train_dataloader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, collate_fn=collate_fn, num_workers=2, pin_memory=True)
    val_dataloader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False, collate_fn=collate_fn, num_workers=2, pin_memory=True)

    print(f"Train Dataloader: {len(train_dataloader)} batches of size {batch_size}")
    print(f"Validation Dataloader: {len(val_dataloader)} batches of size {batch_size}")

    return train_dataloader, val_dataloader

if __name__ == '__main__':
    from tokenizer_utils import get_tokenizer
    print("Initializing tokenizer for dataset utils example...")
    lt = get_tokenizer() # Load or train tokenizer

    if lt and lt.tokenizer:
        print("Attempting to get dataloaders...")
        try:
            train_dl, val_dl = get_dataloaders(lt, batch_size=2)

            if train_dl and val_dl:
                print("Successfully created dataloaders.")
                # Test one batch from train_dataloader
                sample_batch = next(iter(train_dl))
                print("Sample batch pixel_values shape:", sample_batch['pixel_values'].shape)
                print("Sample batch labels shape:", sample_batch['labels'].shape)
                print("Sample batch labels (first item):", sample_batch['labels'][0, :20]) # Print first 20 tokens
                decoded_sample = lt.decode(sample_batch['labels'][0].unsqueeze(0).tolist(), skip_special_tokens=False)
                print("Decoded sample (first item):", decoded_sample[0])
            else:
                print("Failed to create dataloaders. Dataset might not be accessible or configured correctly.")
        except Exception as e:
            print(f"An error occurred while testing dataloaders: {e}")
            print("This might be due to issues with the placeholder dataset name or its structure.")
    else:
        print("Tokenizer not available, cannot proceed with dataloader example.")