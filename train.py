# train.py
"""
Training script for the LaTeX OCR model.
"""
import torch
import torch.optim as optim
from torch.utils.data import DataLoader
import os
import numpy as np
from tqdm import tqdm
import time

import config
from tokenizer_utils import get_tokenizer
from dataset_utils import get_dataloaders
from model import OCRModel

# For reproducibility
torch.manual_seed(42)
np.random.seed(42)

def save_checkpoint(model, optimizer, epoch, loss, is_best=False, latest=True):
    os.makedirs(config.MODEL_SAVE_DIR, exist_ok=True)
    checkpoint_path = os.path.join(config.MODEL_SAVE_DIR, config.LATEST_MODEL_NAME)
    
    simple_config = {}
    for attr in dir(config):
        if attr.startswith('__'):
            continue
        val = getattr(config, attr)
        # Only save primitives and containers of primitives
        if isinstance(val, (int, float, str, bool, list, dict, type(None))):
            simple_config[attr] = val

    payload = {
        'epoch': epoch,
        'model_state_dict': model.state_dict(),
        'optimizer_state_dict': optimizer.state_dict(),
        'loss': loss,
        'config': simple_config
    }
    if latest:
        torch.save(payload, checkpoint_path)
        print(f"Saved latest checkpoint to {checkpoint_path}")
    
    if is_best:
        best_checkpoint_path = os.path.join(config.MODEL_SAVE_DIR, config.BEST_MODEL_NAME)
        torch.save(payload, best_checkpoint_path)
        print(f"Saved BEST checkpoint to {best_checkpoint_path}")


def load_checkpoint(model, optimizer, device):
    checkpoint_path = os.path.join(config.MODEL_SAVE_DIR, config.LATEST_MODEL_NAME)
    start_epoch = 0
    min_val_loss = float('inf') # Or load from checkpoint if saved

    if os.path.exists(checkpoint_path):
        try:
            print(f"Loading checkpoint from {checkpoint_path}")
            checkpoint = torch.load(checkpoint_path, map_location=device)
            model.load_state_dict(checkpoint['model_state_dict'])
            if optimizer and 'optimizer_state_dict' in checkpoint:
                 optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
            start_epoch = checkpoint['epoch'] + 1
            if 'loss' in checkpoint: # Could be validation loss
                min_val_loss = checkpoint.get('val_loss', checkpoint['loss']) # Prefer 'val_loss' if it exists
            print(f"Resuming training from epoch {start_epoch}. Previous loss: {min_val_loss:.4f}")
        except Exception as e:
            print(f"Error loading checkpoint: {e}. Starting from scratch.")
            start_epoch = 0
            min_val_loss = float('inf')
    else:
        print("No checkpoint found. Starting from scratch.")
    
    return start_epoch, min_val_loss


def train_one_epoch(model, dataloader, optimizer, device, epoch_num):
    model.train()
    total_loss = 0
    progress_bar = tqdm(dataloader, desc=f"Epoch {epoch_num+1} [TRAIN]", leave=False)

    for batch_idx, batch in enumerate(progress_bar):
        images = batch['pixel_values'].to(device)
        formulas = batch['labels'].to(device) # Target sequences with SOS, EOS, PAD

        optimizer.zero_grad()
        loss = model(images, formulas, return_outputs = False) # Model's forward in training mode returns loss

        if torch.isnan(loss) or torch.isinf(loss):
            print(f"Warning: NaN or Inf loss encountered at batch {batch_idx}. Skipping update.")
            # Optionally: print details of inputs/outputs if this happens frequently
            # print("Image shape:", images.shape)
            # print("Formula sample (first 10 tokens):", formulas[0, :10])
            # with torch.no_grad():
            #     enc_out = model.vision_encoder(images) # Check encoder output
            #     if model.encoder_is_vit: enc_out = model.encoder_proj(enc_out)
            #     else: # resnet
            #         f_list = model.vision_encoder(images); f_map = f_list[0]
            #         p_feat = model.feature_processor(f_map)
            #         bs,c,h,w = p_feat.shape
            #         enc_out = p_feat.permute(0,2,3,1).reshape(bs,h*w,c)
            #     print("Encoder output sample min/max/mean:", enc_out.min(), enc_out.max(), enc_out.mean())
            continue # Skip backpropagation for this batch

        loss.backward()

        if config.GRAD_CLIP_NORM > 0:
            torch.nn.utils.clip_grad_norm_(model.parameters(), config.GRAD_CLIP_NORM)
        
        optimizer.step()
        
        total_loss += loss.item()
        progress_bar.set_postfix(loss=loss.item())
        
    return total_loss / len(dataloader)

def validate_one_epoch(model, dataloader, device, epoch_num):
    model.eval()
    total_loss = 0
    progress_bar = tqdm(dataloader, desc=f"Epoch {epoch_num+1} [VALID]", leave=False)

    with torch.no_grad():
        for batch in progress_bar:
            images = batch['pixel_values'].to(device)
            formulas = batch['labels'].to(device)
            
            loss = model(images, formulas) # Model's forward in eval mode also returns loss
            if torch.isnan(loss) or torch.isinf(loss):
                print(f"Warning: NaN or Inf validation loss encountered. Skipping.")
                continue

            total_loss += loss.item()
            progress_bar.set_postfix(loss=loss.item())
            
    return total_loss / len(dataloader)

def main():
    print("Starting training process...")
    print(f"Device: {config.DEVICE}")

    # 1. Initialize Tokenizer
    print("\n--- Initializing Tokenizer ---")
    latex_tokenizer = get_tokenizer(force_train=False) # Set force_train=True if you want to retrain
    if not latex_tokenizer or not latex_tokenizer.tokenizer:
        print("Failed to initialize tokenizer. Exiting.")
        return
    if config.DYNAMIC_VOCAB_SIZE is None or config.DYNAMIC_PAD_TOKEN_ID is None:
        print("CRITICAL: Tokenizer did not update dynamic config values (VOCAB_SIZE, PAD_TOKEN_ID).")
        print("Please ensure your tokenizer is trained and loaded correctly.")
        print("Using fallback static config values, but this is likely incorrect.")
        # Use static config values as a last resort, but this is problematic
        current_vocab_size = config.VOCAB_SIZE
        current_pad_id = config.PAD_TOKEN_ID
        current_sos_id = config.SOS_TOKEN_ID
    else:
        current_vocab_size = config.DYNAMIC_VOCAB_SIZE
        current_pad_id = config.DYNAMIC_PAD_TOKEN_ID
        current_sos_id = config.DYNAMIC_SOS_TOKEN_ID

    print(f"Using Tokenizer - Vocab Size: {current_vocab_size}, PAD ID: {current_pad_id}, SOS ID: {current_sos_id}")

    # 2. Create DataLoaders
    print("\n--- Creating DataLoaders ---")
    train_dataloader, val_dataloader = get_dataloaders(latex_tokenizer, batch_size=config.BATCH_SIZE)
    if not train_dataloader or not val_dataloader:
        print("Failed to create dataloaders. Ensure dataset is configured correctly and accessible. Exiting.")
        return

    # 3. Initialize Model
    print("\n--- Initializing Model ---")
    model = OCRModel(
        vision_encoder_name=config.VISION_ENCODER_NAME,
        vision_encoder_pretrained=config.VISION_ENCODER_PRETRAINED,
        decoder_dim=config.DECODER_DIM,
        decoder_depth=config.DECODER_DEPTH,
        decoder_heads=config.DECODER_HEADS,
        vocab_size=current_vocab_size,
        max_seq_len=config.MAX_SEQ_LEN,
        pad_token_id=current_pad_id
    ).to(config.DEVICE)
    print(f"Model: {config.VISION_ENCODER_NAME} encoder, {config.DECODER_DEPTH}-layer Transformer decoder.")
    num_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"Total trainable parameters: {num_params / 1e6:.2f} M")


    # 4. Optimizer and Scheduler
    optimizer = optim.AdamW(model.parameters(), lr=config.LEARNING_RATE, weight_decay=config.WEIGHT_DECAY)
    # scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, 'min', patience=3, factor=0.5, verbose=True)
    scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=config.NUM_EPOCHS * len(train_dataloader), eta_min=1e-6)


    # 5. Load Checkpoint (if any)
    start_epoch, min_val_loss = load_checkpoint(model, optimizer, config.DEVICE)


    # 6. Training Loop
    print("\n--- Starting Training ---")
    for epoch in range(start_epoch, config.NUM_EPOCHS):
        epoch_start_time = time.time()
        
        train_loss = train_one_epoch(model, train_dataloader, optimizer, config.DEVICE, epoch)
        val_loss = validate_one_epoch(model, val_dataloader, config.DEVICE, epoch)
        
        epoch_duration = time.time() - epoch_start_time
        
        print(f"Epoch {epoch+1}/{config.NUM_EPOCHS} | "
              f"Train Loss: {train_loss:.4f} | Val Loss: {val_loss:.4f} | "
              f"Time: {epoch_duration:.2f}s | LR: {optimizer.param_groups[0]['lr']:.2e}")

        # Update learning rate scheduler that steps per epoch
        # if isinstance(scheduler, optim.lr_scheduler.ReduceLROnPlateau):
        #    scheduler.step(val_loss)
        # For CosineAnnealingLR or similar, step is usually done per batch or per epoch if T_max is num_epochs
        # If T_max is total steps, then scheduler.step() is called after optimizer.step() in train_one_epoch.
        # For this setup where T_max = config.NUM_EPOCHS * len(train_dataloader), call scheduler.step() after each batch.
        # The current setup calls it per epoch which is fine if T_max = num_epochs.
        # Let's adjust CosineAnnealing to be per epoch for simplicity here.
        # Re-init scheduler if it's CosineAnnealingLR for per-epoch step:
        if not isinstance(scheduler, optim.lr_scheduler.ReduceLROnPlateau): # If it's not ReduceLROnPlateau
             scheduler.step() # For schedulers like CosineAnnealingLR with T_max=NUM_EPOCHS

        # Save checkpoint
        is_best = val_loss < min_val_loss
        if is_best:
            min_val_loss = val_loss
        save_checkpoint(model, optimizer, epoch, val_loss, is_best=is_best, latest=True)

    print("Training complete.")
    print(f"Best validation loss: {min_val_loss:.4f}")
    print(f"Best model saved to: {os.path.join(config.MODEL_SAVE_DIR, config.BEST_MODEL_NAME)}")

if __name__ == '__main__':
    main()