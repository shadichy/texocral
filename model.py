# model.py
"""
The OCRModel combining timm vision encoders and x_transformers decoder.
"""
import torch
import torch.nn as nn
import timm
from timm.layers import StdConv2dSame # For explicit use if needed, ResNetV2 uses it internally
from x_transformers import TransformerWrapper, Decoder, AutoregressiveWrapper

import config # Assuming config.py is in the same directory

class OCRModel(nn.Module):
    def __init__(self,
                 vision_encoder_name=config.VISION_ENCODER_NAME,
                 vision_encoder_pretrained=config.VISION_ENCODER_PRETRAINED,
                 decoder_dim=config.DECODER_DIM,
                 decoder_depth=config.DECODER_DEPTH,
                 decoder_heads=config.DECODER_HEADS,
                 vocab_size=None, # Must be provided, from tokenizer
                 max_seq_len=config.MAX_SEQ_LEN,
                 pad_token_id=None # Must be provided, from tokenizer
                 ):
        super().__init__()

        if vocab_size is None or pad_token_id is None:
            raise ValueError("vocab_size and pad_token_id must be provided from the tokenizer.")
        
        self.vision_encoder_name = vision_encoder_name
        self.decoder_dim = decoder_dim

        # Vision Encoder
        if 'vit' in vision_encoder_name.lower():
            self.vision_encoder = timm.create_model(
                vision_encoder_name,
                pretrained=vision_encoder_pretrained,
                num_classes=0  # Remove classification head, get features
            )
            self.encoder_dim = self.vision_encoder.embed_dim
            self.encoder_is_vit = True
            # Project ViT features to decoder_dim if they differ
            if self.encoder_dim != decoder_dim:
                self.encoder_proj = nn.Linear(self.encoder_dim, decoder_dim)
            else:
                self.encoder_proj = nn.Identity()

        elif 'resnet50' in vision_encoder_name.lower():
            self.vision_encoder = timm.create_model(
            vision_encoder_name,
            pretrained=vision_encoder_pretrained,
            features_only=True,
            out_indices=(-1,)
        )
            # ResNetV2 uses StdConv2d internally.
            # The output of features_only=True, out_indices=(-1,) is a list with one tensor:
            # [batch_size, num_features, H, W]
            self.encoder_dim = self.vision_encoder.feature_info[-1]['num_chs'] # e.g., 2048 for ResNet50v2
            self.encoder_is_vit = False

            # Feature processor for ResNet: project channels and flatten
            # This StdConv2dSame is an *additional* processing layer here.
            self.feature_processor = nn.Sequential(
                StdConv2dSame(self.encoder_dim, decoder_dim, kernel_size=1, bias=False), # Project channels to decoder_dim
                nn.BatchNorm2d(decoder_dim), # Add BatchNorm after conv
                nn.GELU() # Activation
            )
        else:
            raise ValueError(f"Unsupported vision_encoder_name: {vision_encoder_name}")

        # Decoder (x_transformers)
        # The context_dim for cross-attention in the Decoder should match the output dim of the encoder features
        # after any projection (i.e., decoder_dim).
        _decoder_instance = Decoder(
            dim=decoder_dim,
            depth=decoder_depth,
            heads=decoder_heads,
            cross_attend=True,
            attn_dropout=0.1,
            ff_dropout=0.1,
            rotary_pos_emb=True, # Recommended for better performance
            # context_dim will default to `dim` if not specified, which is what we want here
            # as encoder features are projected to decoder_dim.
        )

        self.decoder_transformer = TransformerWrapper(
            num_tokens=vocab_size,
            max_seq_len=max_seq_len,
            attn_layers=_decoder_instance,
            emb_dim=decoder_dim, # Embedding dimension for target tokens
            emb_dropout=0.1
        )

        self.autoregressive_wrapper = AutoregressiveWrapper(
            self.decoder_transformer,
            pad_value=pad_token_id,
            # max_seq_len=max_seq_len # Also used by generate method
        )
        self.pad_token_id = pad_token_id


    def forward(self, img_tensor, tgt_sequence=None, generate=False, return_outputs=False, **kwargs_generate):
        # img_tensor: (batch, channels, height, width)
        # tgt_sequence: (batch, seq_len) - for training (ground truth token IDs)

        # Encode image
        if self.encoder_is_vit:
            encoded_features = self.vision_encoder(img_tensor)
            encoded_features = self.encoder_proj(encoded_features)
        else: # ResNet-like
            feature_maps_list = self.vision_encoder(img_tensor)
            feature_maps = feature_maps_list[0]
            processed_features = self.feature_processor(feature_maps)
            batch_size, channels, height, width = processed_features.shape
            encoded_features = processed_features.permute(0, 2, 3, 1).reshape(batch_size, height * width, channels)
            
        # print(f"DEBUG: OCRModel - encoded_features.shape: {encoded_features.shape}, encoded_features.ndim: {encoded_features.ndim}")


        # Decode
        if generate:
            if 'start_tokens' not in kwargs_generate:
                start_tokens = torch.ones((img_tensor.shape[0], 1), dtype=torch.long, device=img_tensor.device) * config.DYNAMIC_SOS_TOKEN_ID
                kwargs_generate['start_tokens'] = start_tokens
            # Note: The `generate` method itself might have its own `return_outputs` or similar,
            # but typically it just returns generated sequences.
            # The `return_outputs` here is for the training path.
            prompts = kwargs_generate.pop('start_tokens', None)
            if prompts is None:
                prompts = torch.full(
                    (img_tensor.size(0), 1),
                    config.DYNAMIC_SOS_TOKEN_ID,
                    dtype=torch.long,
                    device=img_tensor.device
                )

            # 2) Now call generate with only the supported args
            return self.autoregressive_wrapper.generate(
                prompts,
                context=encoded_features,
                seq_len=kwargs_generate.pop('seq_len', config.GENERATION_MAX_LEN),
                temperature=kwargs_generate.pop('temperature', config.GENERATION_TEMPERATURE),
                # filter_thres=kwargs_generate.pop('filter_thres', config.GENERATION_FILTER_THRES),
                eos_token=kwargs_generate.pop(
                    'eos_token',
                    config.DYNAMIC_EOS_TOKEN_ID if config.DYNAMIC_EOS_TOKEN_ID is not None else config.EOS_TOKEN_ID
                )
            )
        else:
            # print(f"DEBUG: OCRModel - About to call autoregressive_wrapper. tgt_sequence.shape: {tgt_sequence.shape}")
            feature_maps_list = self.vision_encoder(img_tensor)
            feature_maps = feature_maps_list[0] # Should be (B, C_enc, H_enc, W_enc) - 4D
            processed_features = self.feature_processor(feature_maps) # Should be (B, decoder_dim, H, W) - 4D
        
            batch_size, channels, height, width = processed_features.shape
            
            # THIS IS THE CRITICAL SECTION WHERE THE 3D TENSOR SHOULD BE FORMED
            # Correct version:
            encoded_features = processed_features.permute(0, 2, 3, 1).reshape(batch_size, height * width, channels)
            # This should result in (batch_size, sequence_length, channels), e.g., (16, H*W, 512)
            # For training, pass tgt_sequence as the first positional argument (x)
            # and explicitly pass return_outputs.
            # AutoregressiveWrapper typically returns loss if return_outputs=False (its usual default)
            output = self.autoregressive_wrapper(
                tgt_sequence, # Changed from text=tgt_sequence
                context=encoded_features,
                return_outputs=return_outputs # Pass the argument down
            )
            return output # This will be the loss if return_outputs=False

if __name__ == '__main__':
    # This is a very basic test.
    # Ensure your tokenizer is available and config.DYNAMIC_* values are set.
    from tokenizer_utils import get_tokenizer
    print("Initializing tokenizer for model example...")
    latex_tokenizer = get_tokenizer() # This will load or create dummy and set config vars

    if not config.DYNAMIC_VOCAB_SIZE or config.DYNAMIC_PAD_TOKEN_ID is None:
        print("CRITICAL: Tokenizer did not set dynamic config values. Exiting model test.")
    else:
        print(f"Using vocab size: {config.DYNAMIC_VOCAB_SIZE}, pad_id: {config.DYNAMIC_PAD_TOKEN_ID}")
        # Test with ResNet50v2
        print("\n--- Testing with ResNet50v2 ---")
        try:
            model_resnet = OCRModel(
                vision_encoder_name='resnet50v2',
                vocab_size=config.DYNAMIC_VOCAB_SIZE,
                pad_token_id=config.DYNAMIC_PAD_TOKEN_ID,
                max_seq_len=config.MAX_SEQ_LEN
            ).to(config.DEVICE)
            model_resnet.eval() # Set to eval mode for generation

            dummy_image = torch.randn(2, 3, config.IMAGE_SIZE[0], config.IMAGE_SIZE[1]).to(config.DEVICE)
            dummy_target_seq = torch.randint(0, config.DYNAMIC_VOCAB_SIZE, (2, config.MAX_SEQ_LEN)).to(config.DEVICE)
            dummy_target_seq[:, 0] = config.DYNAMIC_SOS_TOKEN_ID # Start with SOS
            dummy_target_seq[:, -1] = config.DYNAMIC_EOS_TOKEN_ID # End with EOS
            dummy_target_seq[:, 10:] = config.DYNAMIC_PAD_TOKEN_ID # Example padding

            # Test training pass (returns loss)
            model_resnet.train()
            loss = model_resnet(dummy_image, dummy_target_seq)
            print(f"ResNet - Training pass, Loss: {loss.item()}")

            # Test generation pass
            model_resnet.eval()
            with torch.no_grad():
                generated_ids = model_resnet(dummy_image, generate=True, seq_len=config.GENERATION_MAX_LEN, temperature=0.7)
                print(f"ResNet - Generated IDs shape: {generated_ids.shape}") # (batch_size, max_seq_len)
                if latex_tokenizer and latex_tokenizer.tokenizer:
                     decoded = latex_tokenizer.decode(generated_ids.cpu().tolist(), skip_special_tokens=True)
                     print(f"ResNet - Decoded sample 0: {decoded[0]}")
                     print(f"ResNet - Decoded sample 1: {decoded[1]}")


        except Exception as e:
            print(f"Error testing ResNet50v2 model: {e}")
            import traceback
            traceback.print_exc()

        # Test with ViT
        print("\n--- Testing with ViT ---")
        try:
            model_vit = OCRModel(
                vision_encoder_name=config.VISION_ENCODER_NAME, # default is ViT from config
                vocab_size=config.DYNAMIC_VOCAB_SIZE,
                pad_token_id=config.DYNAMIC_PAD_TOKEN_ID,
                max_seq_len=config.MAX_SEQ_LEN
            ).to(config.DEVICE)
            model_vit.eval()

            # Test training pass (returns loss)
            model_vit.train()
            loss_vit = model_vit(dummy_image, dummy_target_seq)
            print(f"ViT - Training pass, Loss: {loss_vit.item()}")

            # Test generation pass
            model_vit.eval()
            with torch.no_grad():
                generated_ids_vit = model_vit(dummy_image, generate=True, seq_len=config.GENERATION_MAX_LEN)
                print(f"ViT - Generated IDs shape: {generated_ids_vit.shape}")
                if latex_tokenizer and latex_tokenizer.tokenizer:
                     decoded_vit = latex_tokenizer.decode(generated_ids_vit.cpu().tolist(), skip_special_tokens=True)
                     print(f"ViT - Decoded sample 0: {decoded_vit[0]}")
                     print(f"ViT - Decoded sample 1: {decoded_vit[1]}")


        except Exception as e:
            print(f"Error testing ViT model: {e}")
            import traceback
            traceback.print_exc()