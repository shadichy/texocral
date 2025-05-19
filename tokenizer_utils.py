# tokenizer_utils.py
"""
Utilities for creating, training, and loading the LaTeX tokenizer.
"""
import os
from tokenizers import Tokenizer
from tokenizers.models import BPE
from tokenizers.trainers import BpeTrainer
from tokenizers.pre_tokenizers import WhitespaceSplit, ByteLevel
from tokenizers.processors import TemplateProcessing
from transformers import PreTrainedTokenizerFast
from datasets import load_dataset

import config # Assuming config.py is in the same directory or accessible

class LaTeXTokenizer:
    def __init__(self, tokenizer_path=config.TOKENIZER_PATH,
                 pad_token=config.PAD_TOKEN, sos_token=config.SOS_TOKEN,
                 eos_token=config.EOS_TOKEN, unk_token=config.UNK_TOKEN):
        self.tokenizer_path = tokenizer_path
        self.pad_token = pad_token
        self.sos_token = sos_token
        self.eos_token = eos_token
        self.unk_token = unk_token
        self.special_tokens = [self.pad_token, self.sos_token, self.eos_token, self.unk_token]
        self.tokenizer = None

    def train(self, corpus_iterator, vocab_size=config.VOCAB_SIZE):
        """
        Trains a BPE tokenizer from a corpus iterator.
        corpus_iterator should yield strings (LaTeX formulas).
        """
        print(f"Training tokenizer with vocab size: {vocab_size}...")
        # Initialize a BPE model tokenizer
        self.tokenizer_backend = Tokenizer(BPE(unk_token=self.unk_token))
        self.tokenizer_backend.pre_tokenizer = ByteLevel() # Handles various characters well

        trainer = BpeTrainer(
            vocab_size=vocab_size,
            special_tokens=self.special_tokens,
            min_frequency=2 # Adjust as needed
        )
        self.tokenizer_backend.train_from_iterator(corpus_iterator, trainer=trainer)

        # Set up post-processing to add SOS and EOS tokens
        self.tokenizer_backend.post_processor = TemplateProcessing(
            single=f"{self.sos_token} $A {self.eos_token}",
            special_tokens=[
                (self.sos_token, self.tokenizer_backend.token_to_id(self.sos_token)),
                (self.eos_token, self.tokenizer_backend.token_to_id(self.eos_token)),
            ],
        )
        self.save_tokenizer()
        self.load_pretrained_tokenizer() # Reload as PreTrainedTokenizerFast
        print(f"Tokenizer training complete. Saved to {self.tokenizer_path}")
        return self.tokenizer

    def save_tokenizer(self):
        if self.tokenizer_backend:
            # Create directory if it doesn't exist
            os.makedirs(os.path.dirname(self.tokenizer_path), exist_ok=True)
            self.tokenizer_backend.save(self.tokenizer_path)

    def load_pretrained_tokenizer(self):
        if not os.path.exists(self.tokenizer_path):
            print(f"Tokenizer file not found at {self.tokenizer_path}. Please train or provide one.")
            # Fallback to a simple character tokenizer for demonstration if no file exists
            # THIS IS NOT RECOMMENDED FOR A REAL MODEL.
            print("WARNING: Falling back to a very basic character tokenizer. Train a proper one for good results.")
            self._create_dummy_char_tokenizer()
            return self.tokenizer

        self.tokenizer = PreTrainedTokenizerFast(
            tokenizer_file=self.tokenizer_path,
            bos_token=self.sos_token,
            eos_token=self.eos_token,
            pad_token=self.pad_token,
            unk_token=self.unk_token,
        )
        # Update config with dynamic values from the loaded tokenizer
        config.DYNAMIC_VOCAB_SIZE = self.tokenizer.vocab_size
        config.DYNAMIC_PAD_TOKEN_ID = self.tokenizer.pad_token_id
        config.DYNAMIC_SOS_TOKEN_ID = self.tokenizer.bos_token_id
        config.DYNAMIC_EOS_TOKEN_ID = self.tokenizer.eos_token_id
        print(f"Tokenizer loaded from {self.tokenizer_path}")
        print(f"  Vocab size: {config.DYNAMIC_VOCAB_SIZE}")
        print(f"  PAD ID: {config.DYNAMIC_PAD_TOKEN_ID}, SOS ID: {config.DYNAMIC_SOS_TOKEN_ID}, EOS ID: {config.DYNAMIC_EOS_TOKEN_ID}")
        return self.tokenizer

    def _create_dummy_char_tokenizer(self):
        # Placeholder: create a very simple char tokenizer if no file found
        # You should replace this by training a proper tokenizer.
        dummy_corpus = ["\\frac{a}{b}", "x^2 + y^2 = z^2", "\\sum_{i=0}^n i", "\\alpha \\beta \\gamma"]
        all_chars = set()
        for text in dummy_corpus:
            all_chars.update(list(text))
        
        vocab = self.special_tokens + sorted(list(all_chars))
        
        # Create a temporary tokenizer file for PreTrainedTokenizerFast
        temp_tokenizer = Tokenizer(BPE(unk_token=self.unk_token))
        temp_tokenizer.pre_tokenizer = WhitespaceSplit() # Simple pre_tokenizer for char level
        trainer = BpeTrainer(special_tokens=self.special_tokens, vocab_size=len(vocab))
        
        # We need to simulate training with the vocab we constructed
        # This is a hack for the dummy tokenizer. Real training is better.
        def dummy_iterator():
            for char_token in vocab: # Feed individual chars as "words"
                if char_token not in self.special_tokens:
                    yield char_token
            for formula in dummy_corpus: # Feed full formulas too
                 yield formula

        temp_tokenizer.train_from_iterator(dummy_iterator(), trainer=trainer, length=len(vocab) + len(dummy_corpus))

        temp_tokenizer.post_processor = TemplateProcessing(
            single=f"{self.sos_token} $A {self.eos_token}",
            special_tokens=[
                (self.sos_token, temp_tokenizer.token_to_id(self.sos_token)),
                (self.eos_token, temp_tokenizer.token_to_id(self.eos_token)),
            ],
        )
        temp_tokenizer.save(self.tokenizer_path)
        print(f"Dummy character tokenizer created and saved to {self.tokenizer_path}.")
        self.load_pretrained_tokenizer() # Reload it properly

    def encode(self, text_batch, **kwargs):
        if not self.tokenizer:
            self.load_pretrained_tokenizer()
        return self.tokenizer(text_batch, **kwargs)

    def decode(self, token_ids_batch, **kwargs):
        if not self.tokenizer:
            self.load_pretrained_tokenizer()
        return self.tokenizer.batch_decode(token_ids_batch, **kwargs)

    @property
    def vocab_size(self):
        if not self.tokenizer:
            self.load_pretrained_tokenizer()
        return self.tokenizer.vocab_size if self.tokenizer else 0

    @property
    def pad_token_id(self):
        if not self.tokenizer:
            self.load_pretrained_tokenizer()
        return self.tokenizer.pad_token_id if self.tokenizer else 0
    
    @property
    def sos_token_id(self):
        if not self.tokenizer: self.load_pretrained_tokenizer()
        return self.tokenizer.bos_token_id if self.tokenizer else 1

    @property
    def eos_token_id(self):
        if not self.tokenizer: self.load_pretrained_tokenizer()
        return self.tokenizer.eos_token_id if self.tokenizer else 2


def get_tokenizer(force_train=False):
    """Helper function to get an initialized tokenizer."""
    tokenizer_wrapper = LaTeXTokenizer()

    if force_train or not os.path.exists(config.TOKENIZER_PATH):
        print("Attempting to train a new tokenizer as one was not found or training is forced.")
        # Load dataset for training tokenizer
        # IMPORTANT: Adjust this to your actual dataset loading for the FORMULA_COLUMN
        try:
            print(f"Loading dataset '{config.DATASET_NAME}' for tokenizer training...")
            # Load only the training split and the formula column for tokenizer training
            # This might be memory intensive for very large datasets. Consider streaming or subsets.
            # dataset = load_dataset(config.DATASET_NAME, name=config.DATASET_CONFIG, split=config.TRAIN_SPLIT, streaming=True)
            dataset = load_dataset(config.DATASET_NAME, name=config.DATASET_CONFIG, split=config.TRAIN_SPLIT)

            def corpus_iterator():
                # Ensure your dataset is not too large to iterate this way for training.
                # For very large datasets, consider dataset.iter() or saving formulas to a file first.
                count = 0
                max_formulas_for_tokenizer = 100000 # Limit for safety, adjust
                for example in dataset:
                    if config.FORMULA_COLUMN in example and example[config.FORMULA_COLUMN]:
                        yield example[config.FORMULA_COLUMN]
                        count +=1
                        if count >= max_formulas_for_tokenizer:
                            print(f"Reached max {max_formulas_for_tokenizer} formulas for tokenizer training.")
                            break
                    elif count ==0 and (config.FORMULA_COLUMN not in example or not example[config.FORMULA_COLUMN]):
                        print(f"Warning: Formula column '{config.FORMULA_COLUMN}' not found or empty in the first example.")
                        break

            tokenizer_wrapper.train(corpus_iterator(), vocab_size=config.VOCAB_SIZE)
        except Exception as e:
            print(f"Could not load dataset or train tokenizer automatically: {e}")
            print("Please ensure your Hugging Face dataset is correctly specified in config.py")
            print("and you are logged in (`huggingface-cli login`) if it's a private dataset.")
            print("Falling back to loading (or creating a dummy) tokenizer.")
            tokenizer_wrapper.load_pretrained_tokenizer() # Attempt to load or create dummy
    else:
        tokenizer_wrapper.load_pretrained_tokenizer()

    # Critical: Update config with actual values from the loaded/trained tokenizer
    if tokenizer_wrapper.tokenizer:
        config.DYNAMIC_VOCAB_SIZE = tokenizer_wrapper.vocab_size
        config.DYNAMIC_PAD_TOKEN_ID = tokenizer_wrapper.pad_token_id
        config.DYNAMIC_SOS_TOKEN_ID = tokenizer_wrapper.sos_token_id
        config.DYNAMIC_EOS_TOKEN_ID = tokenizer_wrapper.eos_token_id
    else: # Fallback if everything failed
        print("ERROR: Tokenizer could not be initialized.")
        config.DYNAMIC_VOCAB_SIZE = config.VOCAB_SIZE # Fallback
        config.DYNAMIC_PAD_TOKEN_ID = config.PAD_TOKEN_ID
        config.DYNAMIC_SOS_TOKEN_ID = config.SOS_TOKEN_ID
        config.DYNAMIC_EOS_TOKEN_ID = config.EOS_TOKEN_ID


    return tokenizer_wrapper

if __name__ == '__main__':
    # Example usage:
    # To force training a new tokenizer (assuming you have the dataset configured):
    # print("Training a new tokenizer...")
    # latex_tokenizer = get_tokenizer(force_train=True)

    print("Loading existing tokenizer (or creating dummy if not found)...")
    latex_tokenizer = get_tokenizer()

    if latex_tokenizer.tokenizer:
        print(f"Tokenizer loaded. Vocab size: {latex_tokenizer.vocab_size}")
        sample_text = "\\frac{1}{N} \\sum_{i=1}^{N} x_i"
        encoded = latex_tokenizer.encode(sample_text, add_special_tokens=False) # Post-processor adds special tokens
        print(f"Sample: '{sample_text}'")
        print(f"Encoded IDs: {encoded['input_ids']}")
        decoded = latex_tokenizer.decode([encoded['input_ids']], skip_special_tokens=False)
        print(f"Decoded: {decoded[0]}")
        decoded_skip_special = latex_tokenizer.decode([encoded['input_ids']], skip_special_tokens=True)
        print(f"Decoded (skip special): {decoded_skip_special[0]}")
    else:
        print("Failed to initialize tokenizer.")