# config.py

import torch

DATA_PATH = "data/processed_data.pkl"
MODEL_SAVE_PATH = "saved_models/seq2seq_model.pt"

# Model hyperparameters
USE_ATTENTION = True

# EMBEDDING_DIM = 256
EMBEDDING_DIM = 300
HIDDEN_DIM = 512
N_LAYERS = 2

DROPOUT = 0.3

# Training hyperparameters
BATCH_SIZE = 128
N_EPOCHS = 30

LEARNING_RATE = 0.001
TEACHER_FORCING_RATIO = 0.5

CLIP = 1

DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
PRINT_EVERY = 100

# Early stopping
PATIENCE = 7