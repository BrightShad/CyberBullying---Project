import torch

# Load the file into a variable
checkpoint = torch.load("models/best_model.pt", map_location="cpu", weights_only=False)

# Look at what keys we saved inside it!
print(checkpoint.keys())
# Output: dict_keys(['model_state', 'vocab_path', 'label_map', 'embed_dim', 'hidden_dim', 'dropout', 'task'])

# Check what the model was trained for
print("Task:", checkpoint["task"])

# Look at the actual shape of the first layer's weights
print("Embedding Weights Shape:", checkpoint["model_state"]["embedding.weight"].shape)