# In build_vocab_json.py
import pickle
import json

# This script opens your existing vocab.pkl file and saves its
# contents to a reliable vocab.json file.
class Vocabulary:
    def __init__(self, freq_threshold):
        self.itos = {0: "<PAD>", 1: "<START>", 2: "<END>", 3: "<UNK>"}
        self.stoi = {k: v for v, k in self.itos.items()}
    def __len__(self): return len(self.itos)

print("--- Converting vocab.pkl to vocab.json ---")
with open("vocab.pkl", "rb") as f_pkl:
    vocab_obj = pickle.load(f_pkl)

# The data we need is the word-to-index (stoi) and index-to-word (itos) mappings
vocab_data = {
    'stoi': vocab_obj.stoi,
    'itos': vocab_obj.itos
}

with open("vocab.json", "w") as f_json:
    json.dump(vocab_data, f_json)

print("--- vocab.json created successfully. You can now delete vocab.pkl. ---")