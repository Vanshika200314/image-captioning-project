import os
import pickle
import torch
from flask import Flask, request, render_template
from PIL import Image
from torchvision import transforms
from model import EncoderCNN, DecoderRNN

# This class MUST be defined here for pickle to work with the factory
class Vocabulary:
    def __init__(self, freq_threshold):
        self.itos = {0: "<PAD>", 1: "<START>", 2: "<END>", 3: "<UNK>"}
        self.stoi = {k: v for v, k in self.itos.items()}
    def __len__(self): return len(self.itos)

def create_app():
    """Application Factory Function"""
    app = Flask(__name__)
    app.config['UPLOAD_FOLDER'] = 'static/uploads/'

    print("--- Loading custom-trained model inside create_app ---")
    device = torch.device("cpu")

    try:
        with open("vocab.pkl", "rb") as f:
            vocab = pickle.load(f)

        embed_size, hidden_size, vocab_size, encoder_dim = 256, 256, len(vocab), 2048
        encoder = EncoderCNN(embed_size).to(device)
        decoder = DecoderRNN(embed_size, hidden_size, vocab_size, encoder_dim).to(device)
        encoder.load_state_dict(torch.load("encoder-model.pth", map_location=device))
        decoder.load_state_dict(torch.load("decoder-model.pth", map_location=device))
        encoder.eval()
        decoder.eval()

        transform = transforms.Compose([
            transforms.Resize((224, 224)),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
        ])
        print("--- Model loaded successfully! ---")
        model_loaded_successfully = True
    except Exception as e:
        print(f"!!! FATAL ERROR: Could not load model files: {e} !!!")
        model_loaded_successfully = False

    @app.route('/', methods=['GET', 'POST'])
    def index():
        if request.method == 'POST':
            if 'file' not in request.files:
                return render_template('index.html', error="No file selected.")
            file = request.files.get('file')
            if not file or not file.filename:
                return render_template('index.html', error="No file selected.")
            
            filename = file.filename
            if not os.path.exists(app.config['UPLOAD_FOLDER']):
                os.makedirs(app.config['UPLOAD_FOLDER'])
            filepath = os.path.join(app.config['UPLOAD_FOLDER'], filename)
            file.save(filepath)
            
            caption = generate_caption(filepath)
            
            return render_template('index.html', caption=caption, image_filename=filename)
            
        return render_template('index.html')

    def generate_caption(image_path):
        if not model_loaded_successfully:
            return "Error: The AI model failed to load."
        try:
            image = Image.open(image_path).convert("RGB")
            image_tensor = transform(image).unsqueeze(0).to(device)
            with torch.no_grad():
                features = encoder(image_tensor)
                caption_indices = decoder.sample(features)
            caption_words = [vocab.itos[idx] for idx in caption_indices]
            caption = ' '.join(word for word in caption_words if word not in ["<START>", "<END>", "<PAD>"])
            return caption.capitalize() + '.'
        except Exception as e:
            print(f"ERROR generating caption: {e}")
            return "An error occurred during caption generation."

    return app
