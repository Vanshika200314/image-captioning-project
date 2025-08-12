"""
This is the final and definitive training script for the custom model.
It includes the advanced Attention architecture and adds two professional
techniques: Gradient Clipping and a Learning Rate Scheduler, to ensure
the most stable and effective training process possible.
"""
import torch
import torch.nn as nn
import torch.optim as optim
# Import the learning rate scheduler
from torch.optim.lr_scheduler import StepLR
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
from PIL import Image
from datasets import load_dataset
import pickle
from tqdm import tqdm
import nltk

from model import EncoderCNN, DecoderRNN

class Vocabulary:
    def __init__(self, freq_threshold):
        self.itos = {0: "<PAD>", 1: "<START>", 2: "<END>", 3: "<UNK>"}
        self.stoi = {"<PAD>": 0, "<START>": 1, "<END>": 2, "<UNK>": 3}
        self.freq_threshold = freq_threshold
    def __len__(self):
        return len(self.itos)

class Flickr8kDataset(Dataset):
    def __init__(self, dataset, vocab, transform=None):
        self.dataset = dataset
        self.vocab = vocab
        self.transform = transform
    def __len__(self):
        return len(self.dataset)
    def __getitem__(self, index):
        item = self.dataset[index]
        image = item['image'].convert("RGB")
        if self.transform:
            image = self.transform(image)
        caption = item['caption_0']
        tokens = nltk.word_tokenize(str(caption).lower())
        numerical_caption = []
        numerical_caption.append(self.vocab.stoi["<START>"])
        numerical_caption.extend([self.vocab.stoi.get(token, self.vocab.stoi["<UNK>"]) for token in tokens])
        numerical_caption.append(self.vocab.stoi["<END>"])
        return image, torch.tensor(numerical_caption)

class MyCollate:
    def __init__(self, pad_idx):
        self.pad_idx = pad_idx
    def __call__(self, batch):
        imgs = [item[0].unsqueeze(0) for item in batch]
        imgs = torch.cat(imgs, dim=0)
        targets = [item[1] for item in batch]
        targets = torch.nn.utils.rnn.pad_sequence(targets, batch_first=True, padding_value=self.pad_idx)
        return imgs, targets

def train():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"--- Using device: {device} ---")
    
    transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ])

    print("--- Loading vocabulary and dataset ---")
    with open("vocab.pkl", "rb") as f:
        vocab = pickle.load(f)
    
    raw_train_dataset = load_dataset("jxie/flickr8k", split="train")
    train_dataset = Flickr8kDataset(raw_train_dataset, vocab, transform)
    
    pad_idx = vocab.stoi["<PAD>"]
    train_loader = DataLoader(
        dataset=train_dataset,
        batch_size=32,
        shuffle=True,
        collate_fn=MyCollate(pad_idx=pad_idx)
    )

    print("--- Initializing Attention-based models ---")
    embed_size = 256
    hidden_size = 256
    vocab_size = len(vocab)
    encoder_dim = 2048
    num_epochs = 20 # Let's give it a solid number of epochs to learn properly
    learning_rate = 1e-3 # We can start with a slightly higher learning rate now

    encoder = EncoderCNN(embed_size).to(device)
    decoder = DecoderRNN(embed_size, hidden_size, vocab_size, encoder_dim).to(device)
    
    criterion = nn.CrossEntropyLoss(ignore_index=pad_idx)
    optimizer = optim.Adam(decoder.parameters(), lr=learning_rate)

    # --- NEW: LEARNING RATE SCHEDULER ---
    # This will decrease the learning rate by a factor of 10 every 7 epochs.
    scheduler = StepLR(optimizer, step_size=7, gamma=0.1)

    print("--- Starting Final Training Loop with Gradient Clipping & LR Scheduling ---")
    for epoch in range(num_epochs):
        loop = tqdm(train_loader, total=len(train_loader), leave=True)
        
        for (imgs, captions) in loop:
            imgs = imgs.to(device)
            captions = captions.to(device)

            features = encoder(imgs)
            outputs = decoder(features, captions)
            
            loss = criterion(outputs.reshape(-1, outputs.shape[2]), captions[:, 1:].reshape(-1))
            
            optimizer.zero_grad()
            loss.backward()
            
            # --- NEW: GRADIENT CLIPPING ---
            # This clips the gradients to prevent them from exploding.
            torch.nn.utils.clip_grad_norm_(decoder.parameters(), max_norm=0.5)
            
            optimizer.step()
            
            loop.set_description(f"Epoch [{epoch+1}/{num_epochs}]")
            loop.set_postfix(loss=loss.item())
        
        # --- NEW: STEP THE SCHEDULER ---
        # At the end of each epoch, we step the scheduler.
        scheduler.step()
        print(f"End of Epoch {epoch+1}, current learning rate: {scheduler.get_last_lr()}")
            
    print("--- Training complete. Saving final models. ---")
    torch.save(decoder.state_dict(), "decoder-model.pth")
    torch.save(encoder.state_dict(), "encoder-model.pth")
    print("--- Models saved successfully. ---")

if __name__ == '__main__':
    train()
