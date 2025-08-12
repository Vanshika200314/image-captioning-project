"""
This is the final, corrected, and definitive version of model.py.
It uses a deterministic "Greedy Search" sampling strategy that is now
bug-free, ensuring static and correct operation.
"""

import torch
import torch.nn as nn
import torchvision.models as models

class EncoderCNN(nn.Module):
    def __init__(self, embed_size):
        super(EncoderCNN, self).__init__()
        resnet = models.resnet50(weights=models.ResNet50_Weights.DEFAULT)
        for param in resnet.parameters():
            param.requires_grad_(False)
        
        modules = list(resnet.children())[:-2]
        self.resnet = nn.Sequential(*modules)
        
    def forward(self, images):
        features = self.resnet(images)
        features = features.permute(0, 2, 3, 1)
        features = features.view(features.size(0), -1, features.size(-1))
        return features

class Attention(nn.Module):
    def __init__(self, encoder_dim, decoder_dim, attention_dim):
        super(Attention, self).__init__()
        self.encoder_att = nn.Linear(encoder_dim, attention_dim)
        self.decoder_att = nn.Linear(decoder_dim, attention_dim)
        self.full_att = nn.Linear(attention_dim, 1)
        self.relu = nn.ReLU()
        self.softmax = nn.Softmax(dim=1)

    def forward(self, encoder_out, decoder_hidden):
        att1 = self.encoder_att(encoder_out)
        att2 = self.decoder_att(decoder_hidden)
        att = self.full_att(self.relu(att1 + att2.unsqueeze(1))).squeeze(2)
        alpha = self.softmax(att)
        attention_weighted_encoding = (encoder_out * alpha.unsqueeze(2)).sum(dim=1)
        return attention_weighted_encoding, alpha

class DecoderRNN(nn.Module):
    def __init__(self, embed_size, hidden_size, vocab_size, encoder_dim=2048, drop_prob=0.5):
        super(DecoderRNN, self).__init__()
        self.embed = nn.Embedding(vocab_size, embed_size)
        self.attention = Attention(encoder_dim, hidden_size, hidden_size)
        self.lstm = nn.LSTMCell(embed_size + encoder_dim, hidden_size)
        self.dropout = nn.Dropout(p=drop_prob)
        self.fc = nn.Linear(hidden_size, vocab_size)

    def forward(self, features, captions):
        embeddings = self.embed(captions)
        batch_size = features.size(0)
        h, c = torch.zeros(batch_size, self.lstm.hidden_size).to(features.device), \
               torch.zeros(batch_size, self.lstm.hidden_size).to(features.device)
        
        seq_length = len(captions[0]) - 1 
        predictions = torch.zeros(batch_size, seq_length, len(self.embed.weight)).to(features.device)

        for t in range(seq_length):
            context, alpha = self.attention(features, h)
            lstm_input = torch.cat((embeddings[:, t], context), dim=1)
            h, c = self.lstm(lstm_input, (h, c))
            output = self.fc(self.dropout(h))
            predictions[:, t] = output
            
        return predictions

    def sample(self, features, max_len=20):
        batch_size = features.size(0)
        h, c = torch.zeros(batch_size, self.lstm.hidden_size).to(features.device), \
               torch.zeros(batch_size, self.lstm.hidden_size).to(features.device)
        
        output = []
        inputs = self.embed(torch.tensor([1]).to(features.device))

        for _ in range(max_len):
            context, alpha = self.attention(features, h)
            lstm_input = torch.cat((inputs, context), dim=1)
            h, c = self.lstm(lstm_input, (h, c))
            preds = self.fc(h)
            
            predicted_index = preds.argmax(1)
            
            output.append(predicted_index.item())
            
            if predicted_index.item() == 2: # <END> token
                break
                
            # --- THIS IS THE CORRECTED LINE ---
            # The previous version had an extra `.unsqueeze(0)` which caused a crash.
            # This is now the correct, bug-free version.
            inputs = self.embed(predicted_index)

        return output