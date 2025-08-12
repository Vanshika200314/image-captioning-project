import nltk
import pickle
from collections import Counter
from datasets import load_dataset
from tqdm import tqdm

# This class will handle the word-to-index and index-to-word mappings
class Vocabulary:
    def __init__(self, freq_threshold):
        # Initialize the mappings with special tokens
        # <PAD> for padding sentences to the same length
        # <START> to signify the start of a sentence
        # <END> to signify the end of a sentence
        # <UNK> for unknown words not in our vocabulary
        self.itos = {0: "<PAD>", 1: "<START>", 2: "<END>", 3: "<UNK>"}
        self.stoi = {"<PAD>": 0, "<START>": 1, "<END>": 2, "<UNK>": 3}
        self.freq_threshold = freq_threshold

    def __len__(self):
        return len(self.itos)

    # Helper function to get all captions from the dataset
    @staticmethod
    def get_all_captions(dataset):
        all_captions = []
        print("Gathering all captions from the training set...")
        for item in tqdm(dataset):
            # The dataset has separate keys for each caption, so we collect them all
            all_captions.append(item['caption_0'])
            all_captions.append(item['caption_1'])
            all_captions.append(item['caption_2'])
            all_captions.append(item['caption_3'])
            all_captions.append(item['caption_4'])
        return all_captions
   
    # Build the vocabulary from a list of sentences
    def build_vocabulary(self, sentence_list):
        frequencies = Counter()
        idx = 4  # Start index after special tokens

        print("Tokenizing and counting word frequencies...")
        for sentence in tqdm(sentence_list):
            for word in nltk.word_tokenize(sentence.lower()):
                frequencies[word] += 1
        
        print("Building word-to-index mapping...")
        # Only include words that appear at least freq_threshold times
        for word, count in tqdm(frequencies.items()):
            if count >= self.freq_threshold:
                self.stoi[word] = idx
                self.itos[idx] = word
                idx += 1

# This block will only run when you execute `python build_vocab.py`
if __name__ == "__main__":
    print("Starting vocabulary creation process...")
    
    # Download the NLTK tokenizer model (only needs to be done once)
    try:
        nltk.data.find('tokenizers/punkt')
    except nltk.downloader.DownloadError:
        print("Downloading NLTK 'punkt' model...")
        nltk.download('punkt')

    # Load the Flickr8k training data from Hugging Face
    print("Loading Flickr8k dataset from Hugging Face...")
    train_dataset = load_dataset("jxie/flickr8k", split="train")
    print("--- Inspecting a sample from the dataset ---")
    print(train_dataset[0])
    print("-------------------------------------------")
    
    # Get all captions from the dataset
    all_train_captions = Vocabulary.get_all_captions(train_dataset)
    
    # Create a vocabulary instance. We will only keep words that appear at least 5 times.
    vocab = Vocabulary(freq_threshold=5)
    vocab.build_vocabulary(all_train_captions)
    
    # Save the built vocabulary object to a file for later use
    with open("vocab.pkl", "wb") as f:
        pickle.dump(vocab, f)
        
    print("\nVocabulary creation complete!")
    print(f"Total vocabulary size: {len(vocab)}")
    print("Vocabulary saved to 'vocab.pkl'")