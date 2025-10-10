"""
The dataset module contains any dataset representation as subclass of
``torch.utils.data.Dataset``.

Instruction:
---
To give you full flexibility to implement your preprocessing pipeline,
the only class provided is ``PDTBDataset`` which is required to put in
``torch.utils.data.DataLoader``.

Other than this class, you're free and welcome to implement any function
or class needed.
"""

from torch.utils.data import Dataset
import torch
import json
from pathlib import Path 
import numpy as np
from utils import cache_glove_embeddings, load_glove_cache

# This tells PyTorch it is safe to unpickle the nn.Embedding class
torch.serialization.add_safe_globals([torch.nn.Embedding])

class FeatureExtractor():


#        FEATURE SETS
#        connective-only feature = connective
#        arg1 only
#        arg2 only
#        one-hot for conn, word count for args OR log word count for words
#        tf-idf
#        ignore connectives inside args
#        ignore stop words

# CONNECTIVES ONLY
    def __init__(self, instances, outputs):
        self.instances = instances
        self.outputs = outputs
    
    def tokenize(self):
        return [instance.split() for instance in self.instances]

    
    def map_vocab(self):
        vocab = {}
        vocab["<PAD>"] = 0
        i = 1
        # add index for each token
        sentences = self.tokenize()
        for sentence in sentences:
            for token in sentence:
                if token not in vocab:
                    vocab[token] = i
                    i += 1

        return sentences, vocab
    
    
    
    def flatten_vectors(self, word_vectors : torch.Tensor) -> torch.Tensor:
        """
        Flatten a (w x d) matrix of word vectors for one instance 
        to a (1 x d) instance vector by summing each column of
        the individual word vectors.
        w: number of words in instance.
        d: number of dimensions of vector = 300.
        
        Parameters:
            ...
        
        Return:
            torch.Tensor ...
        
        """
        # check for empty sentence
        if word_vectors.shape[0] == 0:
            return torch.zeros(1, 300)    

        flat_vector = torch.mean(word_vectors, dim=0, keepdim=True)

        return flat_vector    


    def featurize(self, encoding='glove', sentence_type='concat') -> torch.Tensor:
        """
        Featurize raw text data into multi-hot matrix.

        Parameters:
            text_data (str): raw text of dataset.
            vocab_len (int): length of vocabulary of text_data
            encoding (str): the kind of features.
                Options include:
                * **'glove'**: GloVe contextualized word embeddings.
                * **'multihot'**: Multi-hot word embeddings.
                * **'random'**: randomized word embeddings.
            sentence_type (str): flat embedding averaged across words
            or concatenated embedding 

        Returns:
            torch.Tensor: 
                if encoding='multihot':
                    A tensor of shape(n, v): words present in the
                    instance have value 1, else 0.
                    n: number of instances in text_data.
                    v: size of vocbulary in all of text_data.
                if encoding='glove' or 'random':
                    A tensor of shape(n, d): flattened word embeddings
                    for each instance.
                    n: number of instances in text_data.
                    d: fixed length of word embedding = 300.
        """
        sentences, vocab_map = self.map_vocab()

        # ---------------- create vocab embeddings ------------ # 
        if encoding=="glove":
            print(f'FEATURIZING: encoding: glove')
            
            # initialize vocab embedding as random distribution 
            vocab_embedding_list = torch.randn(len(vocab_map), 300, dtype=torch.float32)
            # set padding token embeddings to zeros
            if len(vocab_map) > 0:
                vocab_embedding_list[0].zero_()
            
            # initialize list of word indices
            word_indices = []

            # initialize glove cache dictionary
            glove_cache = {}

            # load embeddings from cache OR create embeddings from txt
            CACHE_DIR = Path("glove/glove_cache.pt")
            if CACHE_DIR.exists():
                print("Loading GloVe dictionary cache!")
                glove_cache = load_glove_cache()

            else:
                print("GloVe dictionary cache does not exist, " \
                "extracting embeddings from .txt file")
                
                # extract from GloVe text file 
                # TODO: make sure to push glove cache to github
                glove_cache = cache_glove_embeddings()

            # add word embeddings to vocab embeddings
            for word_embeddings in glove_cache:
                if glove_cache[word_embeddings] in vocab_map:
                    if len(word_embeddings) != 300:
                        print(f"Error: word_embeddings has length {len(word_embeddings)}, expected 300.")
                        continue # skip corrupted vectors
                    idx = vocab_map[word]
                    vocab_embedding_list[idx].copy_(word_embeddings)

            vocab_embedding = torch.nn.Embedding.from_pretrained(vocab_embedding_list, padding_idx=0)

        else: # encoding="random"
            print(f'FEATURIZING: encoding: random')
            vocab_embedding = torch.nn.Embedding(len(vocab_map), 300, padding_idx=0)

        # ---------------- featurize embeddings ------------- #
                        
        if sentence_type == "flat":
            print(f'FEATURIZING: sentence_type: flat')
            # initialize input weights (s x 300)
            features = torch.zeros(len(sentences), 300)

            # 3. create sentence embedding (w x 300)
            for i in range(len(sentences)): 
                word_indices = []
                for word in sentences[i]:
                    word_indices.append(vocab_map[word])
                sentence_weights = vocab_embedding(torch.tensor(
                                            word_indices, dtype=torch.long))
                
                # flatten each sentence by summing columns to 1d vector (1 x 300)
                features[i] = self.flatten_vectors(sentence_weights)
        
        # TODO: implement concatenated word vectors instead of flattened vector
        elif sentence_type == "concat":
            print(f'FEATURIZING: sentence_type: concat')
            
            # cap max sentence,
            # otherwise memory issues
            MAX_W = 50 

            # initialize input features (s x 300*MAX_W)
            features = torch.randn(len(sentences), 300*MAX_W)

            # create sententence embedding (1 x 300*max(w))
            for i in range(len(sentences)): 
                
                # populate sentence indices   
                word_indices = []
                # only populate first MAX_W indices
                if len(sentences[i]) <= MAX_W:
                    # append real word indices
                    for j in range(len(sentences[i])):
                        word_indices.append(vocab_map[sentences[i][j]])
                    
                    # pad the rest of the sentence with vocab_map["<PAD>"]
                    rem_words = MAX_W - len(word_indices)
                    padding = [vocab_map["<PAD>"] for k in range(rem_words)]
                    word_indices.extend(padding)
                else: # sentence over MAX_W tokens, append only 1st MAX_W tokens
                    for j in range(MAX_W):
                        word_indices.append(vocab_map[sentences[i][j]])

                # make sure exactly MAX_W word indices in sentence
                if len(word_indices) != MAX_W:
                    print(f'padded sentence length: {len(word_indices)}')
                    print(f'but it should be:       {MAX_W}')
                    raise ValueError(f'sentence {i} is not correctly padded!')
                
                # add sentence to features tensor
                sentence_tensor = vocab_embedding(torch.tensor(
                                    word_indices, dtype=torch.long))
                features[i] = sentence_tensor.flatten()

        else:
            raise ValueError("sentence_type must be 'flat' or 'concat'!")

        return features

      

        # --------------------


        # random word embedding

        # --- Parameters ---
        VOCAB_SIZE = vocab_len  # The total number of unique words in your vocabulary
        EMBEDDING_DIM = 300 # The desired size (dimension) of each word vector

        # --- Initialization ---
        # This single line creates a weight matrix (the embedding layer)
        # and initializes it with small random numbers (usually uniform distribution)
        embedding_layer = nn.Embedding(
            num_embeddings=VOCAB_SIZE, 
            embedding_dim=EMBEDDING_DIM
            )

        input_indices : torch.Tensor # vocab token IDs for sentences e.g. [10, 52, 74...]

        # Perform the lookup
        embedded_input = embedding_layer(input_indices)
        # Flatten into (1 x 300) sentence vector
        flattened_input = flatten_vectors(embedded_input)
        print(f"Embedded input shape: {embedded_input.shape}")
        
        NotImplemented
    
         

        
    '''
    # create one-hot matrix of connectives: |I| x |C| (instances * connectives)
    connectives = list(set(instance[1] for instance in text_data))
    features = torch.zeros(len(text_data), len(connectives), dtype=torch.float)
    for i in range(len(text_data)):
        for j in range(len(connectives)):
            if text_data[i][1] == connectives[j]:
                features[i, j] = 1
                break
    '''
    



class PDTBDataset(Dataset):
    """Dataset class for the PDTB dataset"""
    def __init__(self, set="train", sentence_type="concat", encoding="glove"):
        super().__init__()
        self.sentences : list[str] = []
        self.senses : list[str] = []
        self.features : torch.Tensor
        path = ""
        if set == "train":
            path = "pdtb/train.json"
        elif set == "validate":
            path = "pdtb/dev.json"
        elif set == "test":
            path = "pdtb/test.json"
        else:
            raise FileNotFoundError("No such file found!")
        
        with open(path, 'r') as f:
            for line in f:
                line = json.loads(line)
                arg1 = line['Arg1']['RawText'].lower()
                conn = line['Connective']['RawText'].lower()
                arg2 = line['Arg2']['RawText'].lower()
                sense = line['Sense']
                self.sentences.append(''.join([arg1, conn, arg2]))
                self.senses.append(sense[0])
        

    def featurize(self, encoding='glove', sentence_type="concat"):
        extractor = FeatureExtractor(self.sentences, self.senses)
        features = extractor.featurize(encoding=encoding, 
                                    sentence_type=sentence_type)
        self.features = features
        return features

    def __len__(self):
        NotImplemented

    def __getitem__(self, idx):
        NotImplemented
