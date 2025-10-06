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
import numpy as np

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
        i = 0
        # add index for each token
        sentences = self.tokenize()
        for sentence in sentences:
            for token in sentence:
                if token not in vocab:
                    vocab[token] = i
                    i += 1
        # TODO: padding? UNK token?

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


    def featurize(self, encoding='glove') -> torch.Tensor:
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

        if encoding=="glove":
        # 1. search GloVe text file 
            vocab_embedding_list = torch.zeros(len(vocab_map), 300, dtype=torch.float32)
            word_indices = []
            with open('glove/dolma_300_2024_1.2M.100_combined.txt', 'r') as file:
                for line in file:
                    # 2. find embeddings in GloVe text file pertaining to vocab
                    #    populate tensor with GloVe embeddings (v x 300)                
                    word_and_embeddings = line.strip().split()
                    word = word_and_embeddings[0]
                    word_embeddings = [float(value) for value in word_and_embeddings[1:]]
                    if word in vocab_map:
                        if len(word_embeddings) != 300:
                            print(f"Error: word_embeddings has length {len(word_embeddings)}, expected 300.")
                        i = vocab_map[word]
                        vocab_embedding_list[i] = torch.tensor(word_embeddings, dtype=torch.float32)

            # TODO: delete print debug
            # print(f'vocab_embedding_list shape: {vocab_embedding_list.shape}/n')
            # print(f'vocab_embedding_list num rows: {len(vocab_embedding_list)}')
            # for i in range(len(vocab_embedding_list)):
            #     print(f'row {i} length:{len(vocab_embedding_list[i])}')

            # vocab_embedding_tensor = torch.tensor(vocab_embedding_list, dtype=torch.float32)
            vocab_embedding = torch.nn.Embedding.from_pretrained(vocab_embedding_list)

        else: # encoding="random"
            vocab_embedding = torch.nn.Embedding(len(vocab_map), 300)

        # ---------------- #
                        
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
    def __init__(self):
        super().__init__()
        self.sentences : list[str] = []
        self.senses : list[str] = []
        with open("pdtb/train.json", 'r') as f:
            for line in f:
                line = json.loads(line)
                arg1 = line['Arg1']['RawText'].lower()
                conn = line['Connective']['RawText'].lower()
                arg2 = line['Arg2']['RawText'].lower()
                sense = line['Sense']
                self.sentences.append(''.join([arg1, conn, arg2]))
                self.senses.append(sense[0])
        

    def featurize(self, encoding='glove'):
        extractor = FeatureExtractor(self.sentences, self.senses)
        features = extractor.featurize()
        return features

    def __len__(self):
        NotImplemented

    def __getitem__(self, idx):
        NotImplemented
