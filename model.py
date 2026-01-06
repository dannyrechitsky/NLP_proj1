"""
The model module contains only neural models which are used to be trained.

Instructions:
---
The only implementation for this module is implementing multinomial logistic regression
using the subclass of ``torch.nn.Module``.
"""

import torch.nn as nn
from dataset import PDTBDataset
from dataset import *
import utils
from utils import *
from torchmetrics.classification import Accuracy, F1Score, Precision, Recall

def weigh_classes(output:list[str], unique_classes:list) -> torch.Tensor:
    """
    calculate and return tensor of class weights 
    inversely proporitional to class frequency
    to mitigate class imbalance in data
    """
    MAX_WEIGHT_CAP = 100.0
    counts = []
    for sense in unique_classes:
        counts.append(output.count(sense))
    total = sum(counts)
    weights = []
    for count in counts:
        calculated_weight = total / count
        final_weight = min(calculated_weight, MAX_WEIGHT_CAP)
        weights.append(final_weight)

    return torch.tensor(weights, dtype=torch.float32)

def standardize(features, mean=None, std=None,):
    # Standardize the features before model training
    if features.numel() > 0:
        if mean is None or std is None:
            # 1. For training set, calculate Mean and Std Dev across the sentences
            mean = features.mean(dim=0, keepdim=True)
            std = features.std(dim=0, keepdim=True)
        
        # 2. Apply the Standard Scaling formula: (X - mean) / std
        # 1e-6 prevents division by zero
        standardized_features = (features - mean) / (std + 1e-6) 
           

        # return mean and std for standardizing val set and test set
        return standardized_features, mean, std

    return features, None, None


class LogisticRegression(nn.Module):
    """Logistic regression model"""
    def __init__(self, 
                cli_args,
                output_dim, 
                sentence_type="concat", ):
        super().__init__()
        
        
        # to print learning rate for debugging
        self.lr = cli_args.lr

        self.data : PDTBDataset
        self.features : torch.Tensor

        # set encoding to CLI arg
        self.encoding = cli_args.embedding

        # unpickle or featurize dataset/embedding
        from_pickle = cli_args.from_pickle
        
        if from_pickle:
            self.data = unpickle_dataset()
            self.features = unpickle_features(encoding=self.encoding)
        else: # build new dataset and features
            self.data = PDTBDataset(cli_args,
                                    set='train',
                                    sentence_type=sentence_type)


            # featurize data
            self.data.featurize(cli_args, 
                                sentence_type=sentence_type)

            # save training vocab to model
            self.data_vocab = self.data.vocab_map

            # save model features
            self.features = self.data.features

            # pickle features
            if self.encoding == "glove":
                utils.feature_pickler_glove(self.data, sentence_type=sentence_type)
            elif self.encoding == "random":
                utils.feature_pickler_random(self.data, sentence_type=sentence_type)
        
        # Check for alignment between features and senses
        if self.features.shape[0] != len(self.data.senses):
            print(f"Warning: Number of features ({self.features.shape[0]})" 
                  f"does not match number of senses ({len(self.data.senses)})." 
                  f"Possible data misalignment.")
        
        # standardize features for mean=0 and std=1
        self.features, _, _ = standardize(self.features)


        # initialize weights matrix
        input_size = self.features.shape[1]
        self.weights = nn.Linear(
            in_features=input_size, 
            out_features=output_dim
            )
        
        # create sense:index map
        unique_senses = set(self.data.senses)

        self.sorted_senses = sorted(unique_senses)
        self.sense_map = {sense:i for i, sense in enumerate(self.sorted_senses)}
        self.senses_tensor = torch.tensor(
                [self.sense_map[sense] for sense in self.data.senses],
                dtype=torch.long
                )


    def forward(self, x):
        """Takes a batch of sentences and returns the raw scores (logits)"""
        logits = self.weights(x)
        return logits

    
class MLP(nn.Module):
    """Multilayer perceptron"""
    def __init__(self, cli_args, hidden_sizes, output_dim, 
                 encoding="glove", sentence_type="concat", from_pickle=False,
                 dropout=0.0):
        super().__init__()
        
        # to print learning rate for debugging
        self.data : PDTBDataset
        self.features : torch.Tensor

        self.encoding = encoding
        self.sentence_type = sentence_type



        # unpickle or featurize dataset/embedding
        if from_pickle:
            self.data = utils.unpickle_dataset()
            self.features = utils.unpickle_features(
                encoding=encoding,
                sentence_type=sentence_type)
        else: # build new dataset and features
            self.data = PDTBDataset(cli_args, 
                                    set="train", 
                                    sentence_type=sentence_type)
            

            # featurize data
            self.data.featurize(cli_args, 
                                sentence_type=sentence_type)
            
            # save training vocab to model
            self.data_vocab = self.data.vocab_map

            # point model features to data features (save space)
            self.features = self.data.features

            #TODO: remove the old code below: I featurized twice! 
            #     encoding=encoding,
            #     sentence_type=sentence_type)
            # pickle features
            if self.encoding == "glove":
                utils.feature_pickler_glove(self.data, sentence_type=self.sentence_type)
            elif self.encoding == "random":
                utils.feature_pickler_random(self.data, sentence_type=sentence_type)
            
        
        # Check for alignment between features and senses
        if self.features.shape[0] != len(self.data.senses):
            print(f"Warning: Number of features ({self.features.shape[0]})" 
                  f"does not match number of senses ({len(self.data.senses)})." 
                  f"Possible data misalignment.")
        
        # standardize features for mean=0 and std=1
        self.features, self.mean, self.std = standardize(self.features)
        
        if not from_pickle:
            print("Featurizing validation set...")
            utils.feature_pickler_val(cli_args, PDTBDataset(cli_args, set='validate'), self.mean, self.std,
                            encoding=encoding, sentence_type=sentence_type)
            print("Featurizing test set...")
            utils.feature_pickler_test(cli_args, PDTBDataset(cli_args, set='test'), self.mean, self.std,
                            encoding=encoding, sentence_type=sentence_type)
        
        print(f'\n\n\nMLP sentence type:    {self.sentence_type}')
        print(f'input features shape: {self.features.shape}\n\n\n')

        # create sense:index map
        unique_senses = set(self.data.senses)
        self.sorted_senses = sorted(unique_senses)
        self.sense_map = {sense:i for i, sense in enumerate(self.sorted_senses)}
        self.senses_tensor = torch.tensor(
                [self.sense_map[sense] for sense in self.data.senses],
                dtype=torch.long
                )
        

        # TODO: move to FeatureExtractor class
        
        # define layer dimensions
        input_dim = self.features.shape[1]
        self.layer_dims = [input_dim] + hidden_sizes
        
        # store hidden layers in nn.ModuleList()
        self.hidden_layers = nn.ModuleList()
        # layers = [256, 128, 64, ]
        for i in range(len(self.layer_dims) - 1):
            self.hidden_layers.append(
                nn.Linear(self.layer_dims[i], self.layer_dims[i+1])
            )
        
        # output layer
        self.output_layer = nn.Linear(hidden_sizes[-1], output_dim)

        # activation function
        self.relu = nn.ReLU()

        # dropout regularization
        self.dropout = nn.Dropout(p=dropout)


    def forward(self, x):
        # loop through layers
        for layer in self.hidden_layers:
            x = self.relu(layer(x))
            
            # apply dropout
            x = self.dropout(x)

        logits = self.output_layer(x)
        
        return logits


class CNN(nn.Module):
    """CNN model"""
    def __init__(self,
                cli_args, 
                batch_size, 
                embed_dim,
                output_dim, 
                max_sent_len,
                encoding="glove",
                sentence_type="concat",
                from_pickle=False
                ):
        super().__init__()
        
        NUM_FILTERS = 100

        # to print learning rate for debugging
        self.data : PDTBDataset
        self.features : torch.Tensor

        self.encoding = encoding
        self.sentence_type = sentence_type

        # unpickle or featurize dataset/embedding
        if from_pickle:
            self.data = utils.unpickle_dataset()
            self.features = utils.unpickle_features(
                encoding=encoding,
                sentence_type=sentence_type)
        else: # build new dataset and features
            self.data = PDTBDataset(cli_args, 
                                    set="train", 
                                    sentence_type=sentence_type)

            # featurize data
            self.data.featurize(cli_args, 
                                sentence_type=sentence_type)
            
            # save training vocab to model
            self.data_vocab = self.data.vocab_map

            # point model features to data features (save space)
            self.features = self.data.features

            #TODO: remove the old code below: I featurized twice! 
            #     encoding=encoding,
            #     sentence_type=sentence_type)
            # pickle features
            if self.encoding == "glove":
                utils.feature_pickler_glove(self.data, sentence_type=self.sentence_type)
            elif self.encoding == "random":
                utils.feature_pickler_random(self.data, sentence_type=sentence_type)
            
        
        # Check for alignment between features and senses
        if self.features.shape[0] != len(self.data.senses):
            print(f"Warning: Number of features ({self.features.shape[0]})" 
                  f"does not match number of senses ({len(self.data.senses)})." 
                  f"Possible data misalignment.")
        
        # standardize features for mean=0 and std=1
        self.features, self.mean, self.std = standardize(self.features)
        

        if not from_pickle:
            print("Featurizing validation set...")
            utils.feature_pickler_val(cli_args, PDTBDataset(cli_args, set='validate'), self.mean, self.std,
                            encoding=encoding, sentence_type=sentence_type)
            print("Featurizing test set...")
            utils.feature_pickler_test(cli_args, PDTBDataset(cli_args, set='test'), self.mean, self.std,
                            encoding=encoding, sentence_type=sentence_type)
        
        print(f'\n\n\nMLP sentence type:    {self.sentence_type}')
        print(f'input features shape: {self.features.shape}\n\n\n')

        # TODO: move to FeatureExtractor class
        # create sense:index map
        unique_senses = set(self.data.senses)
        self.sorted_senses = sorted(unique_senses)
        self.sense_map = {sense:i for i, sense in enumerate(self.sorted_senses)}
        self.senses_tensor = torch.tensor(
                [self.sense_map[sense] for sense in self.data.senses],
                dtype=torch.long
                )
        
        # Initialize convolution layer
        self.conv = nn.Conv1d(in_channels=embed_dim,
                            out_channels=NUM_FILTERS,
                            kernel_size=3
        )
        
        # Initialize global pooling 
        L_out = max_sent_len - 3 + 1 #L_out = 48 at max_sent_len 50 
        self.pool = nn.MaxPool1d(kernel_size=L_out)

        # Initialize classification layer
        self.fc = nn.Linear(in_features=NUM_FILTERS,
                            out_features=output_dim
        )                         

        # Initialize activation function
        self.relu = nn.ReLU()

    # Perform forward pass
    def forward(self, x):
        # reshape flattened concatenated vectors to 3-dim
        x = x.view(x.size(0), 50, 300)
        
        # permute to swap embedding dimension and max sentence length
        x = x.permute(0, 2, 1)

        # convolve and activate
        x = self.relu(self.conv(x))

        # apply global max pooling: reduce length dim to 1
        # to get a fixed-size feature verctor for each instance
        x = self.pool(x).squeeze(-1)

        # perform classification
        logits = self.fc(x)
        return logits
        
        