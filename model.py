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
    def __init__(self, encoding="glove", sentence_type="concat", from_pickle=False, lr=0.01):
        super().__init__()
        
        # to print learning rate for debugging
        self.lr = lr

        self.data : PDTBDataset
        self.features : torch.Tensor


        self.encoding = encoding

        # unpickle or featurize dataset/embedding
        if from_pickle:
            self.data = unpickle_dataset()
            self.features = unpickle_features(encoding=encoding)
        else: # build new dataset and features
            self.data = PDTBDataset(encoding=encoding, sentence_type=sentence_type)
            self.features = self.data.featurize(encoding=encoding)
        
        # Check for alignment between features and senses
        if self.features.shape[0] != len(self.data.senses):
            print(f"Warning: Number of features ({self.features.shape[0]})" 
                  f"does not match number of senses ({len(self.data.senses)})." 
                  f"Possible data misalignment.")
        
        # standardize features for mean=0 and std=1
        self.features, _, _ = standardize(self.features)


        # initialize weights matrix
        # count num classes
        unique_senses = set(self.data.senses)
        OUTPUT_SIZE = len(unique_senses)
        INPUT_SIZE = 300
        self.weights = nn.Linear(
            in_features=INPUT_SIZE, 
            out_features=OUTPUT_SIZE
            )
        
        # create sense:index map
        self.sorted_senses = sorted(unique_senses)
        self.sense_map = {sense:i for i, sense in enumerate(self.sorted_senses)}
        self.senses_tensor = torch.tensor(
                [self.sense_map[sense] for sense in self.data.senses],
                dtype=torch.long
                )

        # initialize cross-entropy loss criterion and optimizer
        class_weights_tensor = weigh_classes(self.data.senses, self.sorted_senses)
        self.criterion = nn.CrossEntropyLoss(weight=class_weights_tensor)



    ## TRAIN
    
    def train(self, num_epochs=30):
        self.optimizer = torch.optim.SGD(self.weights.parameters(), lr=self.lr)

        for epoch in range(num_epochs):
            
            # randomize order of sentences (input) and senses (true output)
            shuffled_indices = torch.randperm(len(self.features))
            shuffled_sentences = self.features[shuffled_indices]
            shuffled_senses = self.senses_tensor[shuffled_indices]

            
            # separate input into batches
            batch_sentences = None
            batch_senses = None
            i = 0
            while(i < len(shuffled_sentences)):
                # zero out old gradients
                self.optimizer.zero_grad()
                
                # if batch_sentences  = 64 sentences
                if i+63 < len(shuffled_sentences):
                    batch_sentences = shuffled_sentences[i:i+64]
                    batch_senses = shuffled_senses[i:i+64]
                    i += 64
                # if fewer than 64 sentences remaining
                else:
                    batch_sentences = shuffled_sentences[i:len(shuffled_sentences)]
                    batch_senses = shuffled_senses[i:len(shuffled_senses)]
                    i = len(shuffled_sentences)

                # raw score
                logits = self.weights(batch_sentences)

                
                # retrieve list output indices corresponding to input indices  
                true_labels = batch_senses
                loss = self.criterion(logits, true_labels)

                # update weights
                loss.backward()
                print(f'learning rate: {self.lr}')
                print(f"epoch {epoch}: Gradient = {self.weights.weight.grad.norm()}")
                self.optimizer.step()

                # print progress

                print(f'epoch {epoch}: average batch loss = {loss.item()}')

            torch.save(self.weights.state_dict(), 
                    "pickle_jar/weights_"+self.encoding)
                

            ## ------------------------------

            #     # compute y_hat = sigmoid(x·theta) for each row
            # y_hat = scipy.special.expit(np.dot(x,self.theta))

            # # update cross-entropy loss
            # l_ce -= (np.dot(y,np.log(y_hat))) + (np.dot((1-y),np.log(1-y_hat))) # dot product instead of multiply

            # ### calculate loss > adjust weights with GD
            # # compute average loss gradient
            # gradient_loss = (1/len(x))*(np.dot(x.T, y_hat-y))

            # # update weights and bias
            # self.theta = self.theta - eta*gradient_loss
            
    
class MLP(nn.Module):
    """Multilayer perceptron"""
    def __init__(self, hidden_sizes, output_dim, 
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
            self.data = PDTBDataset(set="train", 
                                    sentence_type=sentence_type, 
                                    encoding=encoding
                                    )
            # featurize data
            self.data.featurize(
                encoding=encoding,
                sentence_type=sentence_type
                )
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
            utils.feature_pickler_val(PDTBDataset('validate'), self.mean, self.std,
                            encoding=encoding, sentence_type=sentence_type)
            print("Featurizing test set...")
            utils.feature_pickler_test(PDTBDataset('test'), self.mean, self.std,
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
                batch_size, 
                embed_dim, 
                max_sent_len,
                encoding="glove",
                sentence_type="concat",
                from_pickle=False
                ):
        super().__init__()
        
        NUM_FILTERS = 100
        NUM_CLASSES = 21

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
            self.data = PDTBDataset(set="train", 
                                    sentence_type=sentence_type, 
                                    encoding=encoding
                                    )
            # featurize data
            self.data.featurize(
                encoding=encoding,
                sentence_type=sentence_type
                )
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
            utils.feature_pickler_val(PDTBDataset('validate'), self.mean, self.std,
                            encoding=encoding, sentence_type=sentence_type)
            print("Featurizing test set...")
            utils.feature_pickler_test(PDTBDataset('test'), self.mean, self.std,
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
                            out_features=NUM_CLASSES
        )                         

        # Initialize activation function
        self.relu = nn.ReLU()

        # Perform forward pass
        def forward(self, x):
            # permute to swap embedding dimension and max sentence length
            x = x.permute(0, 2, 1)

            # convolve and activate
            x = self.relu(self.conv1(x))

            # apply global max pooling: reduce length dim to 1
            # to get a fixed-size feature verctor for each instance
            x = self.pool(x).squeeze(-1)

            # perform classification
            logits = self.fc(x)
            return logits
        
        