"""
The model module contains only neural models which are used to be trained.

Instructions:
---
The only implementation for this module is implementing multinomial logistic regression
using the subclass of ``torch.nn.Module``.
"""

import torch.nn as nn
from dataset import *
from utils import *

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

def standardize(features):
    # Standardize the features before model training
    if features.numel() > 0:
        
        # 1. Calculate Mean and Std Dev across the sentences
        mean = features.mean(dim=0, keepdim=True)
        std = features.std(dim=0, keepdim=True)
        
        # 2. Apply the Standard Scaling formula: (X - mean) / std
        # 1e-6 prevents division by zero
        standardized_features = (features - mean) / (std + 1e-6) 
        
        print("Features standardized: Mean ~0.0, Std Dev ~1.0")    

        return standardized_features

    return features


class LogisticRegression(nn.Module):
    """Logistic regression model"""
    def __init__(self, encoding="glove", from_pickle=False, lr=0.01):
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
            self.data = PDTBDataset()
            self.features = self.data.featurize(encoding=encoding)
        
        # Check for alignment between features and senses
        if self.features.shape[0] != len(self.data.senses):
            print(f"Warning: Number of features ({self.features.shape[0]})" 
                  f"does not match number of senses ({len(self.data.senses)})." 
                  f"Possible data misalignment.")
        
        # standardize features for mean=0 and std=1
        self.features = standardize(self.features)


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
    


class CNN(nn.Module):
    """CNN model"""
    NotImplemented