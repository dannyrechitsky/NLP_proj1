import unittest
from dataset import * 
from utils import *
from model import *
import torch

class TestModel(unittest.TestCase):
    """
    A TestSuite class for grouping related test methods.
    It inherits from unittest.TestCase.
    """

    def test_LR_inits(self):
        lr = LogisticRegression(from_pickle=True)
        num_classes = len(set(lr.data.senses))
        self.assertEqual(lr.weights.weight.shape, 
                        torch.Size([num_classes, 300]))
        print(f'weights matrix: {lr.weights.weight.shape}')
    
    def test_LR_shuffle_sentences(self):
        lr = LogisticRegression(from_pickle=True)
        shuffled_indices = torch.randperm(len(lr.features))
        shuffled_sentences = lr.features[shuffled_indices]
        print("Original Tensor:\n", lr.features)
        print("\nShuffled Indices:\n", shuffled_indices)
        print("\nShuffled Tensor:\n", shuffled_sentences)
    
    def test_LR_bachify(self):
        lr = LogisticRegression(from_pickle=True)
        shuffled_sentences = lr.features
        batch = None
        i = 0
        num_batches = 0
        while(i < len(shuffled_sentences)):
            # if batch  = 64 sentences
            if i+63 < len(shuffled_sentences):
                batch = shuffled_sentences[i:i+64]
                i += 64
                num_batches += 1
            # if fewer than 64 sentences remaining
            else:
                batch = shuffled_sentences[i:len(shuffled_sentences)]
                i = len(shuffled_sentences)
                num_batches += 1
            # print(f'batch size: {len(batch)}')
            # print(f'sentences left: {len(shuffled_sentences) - i}')
            logits = lr.weights(batch)
            print(f'\nlogits: {logits}')
            print(f'logits size: {logits.shape}')
            
    def test_sense_map(self):
        lr = LogisticRegression(from_pickle=True)
        for sense, index in lr.sense_map.items():
            print(f'{sense}:{index}')
    
    def test_sentence_sense_match(self):
        lr = LogisticRegression(from_pickle=True)
        self.assertEquals(len(lr.data.senses), len(lr.features))
    
    def test_shuffle_match(self):
        lr = LogisticRegression(from_pickle=True)
        shuffled_indices = torch.randperm(len(lr.features))
        shuffled_sentences = lr.features[shuffled_indices]
        senses_tensor = torch.tensor(
            [lr.sense_map[sense] for sense in lr.data.senses]
            )
        shuffled_senses = senses_tensor[shuffled_indices]

        reg_index = 0
        for shuffle_index in shuffled_indices[:20]:
            print(f'shuffled sentence at index {reg_index}:' 
                  f'{shuffled_sentences[reg_index, :5]}')
            print(f'regular  sentence at index {shuffle_index}:' 
                  f'{lr.features[shuffle_index, :5]}')
            print(f'shuffled SENSE at index {reg_index}:' 
                  f'{shuffled_senses[reg_index]}')
            print(f'regular  SENSE at index {shuffle_index}:' 
                  f'{senses_tensor[shuffle_index]}')
            print("\n\n")
            reg_index+=1
    
    def test_requires_grad(self):
        lr = LogisticRegression(from_pickle=True)
        self.assertEquals(lr.weights.weight.requires_grad, True)
    
    def test_class_weights(self):
        lr = LogisticRegression(from_pickle=True)
        class_weights = weigh_classes(lr.data.senses, lr.sorted_senses)
        for i in range(len(class_weights)):
            print(f'sense: {lr.sorted_senses[i]}')
            print(f'frequency: {lr.data.senses.count(lr.sorted_senses[i])}')
            print(f'weight: {class_weights[i]}\n\n')
    
    def test_learning_rates(self):
        rates = [0.001, 0.002, 0.005, 0.01, 0.02]
        for rate in rates:
            print(f'RRRRAAAATTTTTEEEE: {rate}')
            model = LogisticRegression(from_pickle=True, lr=rate)
            model.train()

            


