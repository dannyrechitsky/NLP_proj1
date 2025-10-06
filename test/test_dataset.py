import unittest
from dataset import * 
import pickle

class TestDataset(unittest.TestCase):
    """
    A TestSuite class for grouping related test methods.
    It inherits from unittest.TestCase.
    """
    data = PDTBDataset()


    @classmethod
    def test_clean_data(cls):
        sentences = cls.data.sentences
        senses = cls.data.senses
        for i in range(len(sentences)):
            print(f'sentence: {sentences[i]}\nsense: {senses[i]}\n')

    @classmethod
    def test_data_types(cls):
        tester = TestDataset()
        tester.assertIsInstance(cls.data.sentences, list)
        tester.assertIsInstance(cls.data.sentences[0], str)
        tester.assertIsInstance(cls.data.senses, list)
        tester.assertIsInstance(cls.data.senses[0], str)

    @classmethod
    def test_feature_embedding_shape_glove(cls):
        tester = TestDataset()
        with open("pickle_jar/dataset", 'rb') as f:
            cls.data = pickle.load(f)
        features = torch.load("pickle_jar/features_glove")
        # features embedding contains all sentence embeddings
        tester.assertEqual(len(cls.data.sentences), features.shape[0])
        # features embedding has 300 columns
        tester.assertEqual(300, features.shape[1])

    
    @classmethod
    def test_feature_embedding_shape_random(cls):
        tester = TestDataset()
        with open("pickle_jar/dataset", 'rb') as f:
            cls.data = pickle.load(f)
        features = torch.load("pickle_jar/features_random")
        # features embedding contains all sentence embeddings
        tester.assertEqual(len(cls.data.sentences), features.shape[0])
        # features embedding has 300 columns
        tester.assertEqual(300, features.shape[1])

    @classmethod
    def test_sentence_index_spotcheck(cls):
        tester = TestDataset()
        with open("pickle_jar/dataset", 'rb') as f:
            cls.data = pickle.load(f)
        features_embedding = torch.load("pickle_jar/features_embedding_glove")

        featurizer = FeatureExtractor(cls.data.sentences, cls.data.senses)
        sentences, vocab_map = featurizer.map_vocab()

        # get indices for sentences 11
        word_indices = []
        for word in sentences[10]:
            word_indices.append(vocab_map[word])
        
        NotImplemented












if __name__ == '__main__':
    unittest.main()