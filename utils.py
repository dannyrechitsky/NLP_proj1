"""
The utility module contains helper functions/classes that are not related
to internal functionalities and attributes of core modules. That means, the
functions/classes within this module can only be imported from other modules.

Instructions:
---
* The only provided function is ``to_level`` which normalizes the string of "sense"
within the PDTB dataset by reducing its sense level from a higher level to a lower
one. Usage is described in its docstring.
* Other than that, you're welcome to add any functionalities in this module
"""
import pickle
import numpy as np
import torch

def to_level(sense: str, level: int = 2) -> str:
    """converts a sense in string to a desired level

    There are 3 sense levels in PDTB:
        Level 1 senses are the single-word senses like `Temporal` and `Contingency`.
        Level 2 senses add an additional sub-level sense on top of Level 1 senses, as in `Expansion.Exception`
        Level 3 senses adds yet another sub-level sense, as in `Temporal.Asynchronous.Precedence`.

    This function is used to ensure that all senses do not exceed the desired
    sense level provided as the argument `level`. For example,
    >>> to_level('Expansion.Restatement', level=1)
    'Expansion'
    >>> to_level('Temporal.Asynchronous.Succession', level=2)
    'Temporal.Asynchronous'

    When the input sense has a lower sense level than the desired sense level,
    this function will retain the original sense string. For example,

    >>> to_level('Expansion', level=2)
    'Expansion'
    >>> to_level('Comparison.Contrast', level=3)
    'Comparison.Contrast'

    Args:
        sense (str): a sense as given in any of the PDTB data files
        level (int): a desired sense level

    Returns:
        str: a sense below or at the desired sense level
    """
    s_split = sense.split(".")
    s_join = ".".join(s_split[:level])
    return s_join

def count_loss_epochs(prev_epoch_loss, avg_val_loss, inc_loss_epochs) -> int:
    """
    Checks if validation loss is increasing this epoch and increments
    number of consecutive epochs of increasing loss; to be used as
    an early stop condition for training

    Returns:
        int: number of consecutive epochs of increasing validation loss 
    """

    # if loss increasing, increment counter of num epochs
    if avg_val_loss - prev_epoch_loss > 0.05:
        inc_loss_epochs += 1
    # else, zero out counter
    else:
        inc_loss_epochs = 0
    
    return inc_loss_epochs
    
def cache_glove_embeddings():
    glove_cache = {}
    with open('glove/dolma_300_2024_1.2M.100_combined.txt', 'r', encoding="utf-8") as file:
        for line in file:
            # 2. find embeddings in GloVe text file pertaining to vocab
            #    populate tensor with GloVe embeddings (v x 300)                
            word_and_embeddings = line.strip().split()
            if len(word_and_embeddings) != 301:
                raise ValueError(f'length of word + embedding = {len(word_and_embeddings)}'
                                 f', but expected 301')
            
            print(f'length of word + embedding {len(word_and_embeddings)}')
            word = word_and_embeddings[0]

            # safely convert embedding strings to np floats to torch tensor
            embedding = word_and_embeddings[1:]
            try:
                # safe conversion strings to floats
                word_embeddings_array = np.array(
                    word_and_embeddings[1:], dtype=np.float32)
            except ValueError:
                # ignore corrupted lines, treat as OOV
                continue
            word_embeddings_tensor = torch.from_numpy(word_embeddings_array)
            
            # add word embeddings tensor to dictionary cache
            glove_cache[word] = word_embeddings_tensor
    
    # pickle glove_cache in a .pt file
    torch.save(glove_cache, "glove/glove_cache.pt")

    return glove_cache



def load_glove_cache():

    return torch.load("glove/glove_cache.pt")

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

# PICKLE ME TIMBERS!!!
def dataset_pickler(data):
    with open("pickle_jar/dataset", 'wb') as f:
        pickle.dump(data, f)

def feature_pickler_glove(data, sentence_type="concat"):
    if sentence_type=="concat":
        features = data.features
        torch.save(features, "pickle_jar/features_concat_glove")  
    else:
        features = data.features
        torch.save(features, "pickle_jar/features_glove")

def feature_pickler_random(data, sentence_type="concat"):
    if sentence_type=="concat":
        features = data.features
        torch.save(features, "pickle_jar/features_concat_random")  
    else:
        features = data.features
        torch.save(features, "pickle_jar/features_random")

def feature_pickler_val(cli_args, val_set, mean, std, 
                        sentence_type="concat", encoding="glove"):
    if encoding=="glove":
        if sentence_type=="concat":
            val_features = val_set.featurize(cli_args)
            val_features, _, _ = standardize(val_features, mean, std)
            torch.save(val_features, "pickle_jar/val_features_concat_glove")  
        else:
            val_features = val_set.featurize(cli_args)
            val_features, _, _ = standardize(val_features, mean, std)
            torch.save(val_features, "pickle_jar/val_features_glove")
    else: # encoding=="random"
        if sentence_type=="concat":
            val_features = val_set.featurize(cli_args)
            val_features, _, _= standardize(val_features, mean, std)
            torch.save(val_features, "pickle_jar/val_features_concat_random")  
        else:
            val_features = val_set.featurize(cli_args)
            val_features, _, _ = standardize(val_features, mean, std)
            torch.save(val_features, "pickle_jar/val_features_random")    

def feature_pickler_test(cli_args, test_set, mean, std, 
                        sentence_type="concat", encoding="glove"):
    if encoding=="glove":
        if sentence_type=="concat":
            test_features = test_set.featurize(cli_args)
            test_features, _, _ = standardize(test_features, mean, std)
            torch.save(test_features, "pickle_jar/test_features_concat_glove") 
        else:
            test_features = test_set.featurize(cli_args)
            test_features, _, _ = standardize(test_features, mean, std)
            torch.save(test_features, "pickle_jar/test_features_glove")
    else: # encoding == "random"
        if sentence_type=="concat":
            test_features = test_set.featurize(cli_args)
            test_features, _, _ = standardize(test_features, mean, std)
            torch.save(test_features, "pickle_jar/test_features_concat_rando") 
        else:
            test_features = test_set.featurize(cli_args)
            test_features, _, _ = standardize(test_features, mean, std)
            torch.save(test_features, "pickle_jar/test_features_random")    

# UNPICKLE ME NOW!!!
def unpickle_dataset():
    with open("pickle_jar/dataset", 'rb') as f:
        return pickle.load(f)

def unpickle_features(encoding="glove", sentence_type="flat"):
    if sentence_type == "flat":
        if encoding=="glove":
            return torch.load("pickle_jar/features_glove")
        else: # encoding=="random"
            return torch.load("pickle_jar/features_random")
    elif sentence_type == "concat":
        if encoding=="glove":
            return torch.load("pickle_jar/features_concat_glove")
        else: # encoding=="random"
            return torch.load("pickle_jar/features_concat_random")
    else:
        raise ValueError(f'sentence_type={sentence_type} is not valid \n'
                         f'It must be "flat" or "concat"')     

def unpickle_features_val(encoding="glove", sentence_type="flat"):
    if sentence_type == "flat":
        if encoding == "glove":
            return torch.load("pickle_jar/val_features_glove")
        else: # encoding == "random"
            return torch.load("pickle_jar/val_features_random")
    elif sentence_type == "concat":
        if encoding=="glove":
            return torch.load("pickle_jar/val_features_concat_glove")
        else: # encoding=="random"
            return torch.load("pickle_jar/val_features_concat_random")
    else:
        raise ValueError(f'sentence_type={sentence_type} is not valid \n'
                         f'It must be "flat" or "concat"')

def unpickle_features_test(encoding="glove", sentence_type="flat"):
    if sentence_type == "flat":
        if encoding == "glove":
            return torch.load("pickle_jar/test_features_glove")
        else: # encoding == "random"
            return torch.load("pickle_jar/test_features_random")
    elif sentence_type == "concat":
        if encoding=="glove":
            return torch.load("pickle_jar/test_features_concat_glove")
        else: # encoding=="random"
            return torch.load("pickle_jar/test_features_concat_random")
    else:
        raise ValueError(f'sentence_type={sentence_type} is not valid \n'
                         f'It must be "flat" or "concat"')
        
     


# if __name__ == '__main__':
    # dataset_pickler()
    # feature_pickler_random(sentence_type="flat")
    # feature_pickler_glove(sentence_type="flat")

    # model1 = MLP([64], 21, from_pickle=False, 
    #             sentence_type="concat", encoding="glove")

    # feature_pickler_val(PDTBDataset('validate'), model1.mean, model1.std,
    #                     encoding="glove", sentence_type="concat")
    # feature_pickler_test(PDTBDataset('test'), model1.mean, model1.std,
    #                     encoding="glove", sentence_type="concat")