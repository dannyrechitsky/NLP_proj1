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
from dataset import *
data = PDTBDataset()

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



# PICKLE ME TIMBERS!!!
def dataset_pickler():
    with open("pickle_jar/dataset", 'wb') as f:
        pickle.dump(data, f)

def feature_pickler_glove():
        features = data.featurize()
        torch.save(features, "pickle_jar/features_glove")

def feature_pickler_random():
        features = data.featurize(encoding="random")
        torch.save(features, "pickle_jar/features_random")

# UNPICKLE ME NOW!!!
def unpickle_dataset() -> PDTBDataset:
    with open("pickle_jar/dataset", 'rb') as f:
        return pickle.load(f)

def unpickle_features(encoding="glove"):
     if encoding=="glove":
        return torch.load("pickle_jar/features_glove")
     else: # encoding=="random"
        return torch.load("pickle_jar/features_random")
     


if __name__ == '__main__':
    dataset_pickler()
    feature_pickler_glove()
    feature_pickler_random()