"""
The main script for training the model and display the evaluation results.

Instructions:
---
Commonly, the main script contains following functions:
* ``train_loop``
* ``test_loop``
* Evaluation functions
* (Optional) Command line arguments or options
    * If you need explicit control over this script (e.g. learning rate, training size, etc.)
* (Optional) Any functions from ``utils.py`` that helps display results and evaluation

Eventually, this script should be run as
```
uv run main.py <ARGUMENTS> --<OPTIONS>
```

References:
---
https://docs.pytorch.org/tutorials/beginner/basics/optimization_tutorial.html
"""

from model import *

if __name__ == "__main__":
    lr = LogisticRegression(from_pickle="true", lr=0.05)
    lr.train()