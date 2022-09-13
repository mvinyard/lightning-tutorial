# lightning-tutorial

### Table of contents

* [PyTorch Datasets and DataLoaders](#pytorch-datasets-and-dataloaders)
    * Key module: `torch.utils.data.Dataset`
    * Key module: `torch.utils.data.DataLoader`
    * Other essential functions
    * Let’s bring this to single-cell: `torch-adata`

* Lightning basics
* `LightningDataModule`


## PyTorch Datasets and DataLoaders

### Key module: `torch.utils.data.Dataset`

The `Dataset` module is an overwritable python module. You can modify it at will as long as you maintain the following three class methods:
1. `__init__`
2. `__len__`
3. `__getitem__`

These are name-specific handles used by `torch` under the hood when passing data through a model.


```python
from torch.utils.data import Dataset

class TurtleData(Dataset):
    def __init__(self):
        """
        here we should pass requisite arguments
        that enable __len__() and __getitem__()
        """
        
    def __len__(self):
        """
        Returns the length/size/# of samples in the dataset.
        e.g., a 20,000 cell dataset would return `20_000`.
        """
        return # len
    
    def __getitem__(self, idx):
        """
        Subset and return a batch of the data.
        
        `idx` is the batch index (# of idx values = batch size). 
        Maximum `idx` passed is <= `self.__len__()`
        """
        return # sampled data
```

* [Fantastic PyTorch `Dataset` tutorial from Stanford](https://stanford.edu/~shervine/blog/pytorch-how-to-generate-data-parallel)

* **Try it for yourself!** [**Colab `Dataset` tutorial notebook**](https://colab.research.google.com/github/mvinyard/lightning-tutorial/blob/main/notebooks/tutorial_nb.01.pytorch_datasets.ipynb)

[back to table of contents](#table-of-contents)
