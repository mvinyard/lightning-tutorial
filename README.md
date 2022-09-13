# ⚡ lightning-tutorial

### Table of contents

* [PyTorch Datasets and DataLoaders](#pytorch-datasets-and-dataloaders)
    * Key module: `torch.utils.data.Dataset`
    * Key module: `torch.utils.data.DataLoader`
    * Other essential functions
    
* Single-cell data structures meet pytorch: `torch-adata`
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


### Key module: `torch.utils.data.DataLoader`

Similar to the usefulness of `AnnData`, the `Dataset` module creates a base unit for distributing and handling data. We can then take advantage of several torch built-ins to enable not only more organized, but faster data processing.

```python
from torch.utils.data import DataLoader

dataset = TurtleData()
data_size = dataset.__len__()
print(data_size)
```
```
20_000
```

### Other essential functions

```python
from torch.utils.data import random_split

train_dataset, val_dataset = random_split(dataset, [18_000, 2_000])

# this can then be fed to a DataLoader, as above
train_loader = DataLoader(train_dataset)
val_loader = DataLoader(val_dataset)
```

### Useful tutorials and documentation

* **Parent module**: [`torch.utils.data`](https://pytorch.org/docs/stable/data.html)
* **[Datasets and DataLoaders tutorial](https://pytorch.org/tutorials/beginner/basics/data_tutorial.html)**


[☝️ back to table of contents](#table-of-contents)


## Single-cell data structures meet pytorch: `torch-adata`
# ![torch-adata-logo](https://github.com/mvinyard/torch-adata/blob/main/docs/imgs/torch-adata.logo.large.svg)

*Create pytorch Datasets from* [`AnnData`](https://anndata.readthedocs.io/en/latest/)

### Installation
- **Note**: This is already done for you, if you've installed this tutorials associated package
```
pip install torch-adata
```

![torch-adata-concept-overview](https://github.com/mvinyard/torch-adata/blob/main/docs/imgs/torch-adata.concept_overview.svg)

### Example use of the base class

The base class, `AnnDataset` is a subclass of the widely-used `torch.utils.data.Dataset`. 

```python
import anndata as a
import torch_adata

adata = a.read_h5ad("/path/to/data.h5ad")
dataset = torch_adata.AnnDataset(adata)
```

Returns sampled data `X_batch` as a `torch.Tensor`.
```python
# create a dummy index
idx = np.random.choice(range(dataset.__len__()), 5)
X_batch = dataset.__getitem__(idx)
```

#### `TimeResolvedAnnDataset`

Specialized class for time-resolved datasets. A subclass of the class, `AnnDataset`.

```python
import anndata as a
import torch_adata as ta

adata = a.read_h5ad("/path/to/data.h5ad")
dataset = torch_adata.TimeResolvedAnnDataset(adata, time_key="Time point")
```
