# ⚡ lightning-tutorial

[![PyPI pyversions](https://img.shields.io/pypi/pyversions/lightning-tutorial.svg)](https://pypi.python.org/pypi/lightning-tutorial/)
[![PyPI version](https://badge.fury.io/py/lightning-tutorial.svg)](https://badge.fury.io/py/lightning-tutorial)
[![Code style: black](https://img.shields.io/badge/code%20style-black-000000.svg)](https://github.com/psf/black)

### Installation of the partner package

```BASH
pip install lightning-tutorial
```

### Table of contents

* [PyTorch Datasets and DataLoaders](#pytorch-datasets-and-dataloaders)
    * Key module: `torch.utils.data.Dataset`
    * Key module: `torch.utils.data.DataLoader`
    * Other essential functions
    
* [Single-cell data structures meet pytorch: `torch-adata`](#single-cell-data-structures-meet-pytorch-torch-adata)
* [Lightning basics and the `LightningModule`](#lightning-basics-and-the-lightningmodule)
* [`LightningDataModule`](#lightningdatamodule)


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

<a href="https://github.com/mvinyard/torch-adata/" ><img alt="torch-adata-concept-overview" src="https://github.com/mvinyard/torch-adata/blob/main/docs/imgs/torch-adata.concept_overview.svg" width="600" /></a>

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

[☝️ back to table of contents](#table-of-contents)


## Lightning basics and the [`LightningModule`](https://pytorch-lightning.readthedocs.io/en/stable/common/lightning_module.html)


```python
from pytorch_lightning imoport LightningModule

class YourSOTAModel(LightningModule):
    def __init__(self,
                 net,
                 optimizer_kwargs={"lr":1e-3},
                 scheduler_kwargs={},
                ):
        super().__init__()
        
        self.net = net
        self.optimizer_kwargs = optimizer_kwargs
        self.scheduler_kwargs = scheduler_kwargs
        
        
    def forward(self, batch):
        
        x, y = batch
        
        y_hat = self.net(x)
        loss  = LossFunc(y_hat, y)
        
        return y_hat, loss
        
    def training_step(self, batch, batch_idx):
        
        y_hat, loss = self.forward(batch)
        
        return loss.sum()
    
    def validation_step(self, batch, batch_idx):
        
        y_hat, loss = self.forward(batch)
        
        return loss.sum()
    
    def test_step(self, batch, batch_idx):
        
        y_hat, loss = self.forward(batch)
        
        return loss.sum()
    
    def configure_optimizers(self):
        optimizer = torch.optim.Adam(self.parameters(), **self._optim_kwargs)
        scheduler = torch.optim.lr_scheduler.StepLR(optimizer(), **self._scheduler_kwargs)
        
        return [optimizer, ...], [scheduler, ...]
```

* **Try it for yourself!** [**Lightning Classifier tutorial notebook**](https://colab.research.google.com/github/mvinyard/lightning-tutorial/blob/main/notebooks/tutorial_nb.02.LightningClassifier.ipynb)


#### Additional useful documentation and standalone tutorials

* [Lightning in 15 minutes](https://pytorch-lightning.readthedocs.io/en/stable/starter/introduction.html)
* [Logging metrics at each epoch](https://pytorch-lightning.readthedocs.io/en/stable/common/lightning_module.html#train-epoch-level-metrics)

[☝️ back to table of contents](#table-of-contents)


## [`LightningDataModule`](https://pytorch-lightning.readthedocs.io/en/stable/notebooks/lightning_examples/datamodules.html)

**Purpose**: Make your model independent of a given dataset, while at the same time making your dataset reproducible and perhaps just as important: **easily shareable**.

```python
from pytorch_lightning import LightningDataModule
from torch.data.utils import DataLoader

class YourDataModule(LightningDataModule):
    
    def __init__(self):
        # define any setup computations
        
    def prepare_data(self):        
        # download data if applicable
        
    def setup(self, stage):
        # assign data to `Dataset`(s)
        
    def train_dataloader(self):
        return DataLoader(self.train_dataset, batch_size=self.batch_size, num_workers=self.num_workers)
        
    def val_dataloader(self):
        return DataLoader(self.val_dataset, batch_size=self.batch_size, num_workers=self.num_workers)
        
    def test_dataloader(self):
        return DataLoader(self.test_dataset, batch_size=self.batch_size, num_workers=self.num_workers)
        
```

* **Try it for yourself!** [**LightningDataModule tutorial notebook**](https://colab.research.google.com/github/mvinyard/lightning-tutorial/blob/main/notebooks/tutorial_nb.03.LightningDataModule.ipynb)

When it comes to actually using one of these, it looks something like the following:

```python
# Init the LightningDataModule as well as the LightningModel
data = YourDataModule()
model = YourLightningModel()

# Define trainer
trainer = Trainer(accelerator="auto", devices=1)

# Ultimately, both  model and data are passed as an arg to trainer.fit
trainer.fit(model, data)
```

* **Try it for yourself!** [**LightningGAN tutorial notebook**](https://colab.research.google.com/github/mvinyard/lightning-tutorial/blob/main/notebooks/tutorial_nb.04.LightningGAN.ipynb)

* [Official `LightningDataModule` documentation](https://pytorch-lightning.readthedocs.io/en/stable/notebooks/lightning_examples/datamodules.html)


Here's an example of a `LightningDataModule` implemented in practice, using the LARRY single-cell dataset: [**link**](https://github.com/mvinyard/LARRY-dataset). Initial downloading and formatting occurs only once but takes several minutes so we will leave it outside the scope of this tutorial.

[☝️ back to table of contents](#table-of-contents)

