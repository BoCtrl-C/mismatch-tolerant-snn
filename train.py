from models import SpikingLightningModel, SpikingMLP, SpikingGEM
from yin_yang_data_set.dataset import YinYangDataset

import argparse
from functools import partial
from multiprocessing import Process
import os

from sklearn import datasets
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler

import lightning
from lightning import Trainer, seed_everything
from lightning.pytorch.callbacks import ModelCheckpoint, LearningRateMonitor
from lightning.pytorch.loggers import WandbLogger

from snntorch import spikegen

import torch
from torch.utils.data import DataLoader, TensorDataset, random_split
from torchvision import transforms
from torchvision.datasets import MNIST

import wandb


PROJECT = 'gen-robust'
BETA = .9 # NOTE: for tau=10
NUM_STEPS = 100
BS = 512 # NOTE: 16 for iris
NUM_EPOCHS = 300
NUM_WORKERS = 0
CKPT_DIR = 'ckpts'

GPUS = [2, 5, 7]
NUM_PROCESSES = len(GPUS)*1


parser = argparse.ArgumentParser()
parser.add_argument('-m', '--model', type=str)
parser.add_argument('-d', '--dataset', type=str)
parser.add_argument('-ha', '--hardware_aware', action='store_true')
args = parser.parse_args()

if args.dataset == 'mnist':
    MNIST_DIR = 'datasets'

    transform=transforms.Compose([
        transforms.ToTensor(),
        transforms.Lambda(
            lambda x: spikegen.rate(x.unsqueeze(0), num_steps=NUM_STEPS, gain=.1)\
                .flatten(start_dim=1)
        )
    ])

    dataset = MNIST(MNIST_DIR, train=True, transform=transform)
    test_dataset = MNIST(MNIST_DIR, train=False, transform=transform)

    val_size = 5000
    train_dataset, val_dataset = random_split(
        dataset,
        [len(dataset) - val_size, val_size],
        generator=torch.Generator().manual_seed(1)
    )

    layer_sizes = [28**2, 28**2, 10]
    num_genes = [8, 64, 256]
elif args.dataset == 'yinyang':    
    def rate_encoding(sample):
        """Converts the input data into Poisson spike trains.
        """
        
        x, y = sample
        x, y = torch.tensor(x).float(), torch.tensor(y)
        x = spikegen.rate(x.unsqueeze(0), num_steps=NUM_STEPS, gain=1.).squeeze()
        
        return x, y

    # initialize the dataset splits
    train_dataset = YinYangDataset(size=5000, seed=42, transform=rate_encoding)
    val_dataset = YinYangDataset(size=1000, seed=41, transform=rate_encoding)
    test_dataset = YinYangDataset(size=1000, seed=40, transform=rate_encoding)

    # TODO: update before training
    layer_sizes = [4, 32, 3]
    num_genes = [4, 8, 16] # NOTE: it seems very beneficial to over parametrize
    # num_genes = [4, 16, 64] # NOTE: it seems very beneficial to over parametrize
elif args.dataset == 'iris':
    # load the Iris dataset
    iris = datasets.load_iris()
    X, Y = iris['data'], iris['target']

    # normalize
    scaler = MinMaxScaler()
    X = scaler.fit_transform(X)

    # split dataset into train, validation and test splits
    X_train_val, X_test, Y_train_val, Y_test = train_test_split(X, Y, test_size=0.3, random_state=1)
    X_train, X_val, Y_train, Y_val = train_test_split(X_train_val, Y_train_val, test_size=0.1, random_state=1)

    # rate encoding
    seed_everything(seed=1)
    X_train = spikegen.rate(torch.tensor(X_train), num_steps=NUM_STEPS).permute(1, 0, -1)
    X_val = spikegen.rate(torch.tensor(X_val), num_steps=NUM_STEPS).permute(1, 0, -1)
    X_test = spikegen.rate(torch.tensor(X_test), num_steps=NUM_STEPS).permute(1, 0, -1)

    # convert labels into tensors
    Y_train, Y_val, Y_test = torch.tensor(Y_train), torch.tensor(Y_val), torch.tensor(Y_test)

    # to PyTorch Datasets
    train_dataset = TensorDataset(X_train.float(), Y_train.long())
    val_dataset = TensorDataset(X_val.float(), Y_val.long())
    test_dataset = TensorDataset(X_test.float(), Y_test.long())

    layer_sizes = [4, 128, 3]
    num_genes = [2, 4, 6]
else:
    raise Exception('Invalid dataset')

# initialize the sweep name
sweep_name = args.model

# build a suffix for making the sweep name descriptive
sweep_suffix = '-' + '-'.join([str(size) for size in layer_sizes[1:-1]])
if args.hardware_aware: sweep_suffix += '-ha'
sweep_name += sweep_suffix

# W&B config
wandb.login()
CONFIG = {
    'name': sweep_name,
    'method': 'grid',
    'metric': {'goal': 'minimize', 'name': 'val_loss'},
    'parameters': {
        'model': {'values': [args.model]},
        'dataset': {'values': [args.dataset]},
        'layer_sizes': {'values': [layer_sizes]},
        'num_genes': {'values': num_genes if args.model == 'gem' else [0]},
        'learning_rate': {'values': [3e-2, 3e-3, 3e-4]},
        'seed': {'values': [1]},
        'hardware_aware': {'values': [args.hardware_aware]},
    }
}
sweep_id = wandb.sweep(CONFIG, project=PROJECT)

def train(device_id):
    """Wrapper for training.
    """
    
    # set hyperparameters
    run = wandb.init()
    config = wandb.config

    layer_sizes = config['layer_sizes']
    lr = config['learning_rate']
    seed = config['seed']
    if args.model == 'gem': num_genes = config['num_genes']

    # initialize data loaders
    seed_everything(seed=1, workers=True)
    train_dataloader = DataLoader(train_dataset, batch_size=BS, shuffle=True, num_workers=NUM_WORKERS)
    val_dataloader = DataLoader(val_dataset, batch_size=BS, num_workers=NUM_WORKERS)
    test_dataloader = DataLoader(test_dataset, batch_size=BS, num_workers=NUM_WORKERS)

    # initialize the model
    seed_everything(seed=seed, workers=True)
    if args.model == 'mlp':
        model = SpikingMLP(layer_sizes=layer_sizes, beta=BETA, num_steps=NUM_STEPS, bias=False)
    elif args.model == 'gem':
        model = SpikingGEM(layer_sizes=layer_sizes, num_genes=num_genes, beta=BETA, num_steps=NUM_STEPS, bias=False)
    else:
        raise Exception('Invalid model')
    
    model = SpikingLightningModel(
        model,
        num_classes=layer_sizes[-1],
        max_epochs=NUM_EPOCHS,
        lr=lr # NOTE: very important for GEM
    )
    
    # define callbacks
    ckpt_best = ModelCheckpoint(
        dirpath=os.path.join(CKPT_DIR, sweep_name),
        filename=run.name,
        monitor='val_loss',
        mode='min'
    )

    lr_monitor = LearningRateMonitor(logging_interval='epoch')
    
    class SetValidationSeed(lightning.Callback):
        """Callback that sets the random seed for validation in order to keep the rate encoding of validation samples the same across validation epochs.
        """

        def on_validation_start(self, trainer, pl_module):
            seed_everything(seed=1, workers=True)

    class ChangeWeightNoiseSeed(lightning.Callback):
        """Callback that changes the random seed for weight noise generation at each training step. It is used for hardware-aware training. The callback also disables noise generation during validation epochs.
        """

        def __init__(self, weight_cv):
            """Sets the coefficient of variation for weight noise.
            """
            
            super().__init__()

            self.weight_cv = weight_cv

        def on_train_epoch_start(self, trainer, pl_module):
            # enable weight noise
            pl_module.model.weight_cv = self.weight_cv

        def on_train_batch_start(self, trainer, pl_module, batch, batch_idx):
            # increment the random seed
            pl_module.model.weight_noise_seed += 1
        
        def on_validation_start(self, trainer, pl_module):
            # disable weight noise
            pl_module.model.weight_cv = 0.

        def on_train_end(self, trainer, pl_module):
            # disable weight noise
            pl_module.model.weight_cv = 0.

    callbacks = [ckpt_best, lr_monitor, SetValidationSeed()]
    if args.hardware_aware: callbacks.append(ChangeWeightNoiseSeed(weight_cv=0.1)) # TODO: set CV

    trainer = Trainer(
        max_epochs=NUM_EPOCHS,
        callbacks=callbacks,
        accelerator='gpu',
        devices=[device_id],
        logger=WandbLogger(),
        enable_progress_bar=False
    )
    
    # train
    trainer.fit(model, train_dataloader, val_dataloader)
    
    # test
    seed_everything(seed=1, workers=True) # NOTE: rate encoding is stochastic
    trainer.test(dataloaders=test_dataloader, ckpt_path='best')
                     
    run.finish()

# parallelize sweep on multiple processes
def run_agent(device_id):
    wandb.agent(
        sweep_id,
        function=partial(train, device_id=device_id),
        project=PROJECT
    )

# create a list to store the processes
processes = []

# start the parallel processes
process_devices = []
for gpu in GPUS:
    process_devices += NUM_PROCESSES//len(GPUS)*[gpu]
for device_id in process_devices:
    process = Process(target=run_agent, args=(device_id,))
    process.start()
    processes.append(process)

# wait for all processes to finish
for process in processes:
    process.join()