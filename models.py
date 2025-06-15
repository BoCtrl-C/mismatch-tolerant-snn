import torch
import torch.nn as nn
from torch.optim import Adam

import snntorch as snn
import snntorch.functional as SF

from lightning import LightningModule
from torchmetrics.classification import MulticlassAccuracy


class SpikingLightningModel(LightningModule):
    """Wrapper for training and logging.
    """
    
    def __init__(
        self,
        model,
        num_classes,
        max_epochs,
        lr=3e-3
    ):
        super().__init__()
        self.save_hyperparameters()

        self.model = model
        self.max_epochs = max_epochs
        self.lr = lr

        self.loss_fn = SF.ce_count_loss()
        self.train_acc = MulticlassAccuracy(num_classes=num_classes, average='micro')
        self.val_acc = MulticlassAccuracy(num_classes=num_classes, average='micro')
        self.test_acc = MulticlassAccuracy(num_classes=num_classes, average='micro')

    def forward(self, x):
        return self.model(x)

    def configure_optimizers(self):
        optimizer = Adam(self.model.parameters(), lr=self.lr)
        
        return {'optimizer': optimizer, 'monitor': 'val_loss'}

    def training_step(self, train_batch, batch_idx):
        x, y = train_batch

        spk, mem = self(x)

        loss = self.loss_fn(spk, y)
        self.log('train_loss', loss, on_step=False, on_epoch=True, prog_bar=True)

        self.train_acc(spk.sum(dim=0), y)
        self.log('train_acc', self.train_acc, on_step=False, on_epoch=True, prog_bar=True)
        
        return loss
    
    def validation_step(self, val_batch, batch_idx):
        x, y = val_batch

        spk, mem = self(x)

        loss = self.loss_fn(spk, y)
        self.log('val_loss', loss, prog_bar=True)

        self.val_acc(spk.sum(dim=0), y)
        self.log('val_acc', self.val_acc, on_step=False, on_epoch=True, prog_bar=True)
    
    def test_step(self, test_batch, batch_idx):
        x, y = test_batch

        spk, mem = self(x)
 
        self.test_acc(spk.sum(dim=0), y)
        self.log('test_acc', self.test_acc, on_step=False, on_epoch=True, prog_bar=True)

class SpikingMLP(nn.Module):
    """snnTorch implementation of a spiking MLP.
    """
    
    def __init__(self, layer_sizes, beta=.95, num_steps=25,
        beta_cv = 0., # NOTE: coefficient of variation for betas' perturbation
        beta_noise_seed = 0,
        bias=False
    ):
        super().__init__()
        
        # initialize neurons, biases and synaptic weights
        lifs = []
        biases = nn.ParameterList()
        Ws = nn.ParameterList()
        for size_pre, size_post in zip(layer_sizes[:-1], layer_sizes[1:]):
            if beta_cv == 0.:
                lifs.append(snn.Leaky(beta, reset_mechanism='zero'))
            else:
                lifs.append(snn.Leaky(beta + beta_cv*beta*torch.randn(size_post,
                    generator=torch.Generator(device='cuda:7').manual_seed(beta_noise_seed), device='cuda:7'),
                    reset_mechanism='zero'
                )) # TODO: fix device

            W = torch.zeros(size_pre + 1, size_post)
            nn.init.kaiming_normal_(W.T, mode='fan_in')
            
            biases.append(W[-1])
            Ws.append(W[:-1])
        
        self.lifs = lifs
        self.biases = biases
        self.Ws = Ws
        self.num_steps = num_steps

        # instance variables for robustness analysis
        self.weight_cv = 0. # NOTE: coefficient of variation for weights' perturbation
        self.weight_noise_seed = 0

        self.bias = bias
    
    def forward(self, x):
        # initialize membrane potentials
        mems = [lif.init_leaky().to(x.device) for lif in self.lifs]

        # record the final layer
        spk_rec = []
        mem_rec = []

        # record the neural activity (spikes) of all layers
        self.spks = [torch.zeros(x.shape[0], self.num_steps, W.shape[1], device=x.device) for W in self.Ws]

        # simulation
        for step in range(self.num_steps):
            # input as constant voltages across steps
            spk = x[:,step]
            
            # forward the signals
            for i, (W, bias, lif) in enumerate(zip(self.Ws, self.biases, self.lifs)):
                # compute perturbed weights
                # NOTE: computation has been placed here to keep compatibility with subclasses
                if self.weight_cv > 0.:
                    W = W + torch.normal(0, self.weight_cv*torch.abs(W),
                        generator=torch.Generator(device=W.device).manual_seed(self.weight_noise_seed))
                
                cur = spk@W
                if self.bias: cur += bias

                # LIF neurons
                spk, mems[i] = lif(cur, mems[i])

                # record the spikes of the current layer at the current time step
                self.spks[i][:,step] = spk
        
            # record the final layer
            spk_rec.append(spk)
            mem_rec.append(mems[i])

        return torch.stack(spk_rec, dim=0), torch.stack(mem_rec, dim=0)
    
class SpikingGEM(SpikingMLP):
    """Spiking implementation of the Genetic neuroEvolution Model (Barabási et al., 2023).
    """
    
    def __init__(self, layer_sizes, num_genes, beta=.95, num_steps=25, **kwargs):
        super().__init__(layer_sizes, beta, num_steps, **kwargs)
        
        # initialize expression patterns
        Xs = nn.ParameterList()
        for size in layer_sizes:
            X = torch.zeros(size, num_genes)
            nn.init.kaiming_normal_(X.T, mode='fan_out') # NOTE: best init. so far
            Xs.append(X)
        
        # initialize genetic rules
        O = torch.randn(num_genes, num_genes)/num_genes

        self.Xs = Xs
        self.O = nn.Parameter(O)
        
        # remove learnable weights (weights will be computed at each forward pass)
        del self.Ws

    def compute_weights(self):
        """Computes weights by de-compressing the genotype.
        """
        
        Ws = []
        for X_in, X_out in zip(
            self.Xs[:-1],
            self.Xs[1:]
        ):
            Ws.append(X_in@self.O@X_out.T)
        self.Ws = Ws
    
    def forward(self, x):
        # compute weight matrices
        self.compute_weights()

        return super().forward(x)
    
    def predict(self, x):
        """Forward pass optimized for inference only. Weights are not de-compressed in real-time.
        """

        return super().forward(x)