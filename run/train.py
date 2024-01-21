
import torch
import torch.nn as nn
import pytorch_lightning as pl

from scripts.model.Unet_base import Unet
from scripts.maploader.maploader import get_loaders_from_params
from scripts.utils.run_utils import setup_trainer, get_parser
from scripts.utils.params import set_params

    
class SphericalUnet(pl.LightningModule):
    def __init__(self, model, params):
        super().__init__()
        self.save_hyperparameters(params)
        self.model = model(params)
        self.batch_size = params["train"]['batch_size']
        self.learning_rate = params["train"]['learning_rate']
        self.gamma = params["train"]['gamma']
        print("We are using Adam with lr = {}, gamma = {}".format(self.learning_rate, self.gamma))

        self.loss = nn.MSELoss()

    def training_step(self, batch, batch_idx):
        y, x = batch
        y_hat = self.model(x)
        loss = self.loss(y_hat, y)
        self.log('train_loss', loss)
        return loss
    
    def validation_step(self, batch, batch_idx):
        y, x = batch
        y_hat = self.model(x)
        loss = self.loss(y_hat, y)
        self.log('val_loss', loss)
        return loss
    
    def configure_optimizers(self):
        optimizer = torch.optim.Adam(self.parameters(), lr=self.learning_rate)
        scheduler = torch.optim.lr_scheduler.ExponentialLR(optimizer=optimizer, gamma=self.gamma)
        return {"optimizer": optimizer, "lr_scheduler": scheduler} 

if __name__ == '__main__':
    args = get_parser()
    args = args.parse_args()

    pl.seed_everything(1234)
    params = set_params(**vars(args))

    ### get training data
    train_loader, val_loader = get_loaders_from_params(params)

    #get model
    model = SphericalUnet(Unet, params)

    trainer = setup_trainer(params)
    trainer.fit(model, train_loader, val_loader)