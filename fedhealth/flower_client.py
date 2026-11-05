"""Optional Flower adapter example. A real deployment must load data local to each client."""
import numpy as np
import torch
import flwr as fl
from .model import SmallCNN
from .engine import train_one, evaluate

class MedFedNumPyClient(fl.client.NumPyClient):
    def __init__(self, train_loader, test_loader, num_classes=2):
        self.model=SmallCNN(num_classes)
        self.train_loader=train_loader; self.test_loader=test_loader
        self.device=torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    def get_parameters(self, config):
        return [v.detach().cpu().numpy() for _,v in self.model.state_dict().items()]
    def set_parameters(self, parameters):
        keys=list(self.model.state_dict().keys())
        state={k:torch.tensor(v) for k,v in zip(keys,parameters)}
        self.model.load_state_dict(state,strict=True)
    def fit(self, parameters, config):
        self.set_parameters(parameters); self.model.to(self.device)
        train_one(self.model,self.train_loader,self.device,epochs=int(config.get('local_epochs',1)))
        return self.get_parameters(config),len(self.train_loader.dataset),{'client_train_samples':len(self.train_loader.dataset)}
    def evaluate(self, parameters, config):
        self.set_parameters(parameters); self.model.to(self.device)
        m=evaluate(self.model,self.test_loader,self.device)
        return 1.0-m['accuracy'],len(self.test_loader.dataset),{'accuracy':m['accuracy'],'macro_f1':m['macro_f1']}
