"""Generic Flower NumPyClient and FedAvg strategy adapters.

These adapters provide the bridge for a Flower-managed multi-process experiment.
The Streamlit dashboard currently uses the deterministic in-process simulator in
engine_unified.py. To run distributed clients, construct one bundle per client
from data stored at that client and launch a Flower runtime separately.
"""
import numpy as np
import torch
from .engine_unified import build_model, train_local, evaluate_model

def make_numpy_client(bundle, train_loader, validation_loader=None, local_epochs=1):
    try:
        import flwr as fl
    except ImportError as exc:
        raise RuntimeError("Install the Flower dependency from requirements.txt to use the Flower adapter.") from exc

    class GenericMedFedClient(fl.client.NumPyClient):
        def __init__(self):
            self.model=build_model(bundle)
            self.device=torch.device("cuda" if torch.cuda.is_available() else "cpu")
        def get_parameters(self, config):
            return [v.detach().cpu().numpy() for v in self.model.state_dict().values()]
        def set_parameters(self, parameters):
            keys=list(self.model.state_dict().keys())
            state={k:torch.as_tensor(v) for k,v in zip(keys,parameters)}
            self.model.load_state_dict(state,strict=True)
        def fit(self, parameters, config):
            self.set_parameters(parameters); self.model.to(self.device)
            epochs=int(config.get("local_epochs",local_epochs))
            train_local(self.model,train_loader,self.device,epochs,bundle.class_weights)
            return self.get_parameters(config),len(train_loader.dataset),{"local_epochs":epochs}
        def evaluate(self, parameters, config):
            if validation_loader is None: return 0.0,0,{}
            self.set_parameters(parameters); self.model.to(self.device)
            metrics=evaluate_model(self.model,validation_loader,self.device,bundle.class_names)
            return float(1.0-metrics["accuracy"]),len(validation_loader.dataset),{"accuracy":metrics["accuracy"],"macro_f1":metrics["macro_f1"]}
    return GenericMedFedClient()

def make_fedavg_strategy(min_clients=2, local_epochs=1):
    try:
        import flwr as fl
    except ImportError as exc:
        raise RuntimeError("Install Flower from requirements.txt to use the Flower strategy.") from exc
    return fl.server.strategy.FedAvg(
        fraction_fit=1.0,fraction_evaluate=1.0,min_fit_clients=min_clients,
        min_evaluate_clients=min_clients,min_available_clients=min_clients,
        on_fit_config_fn=lambda rnd:{"local_epochs":local_epochs,"server_round":rnd})
