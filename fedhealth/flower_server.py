"""Minimal Flower FedAvg strategy factory for a Flower-managed deployment.

The Streamlit demo uses an in-process simulator so students can run it easily.
For a multi-process Flower experiment, create a Flower ServerApp/ClientApp and
supply the same model/data routines; see the official Flower PyTorch tutorial.
"""
import flwr as fl

def make_strategy(rounds=5, fraction_fit=1.0, fraction_evaluate=1.0):
    return fl.server.strategy.FedAvg(
        fraction_fit=fraction_fit,
        fraction_evaluate=fraction_evaluate,
        min_fit_clients=2,
        min_evaluate_clients=2,
        min_available_clients=2,
        on_fit_config_fn=lambda rnd: {'local_epochs': 1, 'server_round': rnd},
    )

if __name__ == '__main__':
    print('This is a strategy factory, not a standalone server: attach it to a Flower server runtime.')
    print('For current Flower simulation setup, follow https://flower.ai/docs/framework/tutorial-series-write-your-first-flower-app-pytorch.html')
