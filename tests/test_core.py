import io
import numpy as np
import pandas as pd
import torch
from torch.utils.data import DataLoader, TensorDataset
from fedhealth.data_unified import load_csv_bundle, _partition_indices
from fedhealth.engine_unified import fedavg, run_federated, evaluate_model
from fedhealth.models import TabularMLP

def test_fedavg_weighted_average():
    a={"w":torch.tensor([1.0,3.0]),"n":torch.tensor(1,dtype=torch.long)}
    b={"w":torch.tensor([5.0,7.0]),"n":torch.tensor(2,dtype=torch.long)}
    out=fedavg([a,b],[1,3])
    assert torch.allclose(out["w"],torch.tensor([4.0,6.0]))
    assert out["n"].item()==1

def test_partition_preserves_all_examples():
    y=np.array([0]*30+[1]*30)
    chunks=_partition_indices(y,4,"Non-IID",42)
    flat=[i for c in chunks for i in c]
    assert sorted(flat)==list(range(len(y)))
    assert all(len(c)>0 for c in chunks)

def test_csv_loader_and_federated_training():
    rng=np.random.default_rng(7)
    n=160
    glucose=rng.normal(110,20,n)
    bmi=rng.normal(27,5,n)
    sex=rng.choice(["F","M"],n)
    label=np.where((glucose>115)&(bmi>27),"positive","negative")
    df=pd.DataFrame({"patient_id":np.arange(n),"glucose":glucose,"bmi":bmi,"sex":sex,"outcome":label})
    bio=io.BytesIO(df.to_csv(index=False).encode())
    bundle=load_csv_bundle(bio.getvalue(),"outcome",num_clients=3,partition="Non-IID",seed=42)
    assert bundle.task_type=="tabular"
    assert len(bundle.client_loaders)==3
    result=run_federated(bundle,rounds=1,local_epochs=1,seed=42)
    assert len(result["history"])==1
    assert 0.0<=result["final"]["accuracy"]<=1.0
    assert len(result["final"]["confusion_matrix"])==2

if __name__=="__main__":
    test_fedavg_weighted_average(); test_partition_preserves_all_examples(); test_csv_loader_and_federated_training()
    print("All MedFed core smoke tests passed.")
