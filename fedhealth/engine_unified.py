"""Reusable in-process FedAvg engine for image and tabular classification."""
import copy, random, time
import numpy as np
import torch
from torch import nn
from sklearn.metrics import accuracy_score, balanced_accuracy_score, confusion_matrix, f1_score, precision_score, recall_score, classification_report, roc_auc_score
from .models import MedicalImageCNN, TabularMLP

def seed_all(seed=42):
    random.seed(seed); np.random.seed(seed); torch.manual_seed(seed)
    if torch.cuda.is_available(): torch.cuda.manual_seed_all(seed)

def build_model(bundle):
    if bundle.task_type == "image": return MedicalImageCNN(len(bundle.class_names), 1)
    return TabularMLP(bundle.input_dim, len(bundle.class_names))

def _forward_batch(model, x, device):
    x=x.to(device)
    return model(x)

def train_local(model, loader, device, epochs=1, class_weights=None, lr=0.001):
    if len(loader.dataset) == 0: raise ValueError("A simulated hospital has no training samples.")
    model.train(); optimizer=torch.optim.Adam(model.parameters(),lr=lr)
    loss_fn=nn.CrossEntropyLoss(weight=class_weights.to(device) if class_weights is not None else None)
    for _ in range(epochs):
        for x,y in loader:
            y=y.reshape(-1).long().to(device); optimizer.zero_grad(set_to_none=True)
            loss=loss_fn(_forward_batch(model,x,device),y); loss.backward(); optimizer.step()
    return model

def evaluate_model(model, loader, device, class_names):
    model.eval(); ys=[]; probs=[]
    with torch.no_grad():
        for x,y in loader:
            logits=_forward_batch(model,x,device)
            probs.extend(torch.softmax(logits,dim=1).cpu().numpy().tolist())
            ys.extend(y.reshape(-1).long().cpu().numpy().tolist())
    y=np.asarray(ys,dtype=int); p=np.asarray(probs,dtype=float)
    pred=p.argmax(axis=1); labels=list(range(len(class_names)))
    cm=confusion_matrix(y,pred,labels=labels)
    try:
        auc=float(roc_auc_score(y,p[:,1])) if len(class_names)==2 else float(roc_auc_score(y,p,multi_class='ovr',average='macro',labels=labels))
    except Exception: auc=None
    sensitivity=recall_score(y,pred,average='binary' if len(class_names)==2 else 'macro',zero_division=0)
    specificity=None
    if len(class_names)==2:
        tn,fp,fn,tp=cm.ravel(); specificity=float(tn/(tn+fp)) if tn+fp else 0.0
    return {"accuracy":float(accuracy_score(y,pred)),"balanced_accuracy":float(balanced_accuracy_score(y,pred)),
        "macro_f1":float(f1_score(y,pred,average='macro',zero_division=0)),
        "precision":float(precision_score(y,pred,average='binary' if len(class_names)==2 else 'macro',zero_division=0)),
        "recall_sensitivity":float(sensitivity),"specificity":specificity,"roc_auc":auc,
        "confusion_matrix":cm.tolist(),"classification_report":classification_report(y,pred,labels=labels,
        target_names=[str(x) for x in class_names],output_dict=True,zero_division=0),
        "y_true":y.tolist(),"y_pred":pred.tolist(),"y_prob":p.tolist()}

def fedavg(states, sizes):
    if not states or len(states)!=len(sizes) or sum(sizes)<=0: raise ValueError("Invalid or empty client updates.")
    total=float(sum(sizes)); result=copy.deepcopy(states[0])
    for key in result:
        if torch.is_floating_point(result[key]):
            result[key]=sum(state[key]*(n/total) for state,n in zip(states,sizes))
        else: result[key]=states[0][key].clone()
    return result

def run_federated(bundle, rounds=3, local_epochs=1, seed=42, progress=None, strategy="FedAvg"):
    if strategy not in {"FedAvg","FedProx"}: raise ValueError("Strategy must be FedAvg or FedProx.")
    seed_all(seed); device=torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    global_model=build_model(bundle).to(device); history=[]; started=time.time()
    for rnd in range(1,rounds+1):
        round_start=time.time(); updates=[]; sizes=[]
        global_state={k:v.detach().cpu().clone() for k,v in global_model.state_dict().items()}
        for loader,size in zip(bundle.client_loaders,bundle.client_sizes):
            if size<=0: continue
            client=build_model(bundle).to(device); client.load_state_dict(global_state)
            if strategy=="FedAvg":
                train_local(client,loader,device,local_epochs,bundle.class_weights)
            else:
                _train_fedprox(client,loader,device,local_epochs,bundle.class_weights,global_state,mu=0.01)
            updates.append({k:v.detach().cpu().clone() for k,v in client.state_dict().items()}); sizes.append(size)
            del client
        global_model.load_state_dict(fedavg(updates,sizes))
        metrics=evaluate_model(global_model,bundle.test_loader,device,bundle.class_names)
        row={"round":rnd,**{k:metrics[k] for k in ["accuracy","balanced_accuracy","macro_f1","recall_sensitivity","specificity","roc_auc"]},
             "seconds":time.time()-round_start,"strategy":strategy}
        history.append(row)
        if progress: progress(rnd,rounds,row)
    final=evaluate_model(global_model,bundle.test_loader,device,bundle.class_names)
    return {"model":global_model,"model_state":{k:v.detach().cpu() for k,v in global_model.state_dict().items()},
            "history":history,"final":final,"elapsed":time.time()-started,"device":str(device)}

def _train_fedprox(model,loader,device,epochs,class_weights,global_state,mu=0.01,lr=0.001):
    model.train(); opt=torch.optim.Adam(model.parameters(),lr=lr)
    loss_fn=nn.CrossEntropyLoss(weight=class_weights.to(device))
    ref={k:v.to(device) for k,v in global_state.items() if torch.is_floating_point(v)}
    for _ in range(epochs):
        for x,y in loader:
            opt.zero_grad(set_to_none=True); logits=model(x.to(device)); y=y.reshape(-1).long().to(device)
            proximal=sum(torch.sum((p-ref[n])**2) for n,p in model.named_parameters())
            loss=loss_fn(logits,y)+(mu/2.0)*proximal; loss.backward(); opt.step()
    return model

def run_centralized(bundle,epochs=3,seed=42):
    seed_all(seed); device=torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model=build_model(bundle).to(device); train_local(model,bundle.train_loader,device,epochs,bundle.class_weights)
    m=evaluate_model(model,bundle.test_loader,device,bundle.class_names)
    return {"method":"Centralised baseline",**{k:m[k] for k in ["accuracy","balanced_accuracy","macro_f1","recall_sensitivity","specificity","roc_auc"]}}

def run_local_baselines(bundle,epochs=3,seed=42):
    rows=[]; device=torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    for i,loader in enumerate(bundle.client_loaders):
        if not len(loader.dataset): continue
        seed_all(seed+i+1); model=build_model(bundle).to(device); train_local(model,loader,device,epochs,bundle.class_weights)
        m=evaluate_model(model,bundle.test_loader,device,bundle.class_names)
        rows.append({"method":f"Local-only hospital {i+1}",**{k:m[k] for k in ["accuracy","balanced_accuracy","macro_f1","recall_sensitivity","specificity","roc_auc"]}})
    return rows
