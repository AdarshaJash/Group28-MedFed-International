import io, json
from pathlib import Path
import numpy as np
import pandas as pd
import streamlit as st
import torch
from PIL import Image
from torchvision import transforms
from fedhealth.data_unified import load_medmnist_bundle, load_csv_bundle
from fedhealth.engine_unified import run_federated, run_local_baselines, run_centralized
from fedhealth.models import MedicalImageCNN

st.set_page_config(page_title="MedFed | Group 28", page_icon="🏥", layout="wide")
st.markdown("""<style>.block-container{padding-top:1.3rem;max-width:1350px}.hero{padding:1.35rem;border-radius:16px;background:linear-gradient(120deg,#10243c,#176b87);color:white}.hero h1{color:white;margin:0}.small{color:#667085}</style>""",unsafe_allow_html=True)
st.markdown('<div class="hero"><h1>MedFed · Group 28</h1><p>Modular Federated Learning Research Platform for Medical Image and Tabular Classification</p><p>Simulated hospitals · Non-IID data · FedAvg / FedProx · Reproducible experiments</p></div>',unsafe_allow_html=True)
st.warning("Research prototype only. These benchmark models are not clinically validated and must not be used for diagnosis or treatment decisions.")

MODULES={
 "Pneumonia (chest X-ray)":{"kind":"image","key":"pneumoniamnist","dataset":"PneumoniaMNIST","task":"Normal vs pneumonia benchmark classification","note":"Small, downsampled 28×28 chest X-ray benchmark."},
 "CT scan (organ classification)":{"kind":"image","key":"organamnist","dataset":"OrganAMNIST","task":"Abdominal CT organ classification","note":"2D axial CT slices; organ classification, not tumour diagnosis."},
 "Cancer-related imaging (breast ultrasound)":{"kind":"image","key":"breastmnist","dataset":"BreastMNIST","task":"Benign vs malignant breast ultrasound benchmark classification","note":"Research benchmark only; ultrasound images are not CT or histopathology."},
 "Diabetes (labelled CSV)":{"kind":"csv","task":"User-selected diabetes target prediction","note":"Upload a documented, legally usable CSV with a clearly defined diabetes label."},
 "Blood tests (labelled CSV)":{"kind":"csv","task":"User-selected laboratory outcome classification","note":"Upload a documented CSV; specify exactly what the target label means. Lab values alone do not establish a diagnosis."},
}
with st.sidebar:
 st.header("1 · Choose research module")
 module=st.selectbox("Disease / task",list(MODULES))
 info=MODULES[module]
 st.caption(info["note"])
 st.header("2 · Federated experiment")
 clients=st.slider("Simulated hospitals",2,10,3)
 rounds=st.slider("Federated rounds",1,20,5)
 epochs=st.slider("Local epochs per round",1,5,1)
 strategy=st.selectbox("Federated strategy",["FedAvg","FedProx"])
 partition=st.selectbox("Client data distribution",["Non-IID","IID"])
 seed=st.number_input("Random seed",0,999999,42)
 st.caption("FedProx uses a proximal regularisation term (μ = 0.01) in this prototype.")

if info["kind"]=="image":
 c1,c2=st.columns(2)
 with c1: max_train=st.select_slider("Maximum training images",options=[300,600,1200,2400,4000],value=1200)
 with c2: max_test=st.select_slider("Maximum test images",options=[100,200,400,800,1500],value=400)
 st.info(f"Dataset: **{info['dataset']}** · Task: {info['task']}. First run downloads the public MedMNIST benchmark.")
 uploaded_csv=None; target=None
else:
 uploaded_csv=st.file_uploader("Upload a labelled CSV dataset",type=["csv"])
 if uploaded_csv:
  try:
   preview=pd.read_csv(uploaded_csv)
   st.write("Dataset preview",preview.head(8))
   target=st.selectbox("Select the target / label column",list(preview.columns))
  except Exception as e:
   st.error(f"Could not read CSV: {e}"); target=None
 else: target=None
 st.info("No patient dataset is bundled for diabetes or blood tests. Upload a public, documented dataset with a target column. Do not upload identifiable patient records.")

run=st.button("🚀 Run federated research experiment",type="primary",use_container_width=True,disabled=(info["kind"]=="csv" and (uploaded_csv is None or target is None)))
if "medfed_result" not in st.session_state: st.session_state.medfed_result=None
if run:
 try:
  with st.status("Preparing data and simulated hospitals…",expanded=True) as status:
   if info["kind"]=="image":
    bundle=load_medmnist_bundle(info["key"],num_clients=clients,partition=partition,max_train=max_train,max_test=max_test,seed=int(seed))
   else:
    uploaded_csv.seek(0)
    bundle=load_csv_bundle(uploaded_csv.getvalue(),target,num_clients=clients,partition=partition,seed=int(seed))
   st.write(f"Prepared {sum(bundle.client_sizes)} training records across {len(bundle.client_loaders)} simulated hospitals.")
   progress=st.progress(0)
   def cb(i,n,row): progress.progress(i/n,text=f"Round {i}/{n} · accuracy {row['accuracy']:.3f} · macro-F1 {row['macro_f1']:.3f}")
   result=run_federated(bundle,rounds=rounds,local_epochs=epochs,seed=int(seed),progress=cb,strategy=strategy)
   st.write("Training local-only and centralised comparison baselines…")
   # Equal local epoch count is reported explicitly; total compute budgets differ by design.
   baselines=run_local_baselines(bundle,epochs=epochs*rounds,seed=int(seed))+[run_centralized(bundle,epochs=epochs*rounds,seed=int(seed))]
   st.session_state.medfed_result={"module":module,"bundle":bundle,"result":result,"baselines":baselines,
     "config":{"clients":clients,"rounds":rounds,"local_epochs":epochs,"strategy":strategy,"partition":partition,"seed":int(seed)}}
   status.update(label="Experiment completed",state="complete",expanded=False)
 except Exception as e:
  st.exception(e)

saved=st.session_state.medfed_result
if saved and saved["module"] != module:
 saved=None
if saved:
 bundle=saved["bundle"]; result=saved["result"]; metrics=result["final"]
 st.divider(); st.header(f"Results · {saved['module']}")
 cols=st.columns(5)
 cols[0].metric("Accuracy",f"{metrics['accuracy']:.1%}")
 cols[1].metric("Macro-F1",f"{metrics['macro_f1']:.3f}")
 cols[2].metric("Sensitivity / macro recall",f"{metrics['recall_sensitivity']:.3f}")
 cols[3].metric("Specificity",f"{metrics['specificity']:.3f}" if metrics["specificity"] is not None else "N/A")
 cols[4].metric("ROC-AUC",f"{metrics['roc_auc']:.3f}" if metrics["roc_auc"] is not None else "N/A")
 a,b=st.columns(2)
 with a:
  st.subheader("Global model performance by round")
  st.line_chart(pd.DataFrame(result["history"]).set_index("round")[["accuracy","macro_f1","balanced_accuracy"]])
 with b:
  st.subheader("Confusion matrix")
  st.dataframe(pd.DataFrame(metrics["confusion_matrix"],index=[f"Actual {x}" for x in bundle.class_names],columns=[f"Predicted {x}" for x in bundle.class_names]),use_container_width=True)
 st.subheader("Simulated hospital data distribution")
 st.dataframe(pd.DataFrame([{"Hospital":f"Hospital {i+1}","Training samples":bundle.client_sizes[i],**bundle.client_label_counts[i]} for i in range(len(bundle.client_sizes))]),use_container_width=True)
 st.subheader("Baseline comparison")
 comparison=pd.DataFrame([{"method":"Federated global model",**{k:metrics[k] for k in ["accuracy","macro_f1","balanced_accuracy","recall_sensitivity","specificity","roc_auc"]}}]+saved["baselines"])
 st.dataframe(comparison,use_container_width=True)
 st.bar_chart(comparison.set_index("method")[["accuracy","macro_f1"]])
 st.subheader("Per-class evaluation")
 st.dataframe(pd.DataFrame(metrics["classification_report"]).T,use_container_width=True)
 report={"project":"MedFed Group 28","module":saved["module"],"dataset":bundle.dataset_name,"class_names":bundle.class_names,
  "task_type":bundle.task_type,"config":saved["config"],"client_sizes":bundle.client_sizes,"client_label_counts":bundle.client_label_counts,
  "round_history":result["history"],"final_metrics":{k:v for k,v in metrics.items() if k not in {"y_true","y_pred","y_prob","classification_report"}},
  "baselines":saved["baselines"],"elapsed_seconds":result["elapsed"],"device":result["device"],
  "limitations":["Simulated hospitals run on one computer; no real hospital network is connected.","No formal differential privacy or secure aggregation is implemented.","Benchmark performance is not clinical validation.","CSV results depend on the provenance, labels, licensing and quality of the uploaded dataset."]}
 st.download_button("Download experiment report (JSON)",json.dumps(report,indent=2,default=str),"medfed_research_report.json","application/json")
 st.download_button("Download round metrics (CSV)",pd.DataFrame(result["history"]).to_csv(index=False),"medfed_round_metrics.csv","text/csv")
 model_bytes=io.BytesIO(); torch.save(result["model_state"],model_bytes)
 st.download_button("Download global model weights",model_bytes.getvalue(),"medfed_global_model.pt","application/octet-stream")
 if bundle.task_type=="image":
  st.subheader("Research image inference demo")
  st.caption("The output is a benchmark-model prediction, not a diagnosis. Use only images matching the dataset modality and preprocessing assumptions.")
  img_file=st.file_uploader("Upload a matching image (PNG/JPG)",type=["png","jpg","jpeg"],key="inference_image")
  if img_file:
   img=Image.open(img_file).convert("L"); st.image(img,caption="Uploaded research image",width=260)
   tfm=transforms.Compose([transforms.Resize((28,28)),transforms.ToTensor(),transforms.Normalize((0.5,),(0.5,))])
   model=MedicalImageCNN(len(bundle.class_names)); model.load_state_dict(result["model_state"]); model.eval()
   with torch.no_grad(): probs=torch.softmax(model(tfm(img).unsqueeze(0)),dim=1)[0].tolist()
   st.write({name:round(float(p),4) for name,p in zip(bundle.class_names,probs)})
   st.warning("A model score is not a calibrated clinical probability. Do not use this output for patient care.")

with st.expander("Research methodology, reproducibility and privacy"):
 st.markdown("""
 **Federated simulation:** each simulated client trains locally on its own partition. The coordinator aggregates model parameters using sample-count-weighted FedAvg, or FedProx local optimisation. The dashboard's default simulator runs in one Python process; it is not a distributed hospital deployment.

 **Data separation:** MedMNIST's official train, validation and test splits are used. CSV uploads are split into train/validation/test subsets with stratification, and scaling is fitted on training data only. Patient-level splitting must be performed before upload when repeated records from one patient exist.

 **Privacy:** raw training partitions stay within local client objects in this simulation, but model updates can leak information. This version does not implement differential privacy, secure aggregation, authentication or access control. Do not claim formal privacy guarantees.

 **Clinical scope:** PneumoniaMNIST is a small X-ray benchmark; OrganAMNIST is CT organ classification; BreastMNIST is breast ultrasound benign/malignant benchmark classification. These are distinct tasks and modalities. Diabetes and blood-test modules accept user-supplied labelled CSV datasets; the target definition must be documented.
 """)
 st.caption("Research prototype. No clinical validation, regulatory approval or real-hospital integration is claimed.")
if not saved:
 st.subheader("Research workspace")
 st.write("Choose a module in the sidebar. The three image benchmarks can be downloaded automatically; diabetes and blood-test experiments require your documented labelled CSV dataset.")
 st.markdown("**Research comparison:** local-only vs centralised vs federated training, IID vs non-IID partitions, FedAvg vs FedProx, and communication-round analysis.")
 st.info("Start with the pneumonia module to verify the pipeline, then repeat the same experiment protocol for each other module.")
