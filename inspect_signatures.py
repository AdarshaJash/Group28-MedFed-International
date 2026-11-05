import inspect
from fedhealth import data_unified, engine_unified

functions = [
    (data_unified, "load_medmnist_bundle"),
    (data_unified, "load_csv_bundle"),
    (engine_unified, "build_model"),
    (engine_unified, "train_local"),
    (engine_unified, "evaluate_model"),
    (engine_unified, "run_federated"),
    (engine_unified, "run_centralized"),
    (engine_unified, "run_local_baselines"),
]

for module, name in functions:
    fn = getattr(module, name)
    print("\n" + "=" * 60)
    print(name, inspect.signature(fn))
    print(inspect.getdoc(fn) or "No documentation")
