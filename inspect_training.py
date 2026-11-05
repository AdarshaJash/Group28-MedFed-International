import inspect
from fedhealth import data_unified, engine_unified

checks = [
    (data_unified, "UnifiedBundle"),
    (data_unified, "load_medmnist_bundle"),
    (engine_unified, "build_model"),
    (engine_unified, "evaluate_model"),
    (engine_unified, "run_federated"),
]

for module, name in checks:
    obj = getattr(module, name)
    print("\n" + "=" * 70)
    print(f"SOURCE: {name}")
    try:
        print(inspect.getsource(obj))
    except Exception as e:
        print("Could not inspect:", e)
