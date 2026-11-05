import ast
import importlib.util
from pathlib import Path

root = Path.cwd()
report = []

report.append("MEDFED PROJECT AUDIT")
report.append("=" * 55)

# 1. Python syntax
report.append("\n1. PYTHON SYNTAX")
files = list(root.rglob("*.py"))
files = [p for p in files if ".venv" not in p.parts and "__pycache__" not in p.parts]
syntax_failures = []

for p in files:
    try:
        ast.parse(p.read_text(encoding="utf-8-sig"))
    except Exception as e:
        syntax_failures.append(f"{p.relative_to(root)}: {e}")

report.append(f"Python files checked: {len(files)}")
report.append(f"Syntax failures: {len(syntax_failures)}")
report.extend(syntax_failures or ["All Python files passed syntax checking."])

# 2. Dataset availability
report.append("\n2. DATASET CACHE")
cache = Path.home() / ".medmnist"
datasets = [
    "pneumoniamnist.npz",
    "organamnist.npz",
    "breastmnist.npz",
    "pathmnist.npz",
]
for name in datasets:
    p = cache / name
    report.append(f"{name}: {'FOUND' if p.exists() else 'NOT FOUND'}")

# 3. Dependencies
report.append("\n3. DEPENDENCIES")
for name in ["torch", "torchvision", "medmnist", "streamlit",
             "sklearn", "pandas", "numpy", "flwr"]:
    report.append(f"{name}: {'AVAILABLE' if importlib.util.find_spec(name) else 'MISSING'}")

# 4. Module capability indicators
report.append("\n4. MODULE CODE INDICATORS")
source = "\n".join(p.read_text(encoding="utf-8-sig", errors="ignore")
                   for p in files).lower()

for name, terms in {
    "Pneumonia": ["pneumoniamnist"],
    "CT / Organ": ["organamnist"],
    "Cancer": ["breastmnist"],
    "Diabetes": ["diabetes"],
    "Blood tests": ["blood test", "blood_test", "bloodtest"],
}.items():
    found = [term for term in terms if term in source]
    report.append(f"{name}: {'CODE REFERENCE FOUND' if found else 'NO CODE REFERENCE FOUND'}")

output = "\n".join(report)
print(output)

out = root / "artifacts" / "medfed_audit.txt"
out.parent.mkdir(exist_ok=True)
out.write_text(output, encoding="utf-8")
print(f"\nAudit saved to: {out}")
