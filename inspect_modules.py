import ast
from pathlib import Path

for filename in [
    "fedhealth/data_unified.py",
    "fedhealth/engine_unified.py",
    "fedhealth/models.py",
    "fedhealth/data.py",
    "fedhealth/engine.py",
]:
    path = Path(filename)
    print("\n" + "=" * 65)
    print(filename)

    if not path.exists():
        print("FILE MISSING")
        continue

    tree = ast.parse(path.read_text(encoding="utf-8-sig"))

    for node in tree.body:
        if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)):
            print(f"FUNCTION: {node.name} (line {node.lineno})")
        elif isinstance(node, ast.ClassDef):
            print(f"CLASS: {node.name} (line {node.lineno})")
            for item in node.body:
                if isinstance(item, (ast.FunctionDef, ast.AsyncFunctionDef)):
                    print(f"  METHOD: {item.name} (line {item.lineno})")
