import numpy as np
from PIL import Image
from pathlib import Path

data = np.load(r"C:\Users\HP\.medmnist\pneumoniamnist.npz")

images = data["test_images"]
labels = data["test_labels"].reshape(-1)

output = Path("MedFed_Test_Images")
normal_dir = output / "Normal"
pneumonia_dir = output / "Pneumonia"

normal_dir.mkdir(parents=True, exist_ok=True)
pneumonia_dir.mkdir(parents=True, exist_ok=True)

for class_id, folder, name in [
    (0, normal_dir, "Normal"),
    (1, pneumonia_dir, "Pneumonia")
]:
    indices = np.where(labels == class_id)[0][:5]

    for number, index in enumerate(indices, start=1):
        image = Image.fromarray(images[index].astype("uint8"))
        image.save(folder / f"{name}_{number}.png")

    print(f"{name}: {len(indices)} images extracted")

print("\nExtraction completed!")
print("Location:", output.resolve())
