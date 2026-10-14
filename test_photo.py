from pathlib import Path
from time import perf_counter

import torch
from PIL import Image, ImageOps


PROJECT_FOLDER = Path(__file__).resolve().parent

INPUT_PATH = PROJECT_FOLDER / "photo2.jpg"
OUTPUT_PATH = PROJECT_FOLDER / "anime_photo2.png"

if not INPUT_PATH.is_file():
    raise FileNotFoundError(
        "Put a picture named photo.jpg next to test_photo.py."
    )

# Keep downloaded model files inside this project.
torch.hub.set_dir(str(PROJECT_FOLDER / ".model_cache"))

REPOSITORY = "bryandlee/animegan2-pytorch:main"

print("Loading the anime model...")
print("The first run downloads the model code and weights.")

model = torch.hub.load(
    REPOSITORY,
    "generator",
    pretrained="face_paint_512_v2",
    device="cpu",
    trust_repo=True,
).eval()

# Load the project's image-preparation helper.
face2paint = torch.hub.load(
    REPOSITORY,
    "face2paint",
    size=512,
    device="cpu",
    trust_repo=True,
)

# Correct phone-photo orientation and convert to RGB.
with Image.open(INPUT_PATH) as source:
    photo = ImageOps.exif_transpose(source).convert("RGB")

print("Transforming your picture...")
started = perf_counter()

with torch.inference_mode():
    anime_photo = face2paint(model, photo)

elapsed = perf_counter() - started

anime_photo.save(OUTPUT_PATH)

print(f"Saved: {OUTPUT_PATH}")
print(f"Transformation took {elapsed:.2f} seconds.")

# Open the result in your default image viewer.
anime_photo.show()