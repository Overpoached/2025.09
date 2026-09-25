import clip
import torch
from PIL import Image
from pathlib import Path

# number of image handled per batch
BATCH = 32

# Get the directory of the current script
script_dir = Path(__file__).resolve().parent
file_path = script_dir / "labels.txt"

device = "cuda" if torch.cuda.is_available() else "cpu"
model, preprocess = clip.load("ViT-B/32", device=device)
print("Device: ", device)

# Specify the directory path (use '.' for the current directory)
image_directory_path = script_dir / "images"

# Get all file paths in the directory and store them in a list
image_file_array = [str(file) for file in image_directory_path.iterdir() if file.is_file()]

print("Images count: ", len(image_file_array))

with open(file_path, 'r') as file:
    lines = [line.rstrip() for line in file]

print("Labels count: ", len(lines))
print("Labels:     ", lines)

print("===== Starting image identification =====")

texts = clip.tokenize(lines).to(device)

with torch.no_grad():
    text_features = model.encode_text(texts)
    text_features /= text_features.norm(dim=-1, keepdim=True)

    for start in range(0, len(image_file_array), BATCH):
        chunk = image_file_array[start:start + BATCH]
        images = torch.stack([preprocess(Image.open(f)) for f in chunk]).to(device)

        image_features = model.encode_image(images)
        image_features /= image_features.norm(dim=-1, keepdim=True)

        # cosine similarity × CLIP's scale (100), same as model() does internally
        probs = (100.0 * image_features @ text_features.T).softmax(dim=-1).cpu().numpy()

        for image_file, row in zip(chunk, probs):
            i = row.argmax()
            print("Label probs:", row)
            print(f"{image_file}:\n {lines[i]} ({row[i]:.2%})")
            print("----------------------------------------")



print("===== End image identification =====")