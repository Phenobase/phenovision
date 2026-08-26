"""Minimal PhenoVision inference from the Hugging Face Hub.

Runs the published phenobase/phenovision model on an image and prints flower/fruit
probabilities with the CORRECT label order and decision thresholds.

Usage:
    python hf_inference_smoke_test.py path/to/plant.jpg
    python hf_inference_smoke_test.py            # downloads a sample image

Requires the `phenovision-min` conda env (see ../env/phenovision-min.yml).

Label order (IMPORTANT): the model's output logits are
    logits[0] = FRUIT,  logits[1] = FLOWER
i.e. the model was trained on select(fruiting, flowering) and production inference
renames V1->fruit, V2->flower. Do NOT follow the older README snippet that had
these two swapped.
"""
import io, os, sys, urllib.request, urllib.parse
import torch
from PIL import Image
from transformers import AutoImageProcessor, AutoModelForImageClassification

os.environ.setdefault("HF_HUB_DISABLE_SYMLINKS_WARNING", "1")

REPO = "phenobase/phenovision"
# Until preprocessor_config.json is present in the repo, point this at the local
# model_cards/ dir which contains it. Once uploaded, PROC_SRC can just be REPO.
PROC_SRC = os.environ.get(
    "PHENOVISION_PROC",
    os.path.join(os.path.dirname(__file__), "..", "model_cards"),
)

# Decision thresholds (from the model card / final_buffer_params.csv)
THRESHOLDS = {"flower": 0.48, "fruit": 0.60}


def load_image(arg):
    if arg:
        return Image.open(arg).convert("RGB")
    url = "https://commons.wikimedia.org/wiki/Special:FilePath/" + urllib.parse.quote("A sunflower.jpg")
    req = urllib.request.Request(url, headers={"User-Agent": "phenovision-smoke/1.0"})
    return Image.open(io.BytesIO(urllib.request.urlopen(req, timeout=45).read())).convert("RGB")


def main():
    img = load_image(sys.argv[1] if len(sys.argv) > 1 else None)
    processor = AutoImageProcessor.from_pretrained(PROC_SRC)
    model = AutoModelForImageClassification.from_pretrained(REPO).eval()

    inputs = processor(images=img, return_tensors="pt")
    with torch.no_grad():
        probs = torch.sigmoid(model(**inputs).logits)[0]

    fruit_prob = probs[0].item()   # logits[0] == fruit
    flower_prob = probs[1].item()  # logits[1] == flower

    def call(p, thr):
        return "DETECTED" if p > thr else "not detected"

    print(f"processor: {type(processor).__name__}   image: {img.size[0]}x{img.size[1]}")
    print(f"  flower: {flower_prob:.3f}  ({call(flower_prob, THRESHOLDS['flower'])})")
    print(f"  fruit:  {fruit_prob:.3f}  ({call(fruit_prob, THRESHOLDS['fruit'])})")


if __name__ == "__main__":
    main()
