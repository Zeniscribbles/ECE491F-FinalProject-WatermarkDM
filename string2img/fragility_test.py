import os
import io
from PIL import Image, ImageFilter
import torch
from torchvision.transforms import ToTensor
from models import StegaStampDecoder
from tqdm import tqdm

=================================================================
# Run in Google Colab using: !python string2img/fragility_test.py
=================================================================

# ----------------------------
# Configs
# ----------------------------
IMAGE_RESOLUTION = 32
IMAGE_CHANNELS = 3
FINGERPRINT_SIZE = 64

# === UPDATE THIS: to the latest decoder you trained ===
DECODER_PATH = "./_output/cifar10/checkpoints/stegastamp_64_07052025_00:20:32_decoder.pth"

# Directory structure
IMAGE_DIR = "../edm/datasets/embedded/cifar10/images"
NOTE_DIR = "../edm/datasets/embedded/cifar10/note"

# ----------------------------
# Image Transformations
# ----------------------------

def jpeg_compress(image, quality=50):
    buffer = io.BytesIO()
    image.save(buffer, format='JPEG', quality=quality)
    buffer.seek(0)
    return Image.open(buffer)

def blur_image(image, radius=1):
    return image.filter(ImageFilter.GaussianBlur(radius))

def add_noise(tensor_img, noise_level=0.1):
    noise = torch.randn_like(tensor_img) * noise_level
    return torch.clamp(tensor_img + noise, 0, 1)

# ----------------------------
# Load Decoder
# ----------------------------
decoder = StegaStampDecoder(
    image_resolution=IMAGE_RESOLUTION,
    image_channels=IMAGE_CHANNELS,
    fingerprint_size=FINGERPRINT_SIZE
)
decoder.load_state_dict(torch.load(DECODER_PATH))
decoder.eval().cuda()

# ----------------------------
# Run Tests
# ----------------------------

def run_test(attack_fn, attack_name):
    print(f"\n=== Testing robustness against: {attack_name} ===")

    for folder in sorted(os.listdir(NOTE_DIR)):
        note_path = os.path.join(NOTE_DIR, folder, "embedded_fingerprints.txt")
        image_folder = os.path.join(IMAGE_DIR, folder)

        if not os.path.exists(note_path):
            continue

        # Load ground truth fingerprints
        fp_map = {}
        with open(note_path, "r") as f:
            for line in f:
                filename, fingerprint = line.strip().split()
                fp_map[filename] = torch.tensor([int(b) for b in fingerprint])

        avg_acc = []
        for filename in tqdm(fp_map.keys(), desc=f"Folder {folder}"):
            img_path = os.path.join(image_folder, filename)
            image = Image.open(img_path).convert("RGB")

            # Original → attack → tensor
            attacked = attack_fn(image)
            pert_tensor = ToTensor()(attacked).unsqueeze(0).cuda()

            # Decode
            with torch.no_grad():
                decoded_fp = decoder(pert_tensor).round().long().squeeze()
                true_fp = fp_map[filename].cuda()

                acc = (decoded_fp == true_fp).float().mean().item()
                avg_acc.append(acc)

        print(f"[{folder}] Avg bitwise accuracy = {sum(avg_acc) / len(avg_acc):.4f}")

# ----------------------------
# Run All Attacks
# ----------------------------
if __name__ == "__main__":
    run_test(lambda img: jpeg_compress(img, quality=25), "JPEG Compression (Q=25)")
    run_test(lambda img: blur_image(img, radius=2), "Gaussian Blur (r=2)")
    run_test(lambda img: ToTensor()(img), "No Attack (baseline)")
