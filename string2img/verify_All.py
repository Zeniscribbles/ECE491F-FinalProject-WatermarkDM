"""
@brief Phase 3: Fingerprint Verification from Embedded Images

This script verifies fingerprints extracted from CIFAR-10 images previously embedded
via Phase 2. It compares decoder output to ground-truth bitstrings and reports accuracy.
"""

import os
import glob
import torch
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
from tqdm import tqdm  # Add this import at the top
from PIL import Image

from models import StegaStampDecoder  # Ensure this is importable based on your working dir

# ----- CONFIG -----
BIT_LENGTH = 64
IMAGE_RESOLUTION = 32
BATCH_SIZE = 128
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# ----- PATHS -----
IMAGE_ROOT = "../edm/datasets/embedded/cifar10/images"
NOTE_ROOT = "../edm/datasets/embedded/cifar10/note"
CHECKPOINT_DIR = "./_output/cifar10/checkpoints"

# ----- HELPERS -----
def get_latest_decoder(checkpoint_dir, bit_length):
    pattern = f"stegastamp_{bit_length}_*_decoder.pth"
    candidates = sorted(glob.glob(os.path.join(checkpoint_dir, pattern)))
    if not candidates:
        raise FileNotFoundError("No matching decoder checkpoint found.")
    return candidates[-1]

class FingerprintedDataset(Dataset):
    def __init__(self, image_dir, note_path):
        self.samples = []
        self.image_dir = image_dir
        self.transform = transforms.Compose([
            transforms.Resize(IMAGE_RESOLUTION),
            transforms.CenterCrop(IMAGE_RESOLUTION),
            transforms.ToTensor()
        ])
        with open(note_path, "r") as f:
            for line in f:
                filename, bitstring = line.strip().split()
                bits = torch.tensor([int(b) for b in bitstring], dtype=torch.float32)
                self.samples.append((filename, bits))

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        filename, gt_bits = self.samples[idx]
        image_path = os.path.join(self.image_dir, filename)
        image = Image.open(image_path).convert("RGB")
        image = self.transform(image)
        return image, gt_bits

def load_decoder(path):
    model = StegaStampDecoder(IMAGE_RESOLUTION, 3, fingerprint_size=BIT_LENGTH)
    model.load_state_dict(torch.load(path, map_location=DEVICE))
    return model.to(DEVICE).eval()

# ----- MAIN VERIFICATION -----
def verify_all():
    decoder_path = get_latest_decoder(checkpoint_dir, bit_length)
    decoder = load_decoder(decoder_path)

    total_bits = 0
    total_correct = 0
    skipped = 0

    print("🔍 Verifying embedded fingerprints from Phase 2...\n")

    for i in tqdm(range(50), desc="Verifying CIFAR-10 folders"):
        folder = str(i).zfill(5)
        image_dir = os.path.join(image_root, folder)
        note_path = os.path.join(note_root, folder, "embedded_fingerprints.txt")

        if not os.path.exists(note_path):
            tqdm.write(f" Skipping {folder}, note file not found.")
            skipped += 1  # ← Track how many folders are skipped
            continue

        dataset = FingerprintedDataset(image_dir, note_path)
        dataloader = DataLoader(dataset, batch_size=BATCH_SIZE, shuffle=False)

        with torch.no_grad():
            for images, gt_fingerprints in dataloader:
                images = images.to(DEVICE)
                gt_fingerprints = gt_fingerprints.to(DEVICE)

                decoded = decoder(images)
                decoded_bin = (decoded > 0).float()
                matches = (decoded_bin == gt_fingerprints).float().sum().item()
                total_bits += gt_fingerprints.numel()
                total_correct += matches

    if total_bits == 0:
        print("\n No valid fingerprint data found. Phase 3 cannot proceed.")
    else:
        bitwise_accuracy = total_correct / total_bits
        print(f"\n Overall Bitwise Accuracy: {bitwise_accuracy:.4f}")

    accuracy = total_correct / total_bits
    print(f"\n Finished Verification")
    print(f"Bitwise Accuracy: {accuracy:.4f}")
    print(f"Skipped Folders: {skipped} / 50")

if __name__ == "__main__":
    verify_all()
