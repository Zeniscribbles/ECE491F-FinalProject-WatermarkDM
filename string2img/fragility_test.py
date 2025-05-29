# @brief: Phase 2 - Fragility Testing
# This script evaluates the robustness of embedded watermarks in CIFAR-10 images
# against various image attacks (blur, noise, crop). It reports bitwise accuracy,
# MSE, SSIM, and PSNR for each attack.

import os
import glob
import torch
import torch.nn.functional as F
from torchvision import transforms
from torchvision.utils import save_image
from PIL import Image, ImageFilter
from skimage.metrics import structural_similarity as ssim
from skimage.metrics import peak_signal_noise_ratio as psnr

from models import StegaStampEncoder, StegaStampDecoder  # Adjust if needed

# ---------- Metrics ----------
def compute_attack_metrics(original, perturbed, fingerprint_gt, fingerprint_pred):
    fingerprint_pred_bin = (fingerprint_pred > 0).long()
    bitwise_accuracy = (fingerprint_pred_bin == fingerprint_gt).float().mean().item()

    mse = F.mse_loss(original, perturbed).item()
    o_np = original.squeeze().permute(1, 2, 0).detach().cpu().numpy()
    p_np = perturbed.squeeze().permute(1, 2, 0).detach().cpu().numpy()
    ssim_val = ssim(o_np, p_np, channel_axis=2, data_range=1.0)
    psnr_val = psnr(o_np, p_np, data_range=1.0)

    print(f"Bitwise Accuracy: {bitwise_accuracy:.4f}")
    print(f"MSE: {mse:.4f}")
    print(f"SSIM: {ssim_val:.4f}")
    print(f"PSNR: {psnr_val:.2f} dB")

# ---------- Checkpoint Finder ----------
def get_latest_checkpoint_pair(checkpoint_dir, bit_length=None):
    pattern = f'stegastamp_{bit_length}_*_decoder.pth' if bit_length else '*_decoder.pth'
    decoder_files = sorted(glob.glob(os.path.join(checkpoint_dir, pattern)))
    if not decoder_files:
        raise FileNotFoundError(f"No decoder found in {checkpoint_dir}")
    decoder_path = decoder_files[-1]
    encoder_path = decoder_path.replace('decoder', 'encoder')
    if not os.path.exists(encoder_path):
        raise FileNotFoundError(f"Encoder not found for {decoder_path}")
    return encoder_path, decoder_path

# ---------- Fingerprint Generator ----------
def generate_fingerprint(bit_length, batch_size):
    return torch.randint(0, 2, (batch_size, bit_length)).float()

# ---------- Attacks ----------
def apply_attack(tensor_images, attack_type='blur'):
    attacked = []
    for img in tensor_images:
        pil = transforms.ToPILImage()(img)
        if attack_type == 'blur':
            pil = pil.filter(ImageFilter.GaussianBlur(radius=2))
        elif attack_type == 'noise':
            noisy = img + 0.2 * torch.randn_like(img)
            pil = transforms.ToPILImage()(torch.clamp(noisy, 0, 1))
        elif attack_type == 'crop':
            pil = pil.crop((4, 4, 28, 28)).resize((32, 32))
        attacked.append(transforms.ToTensor()(pil))
    return torch.stack(attacked)

# ---------- Fragility Testing ----------
def test_fragility_all_attacks(encoder_path, decoder_path, attacks, batch_size=16, bit_length=64, image_resolution=32):
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    encoder = StegaStampEncoder(image_resolution, 3, fingerprint_size=bit_length).to(device)
    decoder = StegaStampDecoder(image_resolution, 3, fingerprint_size=bit_length).to(device)
    encoder.load_state_dict(torch.load(encoder_path, map_location=device))
    decoder.load_state_dict(torch.load(decoder_path, map_location=device))
    encoder.eval()
    decoder.eval()

    dummy_img = torch.ones((batch_size, 3, image_resolution, image_resolution)).to(device)
    fingerprint_gt = generate_fingerprint(bit_length, batch_size).to(device)
    embedded = encoder(fingerprint_gt, dummy_img)

    save_image(embedded.cpu(), "embedded_clean.png")

    for attack in attacks:
        print(f"\n--- Attack: {attack} ---")
        attacked = apply_attack(embedded.cpu(), attack_type=attack).to(device)
        save_image(attacked.cpu(), f"embedded_attacked_{attack}.png")
        with torch.no_grad():
            fingerprint_pred = decoder(attacked)
        compute_attack_metrics(embedded[0], attacked[0], fingerprint_gt[0], fingerprint_pred[0])

# ---------- MAIN ----------
if __name__ == "__main__":
    checkpoint_dir = './_output/cifar10/checkpoints'  # Adjust path if needed
    bit_length = 64
    attacks = ['blur', 'noise', 'crop']
    encoder_path, decoder_path = get_latest_checkpoint_pair(checkpoint_dir, bit_length)
    test_fragility_all_attacks(encoder_path, decoder_path, attacks)
