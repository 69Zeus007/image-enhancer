import sys
import os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))
import torch
import skimage
from model.enhancer_torch import SRCNN
from PIL import Image
import torchvision.transforms as T
import os
import numpy as np
from skimage.metrics import peak_signal_noise_ratio as psnr, structural_similarity as ssim

def load_model(path="srcnn.pt"):
    model = SRCNN()
    model.load_state_dict(torch.load(path, map_location="cpu"))
    model.eval()
    return model

def preprocess_y_channel(image):
    ycbcr = image.convert("YCbCr")
    y = ycbcr.split()[0]
    return T.ToTensor()(y).unsqueeze(0), ycbcr.split()[1], ycbcr.split()[2]

def evaluate(model, lr_path, hr_path):
    lr_img = Image.open(lr_path).convert("RGB")
    hr_img = Image.open(hr_path).convert("RGB")

    # Resize LR to HR size using bicubic
    lr_img = lr_img.resize(hr_img.size, Image.BICUBIC)

    # Preprocess
    input_y, cb, cr = preprocess_y_channel(lr_img)
    target_y, _, _ = preprocess_y_channel(hr_img)

    with torch.no_grad():
        output_y = model(input_y).squeeze(0).clamp(0.0, 1.0)

    # Convert to numpy
    output_np = output_y.squeeze().numpy()
    target_np = target_y.squeeze().numpy()

    # Compute metrics
    psnr_val = psnr(target_np, output_np, data_range=1.0)
    ssim_val = ssim(target_np, output_np, data_range=1.0)

    return psnr_val, ssim_val

if __name__ == "__main__":
    model = load_model()
    test_lr_dir = "DIV2K/DIV2K_valid_LR_bicubic/X2"
    test_hr_dir = "DIV2K/DIV2K_valid_HR"

    lr_images = sorted(os.listdir(test_lr_dir))
    hr_images = sorted(os.listdir(test_hr_dir))

    scores = []
    for lr_name, hr_name in zip(lr_images, hr_images):
        lr_path = os.path.join(test_lr_dir, lr_name)
        hr_path = os.path.join(test_hr_dir, hr_name)
        psnr_val, ssim_val = evaluate(model, lr_path, hr_path)
        scores.append((psnr_val, ssim_val))

    avg_psnr = np.mean([s[0] for s in scores])
    avg_ssim = np.mean([s[1] for s in scores])
    print(f"Average PSNR: {avg_psnr:.2f}")
    print(f"Average SSIM: {avg_ssim:.4f}")