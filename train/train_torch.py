import sys
import os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))
import torch
import torch.nn as nn
import torch.optim as optim
from PIL import Image
import torchvision.transforms as T

from model.enhancer_torch import SRCNN

def preprocess_pair(lr_path, hr_path):
    # Load images and extract Y channel
    img_lr = Image.open(lr_path).convert("YCbCr").split()[0]
    img_hr = Image.open(hr_path).convert("YCbCr").split()[0]

    # Convert to tensors
    tensor_lr = T.ToTensor()(img_lr)
    tensor_hr = T.ToTensor()(img_hr)
    return tensor_lr, tensor_hr

def train_model(lr_dir, hr_dir, epochs=10, batch_size=8, save_path="srcnn.pt"):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = SRCNN().to(device)

    # Prepare image pairs
    lr_images = sorted(os.listdir(lr_dir))
    hr_images = sorted(os.listdir(hr_dir))
    image_pairs = [(os.path.join(lr_dir, l), os.path.join(hr_dir, h)) for l, h in zip(lr_images, hr_images)]

    criterion = nn.L1Loss()
    optimizer = optim.Adam(model.parameters(), lr=1e-4)

    for epoch in range(epochs):
        model.train()
        total_loss = 0

        for i in range(0, len(image_pairs), batch_size):
            batch = image_pairs[i:i+batch_size]
            inputs, targets = [], []

            for lr_path, hr_path in batch:
                lr, hr = preprocess_pair(lr_path, hr_path)
                inputs.append(lr)
                targets.append(hr)

            input_tensor = torch.stack(inputs).to(device)  # [B, 1, H, W]
            target_tensor = torch.stack(targets).to(device)

            optimizer.zero_grad()
            output = model(input_tensor)
            loss = criterion(output, target_tensor)
            loss.backward()
            optimizer.step()
            total_loss += loss.item()

        avg_loss = total_loss / (len(image_pairs) // batch_size)
        print(f"Epoch {epoch+1}/{epochs} - Avg Loss: {avg_loss:.4f}")
        torch.save(model.state_dict(), f"checkpoints/srcnn_epoch{epoch+1}.pt")

    torch.save(model.state_dict(), save_path)
    print(f"Model saved to {save_path}")

if __name__ == "__main__":
    os.makedirs("checkpoints", exist_ok=True)
    train_model(
        lr_dir="DIV2K/DIV2K_train_LR_bicubic/X2",
        hr_dir="DIV2K/DIV2K_train_HR"
    )