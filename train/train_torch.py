import sys
import os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from model.enhancer_torch import SRCNN
from div2k_loader import DIV2KDataset  # Custom loader
import os

def train_model(lr_dir, hr_dir, epochs=10, batch_size=8, save_path="srcnn.pt"):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = SRCNN().to(device)

    dataset = DIV2KDataset(lr_dir, hr_dir)
    loader = DataLoader(dataset, batch_size=batch_size, shuffle=True)

    criterion = nn.MSELoss()
    optimizer = optim.Adam(model.parameters(), lr=1e-4)

    for epoch in range(epochs):
        model.train()
        total_loss = 0
        for lr, hr in loader:
            lr, hr = lr.to(device), hr.to(device)
            output = model(lr)
            loss = criterion(output, hr)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            total_loss += loss.item()
        avg_loss = total_loss / len(loader)
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