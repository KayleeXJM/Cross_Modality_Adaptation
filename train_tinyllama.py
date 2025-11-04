"""
train_baseline.py
Baseline implementation for Cross-Modal TinyLlama (Stage 1)
"""

import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from torchvision import datasets, transforms
from transformers import AutoModel, AutoTokenizer, AutoConfig
import mlflow
import os
from tqdm import tqdm

# ===================== Config =====================
MODEL_NAME = "nickypro/tinyllama-110M"
BATCH_SIZE = 16
EPOCHS = 20
LR = 1e-4
PATCH_SIZE = 4  # (32/4 = 8) yielding patch number 8x8 = 64 
EMBED_DIM = AutoConfig.from_pretrained(MODEL_NAME).hidden_size # 768
DEVICE = "cuda" if torch.cuda.is_available() else "mps" if torch.backends.mps.is_available() else "cpu"
NUM_CLASSES = 10

# ===================== Data Pipeline =====================
transform = transforms.Compose([transforms.ToTensor(),])

trainset = datasets.CIFAR10(root="./data", train=True, download=True, transform=transform)
testset = datasets.CIFAR10(root="./data", train=False, download=True, transform=transform)
train_loader = DataLoader(trainset, batch_size=BATCH_SIZE, shuffle=True)
test_loader = DataLoader(testset, batch_size=BATCH_SIZE, shuffle=False)

# Patchify: image 32x32 --> (num_patches, patch_dim)
class ImageTokenizer(nn.Module):
    def __init__(self, patch_size=PATCH_SIZE, in_channels=3, embed_dim=EMBED_DIM):
        super().__init__()
        self.patch_size = patch_size
        self.proj = nn.Linear(in_channels * patch_size * patch_size, embed_dim)

    def forward(self, x):
        B, C, H, W = x.shape # (B, C, H, W) -> (B, num_patches, patch_dim)
        patches = x.unfold(2, self.patch_size, self.patch_size).unfold(3, self.patch_size, self.patch_size)
        patches = patches.contiguous().view(B, C, -1, self.patch_size*self.patch_size)
        patches = patches.permute(0, 2, 1, 3).reshape(B, -1, C * self.patch_size * self.patch_size)
        return self.proj(patches)  # [B, num_patches, embed_dim]


# ======================== Model ========================
class CrossModalTinyLlama(nn.Module):
    def __init__(self, base_model_name=MODEL_NAME, embed_dim=EMBED_DIM, num_classes=NUM_CLASSES):
        super().__init__()
        self.llm = AutoModel.from_pretrained(base_model_name)
        for param in self.llm.parameters():
            param.requires_grad = False  # frozen backbone

        for name, param in self.llm.named_parameters():
            if "norm" in name.lower():
                param.requires_grad = True # unfreeze layer norm 

        # Patch Embedding
        self.tokenizer = ImageTokenizer(embed_dim=embed_dim)

        # Learnable positional embedding
        self.num_patches = (32 // 4) ** 2  # CIFAR-10: 32x32 image, patch=4 -> 8x8=64 patches
        self.pos_embed = nn.Parameter(torch.zeros(1, self.num_patches, embed_dim))
        nn.init.trunc_normal_(self.pos_embed, std=0.02)

        # Classification head
        self.cls_head = nn.Linear(embed_dim, num_classes)

        # Parameters finetuned: 
        # 1) input embedding, 
        # 2) layer norm, 
        # 3) positional embedding, 
        # 4) output layer

    def forward(self, x):
        tokens = self.tokenizer(x) 
        tokens = tokens + self.pos_embed

        outputs = self.llm(inputs_embeds=tokens)
        last_hidden = outputs.last_hidden_state.mean(dim=1)
        return self.cls_head(last_hidden)


# ===================== Training Loop =====================
def train_one_epoch(model, dataloader, optimizer, criterion):
    model.train()
    total_loss, total_correct, total_samples = 0, 0, 0

    for imgs, labels in tqdm(dataloader, desc="Training", leave=False):
        imgs, labels = imgs.to(DEVICE), labels.to(DEVICE)
        optimizer.zero_grad()
        logits = model(imgs)
        loss = criterion(logits, labels)
        loss.backward()
        optimizer.step()

        total_loss += loss.item() * imgs.size(0)
        total_correct += (logits.argmax(1) == labels).sum().item()
        total_samples += imgs.size(0)

    return total_loss / total_samples, total_correct / total_samples

# ===================== Testing =====================
def evaluate(model, dataloader, criterion):
    model.eval()
    total_loss, total_correct, total_samples = 0, 0, 0

    with torch.no_grad():
        for imgs, labels in tqdm(dataloader, desc="Evaluating", leave=False):
            imgs, labels = imgs.to(DEVICE), labels.to(DEVICE)
            logits = model(imgs)
            loss = criterion(logits, labels)

            total_loss += loss.item() * imgs.size(0)
            total_correct += (logits.argmax(1) == labels).sum().item()
            total_samples += imgs.size(0)

    avg_loss = total_loss / total_samples
    avg_acc = total_correct / total_samples
    return avg_loss, avg_acc


# ===================== Main =====================
def main():
    mlflow.set_tracking_uri("file:./mlruns")
    mlflow.set_experiment("CrossModal_TinyLlama_Baseline")

    model = CrossModalTinyLlama().to(DEVICE)
    optimizer = torch.optim.Adam(filter(lambda p: p.requires_grad, model.parameters()), lr=LR)
    criterion = nn.CrossEntropyLoss()
    
    print("====== Trainable Parameters =======")
    for name, p in model.named_parameters():
        if p.requires_grad:
            print(name, p.numel())
    print("====================================")

    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    total_params = sum(p.numel() for p in model.parameters())
    print(f"Number of Trainable params: {trainable_params} / {total_params} = {trainable_params/total_params:.4f}")
    
    with mlflow.start_run():
        start_epoch = 0
        latest_ckpt = "latest_checkpoint.pt"

        if os.path.exists(latest_ckpt):
            ckpt = torch.load(latest_ckpt, map_location=DEVICE)
            model.load_state_dict(ckpt["model_state"])
            optimizer.load_state_dict(ckpt["optimizer_state"])
            start_epoch = ckpt["epoch"] + 1
            print(f"Resumed at epoch {start_epoch}")

        for epoch in range(start_epoch, EPOCHS):
            # ---------- Training ----------
            loss, acc = train_one_epoch(model, train_loader, optimizer, criterion)
            mlflow.log_metric("train_loss", loss, step=epoch)
            mlflow.log_metric("train_acc", acc, step=epoch)
            print(f"[Epoch {epoch+1}] Train Loss={loss:.4f}, Train Acc={acc:.4f}")

            # ---------- Evaluation ----------
            test_loss, test_acc = evaluate(model, test_loader, criterion)
            mlflow.log_metric("test_loss", test_loss, step=epoch)
            mlflow.log_metric("test_acc", test_acc, step=epoch)
            print(f"[Epoch {epoch+1}] Test Loss={test_loss:.4f}, Test Acc={test_acc:.4f}")

            # ---------- Save checkpoint ----------
            ckpt = {
                "epoch": epoch,
                "model_state": model.state_dict(),
                "optimizer_state": optimizer.state_dict()
            }
            torch.save(ckpt, latest_ckpt)

            if (epoch + 1) % 5 == 0: 
                torch.save(ckpt, f"checkpoint_epoch_{epoch+1}.pt")

        # ---------- Save final model ----------
        torch.save(model.state_dict(), "baseline_model.pt")
        mlflow.log_artifact("baseline_model.pt")


if __name__ == "__main__":
    main()
    print("Training complete!")
