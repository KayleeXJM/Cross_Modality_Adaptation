
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader
from torchvision import datasets, transforms
from transformers import AutoModel, AutoConfig, AutoTokenizer
import mlflow
import os
import argparse
import json
import tempfile
from tqdm import tqdm
from typing import Dict, Optional, Tuple
import numpy as np
from peft import LoraConfig, get_peft_model


MODEL_NAME = "nickypro/tinyllama-110M"
PATCH_SIZE = 4
EMBED_DIM = AutoConfig.from_pretrained(MODEL_NAME).hidden_size
DEVICE = "cuda" if torch.cuda.is_available() else "mps" if torch.backends.mps.is_available() else "cpu"
NUM_CLASSES = 10
NUM_SAMPLES_ALIGNMENT = 50000

class ImageTokenizer(nn.Module):
    def __init__(self, patch_size=PATCH_SIZE, in_channels=3, embed_dim=EMBED_DIM):
        super().__init__()
        self.patch_size = patch_size
        self.proj = nn.Linear(in_channels * patch_size * patch_size, embed_dim)

    def forward(self, x):
        B, C, H, W = x.shape
        patches = x.unfold(2, self.patch_size, self.patch_size).unfold(3, self.patch_size, self.patch_size)
        patches = patches.contiguous().view(B, C, -1, self.patch_size * self.patch_size)
        patches = patches.permute(0, 2, 1, 3).reshape(B, -1, C * self.patch_size * self.patch_size)
        return self.proj(patches)


class ClassificationHead(nn.Module):
    def __init__(self, embed_dim, num_classes, dropout=0.1):
        super().__init__()
        self.dropout = nn.Dropout(dropout)
        self.classifier = nn.Linear(embed_dim, num_classes)
    
    def forward(self, x):
        x = self.dropout(x)
        return self.classifier(x)


class CrossModalModel(nn.Module):
    def __init__(self, base_model_name=MODEL_NAME, embed_dim=EMBED_DIM, num_classes=NUM_CLASSES, 
                 in_channels=3, finetune_mode='fpt', lora_rank=8, lora_alpha=16, lora_dropout=0.1):
        super().__init__()
        self.llm = AutoModel.from_pretrained(base_model_name)
        self.finetune_mode = finetune_mode
        
        if finetune_mode == 'full':
            for param in self.llm.parameters():
                param.requires_grad = True
            print("Full fine-tuning mode")
        elif finetune_mode == 'lora':
            for param in self.llm.parameters():
                param.requires_grad = False
            
            lora_config = LoraConfig(
                r=lora_rank,
                lora_alpha=lora_alpha,
                target_modules=["q_proj", "k_proj", "v_proj", "o_proj", "up_proj", "down_proj"],
                lora_dropout=lora_dropout,
                bias="none",
                task_type="FEATURE_EXTRACTION"
            )
            self.llm = get_peft_model(self.llm, lora_config)
            trainable_params, total_params = self.llm.get_nb_trainable_parameters()
            print(f"LoRA: rank={lora_rank}, alpha={lora_alpha}, trainable={trainable_params:,}/{total_params:,} ({100*trainable_params/total_params:.1f}%)")
        else:
            for param in self.llm.parameters():
                param.requires_grad = False
            for name, param in self.llm.named_parameters():
                if "norm" in name.lower():
                    param.requires_grad = True
            print("FPT mode")

        self.tokenizer = ImageTokenizer(embed_dim=embed_dim, in_channels=in_channels)
        self.num_patches = (32 // PATCH_SIZE) ** 2
        self.pos_embed = nn.Parameter(torch.zeros(1, self.num_patches, embed_dim))
        nn.init.trunc_normal_(self.pos_embed, std=0.02)
        self.cls_head = ClassificationHead(embed_dim, num_classes, dropout=0.1)

    def forward(self, x):
        tokens = self.tokenizer(x)
        tokens = tokens + self.pos_embed
        outputs = self.llm(inputs_embeds=tokens)
        pooled_hidden = outputs.last_hidden_state.mean(dim=1)
        return self.cls_head(pooled_hidden)


class MSEDistance(nn.Module):
    def forward(self, x, y):
        if x.dim() == 3:
            x = x.mean(dim=1)
        if y.dim() == 3:
            y = y.mean(dim=1)
        return F.mse_loss(x, y)


class MMDDistance(nn.Module):
    def __init__(self, kernel_mul=2.0, kernel_num=5):
        super().__init__()
        self.kernel_mul = kernel_mul
        self.kernel_num = kernel_num
    
    def gaussian_kernel(self, source, target):
        n_samples = int(source.size()[0]) + int(target.size()[0])
        total = torch.cat([source, target], dim=0)
        total0 = total.unsqueeze(0).expand(int(total.size(0)), int(total.size(0)), int(total.size(1)))
        total1 = total.unsqueeze(1).expand(int(total.size(0)), int(total.size(0)), int(total.size(1)))
        L2_distance = ((total0 - total1) ** 2).sum(2)
        bandwidth = torch.sum(L2_distance.data) / (n_samples ** 2 - n_samples)
        bandwidth /= self.kernel_mul ** (self.kernel_num // 2)
        bandwidth_list = [bandwidth * (self.kernel_mul ** i) for i in range(self.kernel_num)]
        kernel_val = [torch.exp(-L2_distance / bandwidth_temp) for bandwidth_temp in bandwidth_list]
        return sum(kernel_val)
    
    def forward(self, x, y):
        if x.dim() == 3:
            x = x.mean(dim=1)
        if y.dim() == 3:
            y = y.mean(dim=1)
        xx = self.gaussian_kernel(x, x)
        yy = self.gaussian_kernel(y, y)
        xy = self.gaussian_kernel(x, y)
        return xx.mean() + yy.mean() - 2 * xy.mean()


class OTDDDistance(nn.Module):

    def __init__(self, num_classes=10, reg=0.1, max_iter=100):
        super().__init__()
        self.num_classes = num_classes
        self.reg = reg  # Entropic regularization parameter
        self.max_iter = max_iter
    
    def sinkhorn(self, cost_matrix, p, q, reg=0.1, max_iter=100):
        n, m = cost_matrix.shape
        device = cost_matrix.device
        
        # Initialize dual variables
        u = torch.ones(n, device=device) / n
        v = torch.ones(m, device=device) / m
        
        # K = exp(-C / reg)
        K = torch.exp(-cost_matrix / reg)
        
        for _ in range(max_iter):
            u_prev = u.clone()
            u = p / (K @ v + 1e-10)
            v = q / (K.T @ u + 1e-10)
            
            if torch.norm(u - u_prev) < 1e-6:
                break

        P = torch.diag(u) @ K @ torch.diag(v)
        
        # Compute optimal transport cost
        ot_cost = (P * cost_matrix).sum()
        
        return P, ot_cost
    
    def forward(self, x, y, x_labels, y_labels):
        if x.dim() == 3:
            x = x.mean(dim=1)  # [B, embed_dim]
        if y.dim() == 3:
            y = y.mean(dim=1)  # [B, embed_dim]
        
        # Group embeddings by class and compute class centroids
        x_centroids = []
        y_centroids = []
        x_counts = []
        y_counts = []
        
        for c in range(self.num_classes):
            x_mask = (x_labels == c)
            y_mask = (y_labels == c)
            
            if x_mask.sum() > 0:
                x_centroids.append(x[x_mask].mean(dim=0))
                x_counts.append(x_mask.sum().float())
            else:
                # If no samples for this class, use zero vector with small count
                x_centroids.append(torch.zeros_like(x[0]))
                x_counts.append(torch.tensor(1e-6, device=x.device))
            
            if y_mask.sum() > 0:
                y_centroids.append(y[y_mask].mean(dim=0))
                y_counts.append(y_mask.sum().float())
            else:
                y_centroids.append(torch.zeros_like(y[0]))
                y_counts.append(torch.tensor(1e-6, device=y.device))
        
        x_centroids = torch.stack(x_centroids)  # [num_classes, embed_dim]
        y_centroids = torch.stack(y_centroids)  # [num_classes, embed_dim]
        x_counts = torch.stack(x_counts)  # [num_classes]
        y_counts = torch.stack(y_counts)  # [num_classes]
        
        x_probs = x_counts / (x_counts.sum() + 1e-10)
        y_probs = y_counts / (y_counts.sum() + 1e-10)
        
        x_expanded = x_centroids.unsqueeze(1)  # [num_classes, 1, embed_dim]
        y_expanded = y_centroids.unsqueeze(0)  # [1, num_classes, embed_dim]
        cost_matrix = ((x_expanded - y_expanded) ** 2).sum(dim=2)  # [num_classes, num_classes]
        
        P, ot_cost = self.sinkhorn(cost_matrix, x_probs, y_probs, 
                                   reg=self.reg, max_iter=self.max_iter)
        
        return ot_cost


def sample_text_embeddings(llm_model, num_samples=NUM_SAMPLES_ALIGNMENT, device=DEVICE, num_classes=NUM_CLASSES, infer_labels=True):

    llm_model.eval()
    embed_layer = llm_model.get_input_embeddings()
    vocab_size = embed_layer.weight.size(0)
    
    with torch.no_grad():
        token_ids = torch.randint(0, vocab_size, (num_samples, 64), device=device)
        token_embeds = embed_layer(token_ids)
        
        if infer_labels:

            embeds_flat = token_embeds.mean(dim=1).cpu().numpy()  # [num_samples, embed_dim]

            from sklearn.cluster import KMeans, MiniBatchKMeans
            
            if num_samples <= 10000:
                kmeans = KMeans(n_clusters=num_classes, random_state=42, n_init=10)
            else:
                kmeans = MiniBatchKMeans(n_clusters=num_classes, random_state=42, batch_size=10000, n_init=10)
            
            labels_np = kmeans.fit_predict(embeds_flat)
            labels = torch.from_numpy(labels_np).long().to(device)
        else:
            labels = torch.randint(0, num_classes, (num_samples,), device=device)
    
    return token_embeds, labels


def train_alignment_epoch(model, image_loader, target_embeddings, target_labels, optimizer, distance_metric, device=DEVICE, use_otdd=False):
    model.train()
    total_loss = 0
    num_batches = 0
    
    for images, image_labels in image_loader:
        images = images.to(device)
        image_labels = image_labels.to(device)
        batch_size = images.size(0)
        
        indices = torch.randint(0, target_embeddings.size(0), (batch_size,), device=device)
        target_embeds = target_embeddings[indices].to(device)
        target_labels_batch = target_labels[indices].to(device)
        
        optimizer.zero_grad()
        image_embeds = model.tokenizer(images)
        image_embeds = image_embeds + model.pos_embed
        
        if use_otdd:
            loss = distance_metric(image_embeds, target_embeds, image_labels, target_labels_batch)
        else:
            loss = distance_metric(image_embeds, target_embeds)
        
        loss.backward()
        optimizer.step()
        
        total_loss += loss.item()
        num_batches += 1
    
    return total_loss / num_batches if num_batches > 0 else 0


def train_task_epoch(model, dataloader, optimizer, criterion, device=DEVICE, max_grad_norm=1.0):
    model.train()
    total_loss, total_correct, total_samples = 0, 0, 0

    for imgs, labels in tqdm(dataloader, desc="Training", leave=False):
        imgs, labels = imgs.to(device), labels.to(device)
        optimizer.zero_grad()
        logits = model(imgs)
        loss = criterion(logits, labels)
        loss.backward()
        
        # Gradient clipping to prevent explosion (especially important for LoRA)
        if max_grad_norm > 0:
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_grad_norm)
        
        optimizer.step()

        total_loss += loss.item() * imgs.size(0)
        total_correct += (logits.argmax(1) == labels).sum().item()
        total_samples += imgs.size(0)

    return total_loss / total_samples, total_correct / total_samples


def evaluate(model, dataloader, criterion, device=DEVICE):
    model.eval()
    total_loss, total_correct, total_samples = 0, 0, 0

    with torch.no_grad():
        for imgs, labels in tqdm(dataloader, desc="Evaluating", leave=False):
            imgs, labels = imgs.to(device), labels.to(device)
            logits = model(imgs)
            loss = criterion(logits, labels)

            total_loss += loss.item() * imgs.size(0)
            total_correct += (logits.argmax(1) == labels).sum().item()
            total_samples += imgs.size(0)

    avg_loss = total_loss / total_samples
    avg_acc = total_correct / total_samples
    return avg_loss, avg_acc


def load_hyperparameters(config_path: Optional[str]) -> Dict:
    if config_path and os.path.exists(config_path):
        with open(config_path, 'r') as f:
            config = json.load(f)
        print(f"Loaded config from {config_path}")
        return config
    else:
        return {
            'lr': 5e-5,
            'batch_size': 32,
            'weight_decay': 1e-4,
            'optimizer': 'adamw',
            'scheduler': 'cosine',
            'dropout': 0.2,
            'alignment_epochs': 20,
            'alignment_lr': 1e-4,
            'alignment_batch_size': 16,
            'alignment_distance': 'mse',
            'task_epochs': 50,
            'finetune_mode': 'fpt',
            'lora_rank': 16,
            'lora_alpha': 16,
            'lora_dropout': 0.1,
            'max_grad_norm': 1.0
        }


def create_optimizer(model, config: Dict):
    params = filter(lambda p: p.requires_grad, model.parameters())
    if config['optimizer'] == 'adam':
        return torch.optim.Adam(params, lr=config['lr'], weight_decay=config['weight_decay'])
    else:
        return torch.optim.AdamW(params, lr=config['lr'], weight_decay=config['weight_decay'])


def create_scheduler(optimizer, config: Dict, num_epochs: int):
    if config['scheduler'] == 'cosine':
        return torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=num_epochs, eta_min=1e-6)
    elif config['scheduler'] == 'linear':
        return torch.optim.lr_scheduler.LinearLR(optimizer, start_factor=1.0, end_factor=0.1, total_iters=num_epochs)
    else:
        return None


def train_orca(
    do_alignment: bool = False,
    finetune_mode: str = 'fpt',
    config_path: Optional[str] = None,
    val_loader: Optional[DataLoader] = None,
    return_val_acc: bool = False,
    val_split: float = 0.0,
    alignment_only: bool = False,
    pretrained_embedder: Optional[str] = None,
    use_test_set: bool = False
) -> Tuple[float, Optional[float]]:
    config = load_hyperparameters(config_path)
    config['finetune_mode'] = finetune_mode
    
    if finetune_mode not in ['fpt', 'full', 'lora']:
        raise ValueError(f"finetune_mode must be 'fpt', 'full', or 'lora', got '{finetune_mode}'")
    
    train_transform = transforms.Compose([
        transforms.RandomHorizontalFlip(p=0.5),
        transforms.RandomCrop(32, padding=4),
        transforms.RandomRotation(20),
        transforms.ToTensor(),
        transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010))
    ])
    
    test_transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010))
    ])
    
    if val_loader is None and val_split > 0:
        from torch.utils.data import Subset
        full_trainset = datasets.CIFAR10(root="./data", train=True, download=True, transform=train_transform)
        dataset_size = len(full_trainset)
        indices = list(range(dataset_size))
        np.random.shuffle(indices)
        split = int(np.floor(val_split * dataset_size))
        val_indices = indices[:split]
        train_indices = indices[split:]
        
        trainset = Subset(full_trainset, train_indices)
        
        val_transform = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010))
        ])
        val_dataset = datasets.CIFAR10(root="./data", train=True, download=True, transform=val_transform)
        val_subset = Subset(val_dataset, val_indices)
        val_loader = DataLoader(val_subset, batch_size=config['batch_size'], shuffle=False)
    else:
        trainset = datasets.CIFAR10(root="./data", train=True, download=True, transform=train_transform)
    
    test_loader = None
    if use_test_set:
        testset = datasets.CIFAR10(root="./data", train=False, download=True, transform=test_transform)
        test_loader = DataLoader(testset, batch_size=config['batch_size'], shuffle=False)
    elif val_loader is None and val_split == 0.0:
        print("WARNING: No validation set provided and val_split=0.0.")
    
    train_loader = DataLoader(trainset, batch_size=config['batch_size'], shuffle=True)
    
    lora_rank = config.get('lora_rank', 8)
    lora_alpha = config.get('lora_alpha', 16)
    lora_dropout = config.get('lora_dropout', 0.1)
    
    model = CrossModalModel(
        base_model_name=MODEL_NAME,
        embed_dim=EMBED_DIM,
        num_classes=NUM_CLASSES,
        in_channels=3,
        finetune_mode=finetune_mode,
        lora_rank=lora_rank,
        lora_alpha=lora_alpha,
        lora_dropout=lora_dropout
    ).to(DEVICE)
    
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    total_params = sum(p.numel() for p in model.parameters())
    print(f"Trainable: {trainable_params:,}/{total_params:,} ({100*trainable_params/total_params:.1f}%)")
    
    model.cls_head.dropout.p = config.get('dropout', 0.1)
    
    # Load pretrained embedder if provided
    if pretrained_embedder is not None:
        print(f"\nLoading pretrained embedder from {pretrained_embedder}")
        tokenizer_path = os.path.join(pretrained_embedder, 'tokenizer.pth')
        pos_embed_path = os.path.join(pretrained_embedder, 'pos_embed.pth')
        
        if not os.path.exists(tokenizer_path) or not os.path.exists(pos_embed_path):
            raise FileNotFoundError(f"Pretrained embedder files not found in {pretrained_embedder}")
        
        model.tokenizer.load_state_dict(torch.load(tokenizer_path, map_location=DEVICE))
        model.pos_embed.data = torch.load(pos_embed_path, map_location=DEVICE)
        print("Pretrained embedder loaded successfully")
        do_alignment = False  # Skip alignment if loading pretrained
    
    if do_alignment:
        print("\nAlignment stage")
        
        llm_model = AutoModel.from_pretrained(MODEL_NAME).to(DEVICE)
        for param in llm_model.parameters():
            param.requires_grad = False
        
        print("Sampling text embeddings...")
        target_embeddings, target_labels = sample_text_embeddings(llm_model, num_samples=NUM_SAMPLES_ALIGNMENT, device=DEVICE, num_classes=NUM_CLASSES)
        
        align_transform = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010))
        ])
        align_dataset = datasets.CIFAR10(root="./data", train=True, download=True, transform=align_transform)
        align_loader = DataLoader(align_dataset, batch_size=config.get('alignment_batch_size', 16), shuffle=True)
        
        distance_metric_name = config.get('alignment_distance', 'mse')
        use_otdd = (distance_metric_name == 'otdd')
        
        if distance_metric_name == 'mse':
            distance_metric = MSEDistance().to(DEVICE)
        elif distance_metric_name == 'mmd':
            distance_metric = MMDDistance().to(DEVICE)
        elif distance_metric_name == 'otdd':
            distance_metric = OTDDDistance(num_classes=NUM_CLASSES).to(DEVICE)
        else:
            distance_metric = MSEDistance().to(DEVICE)
        
        align_params = list(model.tokenizer.parameters()) + [model.pos_embed]
        align_optimizer = torch.optim.Adam(align_params, lr=config.get('alignment_lr', 1e-4))
        
        alignment_epochs = config.get('alignment_epochs', 20)
        for epoch in range(alignment_epochs):
            loss = train_alignment_epoch(model, align_loader, target_embeddings, target_labels, align_optimizer, distance_metric, DEVICE, use_otdd=use_otdd)
            if (epoch + 1) % 5 == 0:
                print(f"Align epoch {epoch+1}/{alignment_epochs}: loss={loss:.6f}")
        
        print("Alignment done")
        del llm_model, target_embeddings
        
        # Save embedder if alignment_only flag is set
        if alignment_only:
            # Create embedder directory name based on config
            distance_metric_name = config.get('alignment_distance', 'mse')
            import time
            timestamp = time.strftime("%Y%m%d_%H%M%S")
            embedder_dir = f"pretrained_embedders/embedder_{distance_metric_name}_{config.get('alignment_epochs', 20)}epochs_{timestamp}"
            os.makedirs(embedder_dir, exist_ok=True)
            
            # Save tokenizer and positional embedding
            torch.save(model.tokenizer.state_dict(), os.path.join(embedder_dir, 'tokenizer.pth'))
            torch.save(model.pos_embed.data, os.path.join(embedder_dir, 'pos_embed.pth'))
            
            # Save config for reference
            embedder_config = {
                'alignment_distance': distance_metric_name,
                'alignment_epochs': config.get('alignment_epochs', 20),
                'alignment_lr': config.get('alignment_lr', 1e-4),
                'alignment_batch_size': config.get('alignment_batch_size', 16),
                'model_name': MODEL_NAME,
                'embed_dim': EMBED_DIM,
                'patch_size': PATCH_SIZE
            }
            with open(os.path.join(embedder_dir, 'config.json'), 'w') as f:
                json.dump(embedder_config, f, indent=2)
            
            print(f"\nEmbedder saved to {embedder_dir}")
            print("Exiting (alignment_only mode)")
            return 0.0, None
    
    print("\nTask training")
    
    optimizer = create_optimizer(model, config)
    scheduler = create_scheduler(optimizer, config, config.get('task_epochs', 50))
    criterion = nn.CrossEntropyLoss()
    
    mlflow.set_tracking_uri("file:./mlruns")
    model_short_name = MODEL_NAME.split("/")[-1].split("-")[0] if "/" in MODEL_NAME else MODEL_NAME
    align_suffix = "_WithAlign" if do_alignment else "_Baseline"
    if finetune_mode == 'full':
        finetune_suffix = "_FullFinetune"
    elif finetune_mode == 'lora':
        finetune_suffix = f"_LoRA_r{lora_rank}"
    else:
        finetune_suffix = "_FPT"
    experiment_name = f"{model_short_name}{align_suffix}{finetune_suffix}"
    mlflow.set_experiment(experiment_name)
    print(f"Experiment name: {experiment_name}")

    task_epochs = config.get('task_epochs', 50)
    best_val_acc = 0.0
    
    with mlflow.start_run():
        mlflow.log_params({
            'do_alignment': do_alignment,
            'finetune_mode': finetune_mode,
            **{f'hp_{k}': v for k, v in config.items()}
        })
        
        max_grad_norm = config.get('max_grad_norm', 1.0)  # Default to 1.0 for old configs
        for epoch in range(task_epochs):
            train_loss, train_acc = train_task_epoch(model, train_loader, optimizer, criterion, DEVICE, max_grad_norm)
            mlflow.log_metrics({
                'train_loss': train_loss,
                'train_acc': train_acc
            }, step=epoch)
            
            if val_loader is not None:
                val_loss, val_acc = evaluate(model, val_loader, criterion, DEVICE)
                mlflow.log_metrics({
                    'val_loss': val_loss,
                    'val_acc': val_acc
                }, step=epoch)
                if val_acc > best_val_acc:
                    best_val_acc = val_acc
                print(f"Epoch {epoch+1}/{task_epochs} - Train Loss: {train_loss:.4f}, Train Acc: {train_acc:.4f}, "
                      f"Val Loss: {val_loss:.4f}, Val Acc: {val_acc:.4f}, Best Val Acc: {best_val_acc:.4f}")
            elif test_loader is not None:
                # Only use test set if explicitly allowed (for final evaluation)
                test_loss, test_acc = evaluate(model, test_loader, criterion, DEVICE)
                mlflow.log_metrics({
                    'test_loss': test_loss,
                    'test_acc': test_acc
                }, step=epoch)
                print(f"Epoch {epoch+1}/{task_epochs} - Train Loss: {train_loss:.4f}, Train Acc: {train_acc:.4f}, "
                      f"Test Loss: {test_loss:.4f}, Test Acc: {test_acc:.4f}")
            else:
                # No validation or test set available - only report training metrics
                print(f"Epoch {epoch+1}/{task_epochs} - Train Loss: {train_loss:.4f}, Train Acc: {train_acc:.4f}")
            
            if scheduler:
                scheduler.step()
                current_lr = scheduler.get_last_lr()[0] if hasattr(scheduler, 'get_last_lr') else None
                if current_lr:
                    mlflow.log_metric('learning_rate', current_lr, step=epoch)
        
        if val_loader is not None:
            final_loss, final_acc = evaluate(model, val_loader, criterion, DEVICE)
            mlflow.log_metrics({
                'final_val_loss': final_loss,
                'final_val_acc': final_acc,
                'best_val_acc': best_val_acc
            })
            if return_val_acc:
                return final_acc, best_val_acc
            return final_acc, None
        elif test_loader is not None:
            # Only use test set if explicitly allowed (for final evaluation)
            final_loss, final_acc = evaluate(model, test_loader, criterion, DEVICE)
            mlflow.log_metrics({
                'final_test_loss': final_loss,
                'final_test_acc': final_acc
            })
            return final_acc, None
        else:
            # No validation or test set available
            print("WARNING: No validation or test set available for final evaluation.")
            return 0.0, None


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--do_alignment', action='store_true',
                       help='Perform embedding alignment (Stage 2)')
    parser.add_argument('--finetune_mode', type=str, default='lora',
                       choices=['fpt', 'full', 'lora'],
                       help='Fine-tuning mode: fpt (frozen backbone + layer norms), full (all params), or lora (LoRA adapters)')
    parser.add_argument('--lora_rank', type=int, default=None,
                       help='LoRA rank (only used if finetune_mode=lora). Default: 8')
    parser.add_argument('--lora_alpha', type=int, default=None,
                       help='LoRA alpha scaling factor (only used if finetune_mode=lora). Default: 16')
    parser.add_argument('--lora_dropout', type=float, default=None,
                       help='LoRA dropout rate (only used if finetune_mode=lora). Default: 0.1')
    parser.add_argument('--config', type=str, default=None,
                       help='Path to JSON file with hyperparameters')
    parser.add_argument('--val_split', type=float, default=0.0,
                       help='Validation split ratio (0.0 = use test set)')
    parser.add_argument('--alignment_only', action='store_true',
                       help='Only train embedding alignment stage and save it, then exit')
    parser.add_argument('--pretrained_embedder', type=str, default=None,
                       help='Path to pretrained embedder directory (contains tokenizer.pth and pos_embed.pth)')
    parser.add_argument('--use_test_set', action='store_true',
                       help='Allow use of test set for evaluation (ONLY for final model evaluation, not for hyperparameter tuning)')
    args = parser.parse_args()
    
    finetune_mode = args.finetune_mode
    config_overrides = {}
    if args.lora_rank is not None:
        config_overrides['lora_rank'] = args.lora_rank
    if args.lora_alpha is not None:
        config_overrides['lora_alpha'] = args.lora_alpha
    if args.lora_dropout is not None:
        config_overrides['lora_dropout'] = args.lora_dropout
    
    config_path = args.config
    temp_config_path = None
    if config_overrides:
        if args.config and os.path.exists(args.config):
            with open(args.config, 'r') as f:
                config = json.load(f)
        else:
            config = load_hyperparameters(None)
        config.update(config_overrides)
        temp_config = tempfile.NamedTemporaryFile(mode='w', suffix='.json', delete=False)
        json.dump(config, temp_config, indent=2)
        temp_config.close()
        temp_config_path = temp_config.name
        config_path = temp_config_path
    
    # If alignment_only is set, ensure do_alignment is True
    if args.alignment_only:
        args.do_alignment = True
    
    # If pretrained_embedder is provided, don't do alignment
    if args.pretrained_embedder is not None:
        args.do_alignment = False
    
    final_acc, best_val_acc = train_orca(
        do_alignment=args.do_alignment,
        finetune_mode=finetune_mode,
        config_path=config_path,
        val_loader=None,
        return_val_acc=(args.val_split > 0),
        val_split=args.val_split,
        alignment_only=args.alignment_only,
        pretrained_embedder=args.pretrained_embedder,
        use_test_set=args.use_test_set
    )
    
    print(f"\nFinal accuracy: {final_acc:.4f}")
    if best_val_acc is not None:
        print(f"Best val accuracy: {best_val_acc:.4f}")
    
    if temp_config_path and os.path.exists(temp_config_path):
        os.unlink(temp_config_path)


if __name__ == "__main__":
    main()

