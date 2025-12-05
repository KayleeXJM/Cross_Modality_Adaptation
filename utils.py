import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader
from tqdm import tqdm
from typing import Dict, Optional
import os
import json
import numpy as np
from sklearn.cluster import KMeans, MiniBatchKMeans


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
            'alignment_epochs': 20,  # TODO: might need more epochs for better alignment
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


def sample_text_embeddings(llm_model, num_samples=50000, device=None, num_classes=10, infer_labels=True):
    if device is None:
        device = "cuda" if torch.cuda.is_available() else "mps" if torch.backends.mps.is_available() else "cpu"
    
    llm_model.eval()
    embed_layer = llm_model.get_input_embeddings()
    vocab_size = embed_layer.weight.size(0)
    
    with torch.no_grad():
        token_ids = torch.randint(0, vocab_size, (num_samples, 64), device=device)
        token_embeds = embed_layer(token_ids)
        
        if infer_labels:
            embeds_flat = token_embeds.mean(dim=1).cpu().numpy()
            
            # use minibatch kmeans for large datasets, regular kmeans is fine for smaller ones
            if num_samples <= 10000:
                kmeans = KMeans(n_clusters=num_classes, random_state=42, n_init=10)
            else:
                kmeans = MiniBatchKMeans(n_clusters=num_classes, random_state=42, batch_size=10000, n_init=10)
            
            labels_np = kmeans.fit_predict(embeds_flat)
            labels = torch.from_numpy(labels_np).long().to(device)
        else:
            # just random labels if not inferring
            labels = torch.randint(0, num_classes, (num_samples,), device=device)
    
    return token_embeds, labels


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
        # compute pairwise distances for all samples
        n_samples = int(source.size()[0]) + int(target.size()[0])
        total = torch.cat([source, target], dim=0)
        total0 = total.unsqueeze(0).expand(int(total.size(0)), int(total.size(0)), int(total.size(1)))
        total1 = total.unsqueeze(1).expand(int(total.size(0)), int(total.size(0)), int(total.size(1)))
        L2_distance = ((total0 - total1) ** 2).sum(2)
        # adaptive bandwidth based on median distance
        bandwidth = torch.sum(L2_distance.data) / (n_samples ** 2 - n_samples)
        bandwidth /= self.kernel_mul ** (self.kernel_num // 2)
        # multi-scale kernels
        bandwidth_list = [bandwidth * (self.kernel_mul ** i) for i in range(self.kernel_num)]
        kernel_val = [torch.exp(-L2_distance / bandwidth_temp) for bandwidth_temp in bandwidth_list]
        return sum(kernel_val)
    
    def forward(self, x, y):
        if x.dim() == 3:
            x = x.mean(dim=1)
        if y.dim() == 3:
            y = y.mean(dim=1)
        
        n_x = x.size(0)
        n_y = y.size(0)
        
        kernels = self.gaussian_kernel(x, y)
        
        XX = kernels[:n_x, :n_x] 
        YY = kernels[n_x:, n_x:] 
        XY = kernels[:n_x, n_x:] 
        YX = kernels[n_x:, :n_x]  
        loss = torch.mean(XX + YY - XY - YX)
        return loss


class OTDDDistance(nn.Module):
    def __init__(self, num_classes=10, reg=0.1, max_iter=100):
        super().__init__()
        self.num_classes = num_classes
        self.reg = reg  
        self.max_iter = max_iter
    
    def sinkhorn(self, cost_matrix, p, q, reg=0.1, max_iter=100):
        n, m = cost_matrix.shape
        device = cost_matrix.device
        
        u = torch.ones(n, device=device) / n
        v = torch.ones(m, device=device) / m
        
        K = torch.exp(-cost_matrix / reg)
        
        # iterative updates
        for _ in range(max_iter):
            u_prev = u.clone()
            u = p / (K @ v + 1e-10)
            v = q / (K.T @ u + 1e-10)
            
            if torch.norm(u - u_prev) < 1e-6:
                break  # converged

        P = torch.diag(u) @ K @ torch.diag(v)
        ot_cost = (P * cost_matrix).sum()
        
        return P, ot_cost
    
    def forward(self, x, y, x_labels, y_labels):
        if x.dim() == 3:
            x = x.mean(dim=1)
        if y.dim() == 3:
            y = y.mean(dim=1)
        
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
                # handle empty classes with small count to avoid division issues
                x_centroids.append(torch.zeros_like(x[0]))
                x_counts.append(torch.tensor(1e-6, device=x.device))
            
            if y_mask.sum() > 0:
                y_centroids.append(y[y_mask].mean(dim=0))
                y_counts.append(y_mask.sum().float())
            else:
                y_centroids.append(torch.zeros_like(y[0]))
                y_counts.append(torch.tensor(1e-6, device=y.device))
        
        x_centroids = torch.stack(x_centroids)
        y_centroids = torch.stack(y_centroids)
        x_counts = torch.stack(x_counts)
        y_counts = torch.stack(y_counts)
        
        x_probs = x_counts / (x_counts.sum() + 1e-10)
        y_probs = y_counts / (y_counts.sum() + 1e-10)
        
        x_expanded = x_centroids.unsqueeze(1)
        y_expanded = y_centroids.unsqueeze(0)
        cost_matrix = ((x_expanded - y_expanded) ** 2).sum(dim=2)
        
        P, ot_cost = self.sinkhorn(cost_matrix, x_probs, y_probs, 
                                   reg=self.reg, max_iter=self.max_iter)
        
        return ot_cost
