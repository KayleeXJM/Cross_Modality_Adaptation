import torch
import torch.nn as nn
from torch.utils.data import DataLoader, Subset
from torchvision import datasets, transforms
import numpy as np
import random
import argparse
import json
import os
from typing import Dict, List, Tuple, Optional
import subprocess
import sys
from collections import defaultdict
from datetime import datetime

from train import train_orca, load_hyperparameters, DEVICE

RANDOM_SEED = 42
VAL_SPLIT = 0.2

MAX_RUNG = 4
MIN_RUNG = 0
REDUCTION_FACTOR = 2
MAX_TRIALS = 50


def get_hyperparameter_space(finetune_mode: str = 'fpt') -> Dict:
    if finetune_mode == 'full':
        return {
            'lr': [1e-5, 5e-5, 1e-4, 5e-4, 1e-3],
            'batch_size': [16, 32, 64],
            'weight_decay': [0, 1e-4, 1e-2],
            'optimizer': ['adamw'],
            'scheduler': ['cosine', 'linear'],
            'dropout': [0.0, 0.1, 0.2],
            'alignment_lr': [1e-5, 1e-4, 5e-4],
            'alignment_batch_size': [8, 16, 32],
            'max_grad_norm': [1.0]
        }
    elif finetune_mode == 'lora':
        return {
            'lr': [5e-5, 1e-4, 5e-4, 1e-3],
            'batch_size': [32, 64, 128],
            'weight_decay': [0, 1e-4, 1e-2],
            'optimizer': ['adamw'],
            'scheduler': ['cosine'],
            'dropout': [0.1, 0.2],
            'alignment_lr': [1e-5, 1e-4, 5e-4],
            'alignment_batch_size': [8, 16, 32],
            'lora_rank': [8, 16, 32],
            'lora_alpha': [8, 16, 32, 64],
            'lora_dropout': [0.0, 0.05, 0.1, 0.2],
            'max_grad_norm': [1.0]
        }
    else:
        return {
            'lr': [1e-4, 5e-4, 1e-3, 5e-3],
            'batch_size': [32, 64, 128],
            'weight_decay': [0, 1e-4, 1e-2],
            'optimizer': ['adam', 'adamw'],
            'scheduler': ['cosine', 'linear'],
            'dropout': [0.0, 0.1, 0.2],
            'alignment_lr': [1e-5, 1e-4, 5e-4],
            'alignment_batch_size': [8, 16, 32],
            'max_grad_norm': [1.0]
        }


def sample_hyperparameters(hyperparameter_space: Dict) -> Dict:
    config = {}
    for key, values in hyperparameter_space.items():
        config[key] = random.choice(values)
    return config


def get_epochs_for_rung(base_epochs: int, rung: int, max_rung: int) -> int:
    scale_factor = (rung + 1) / (max_rung + 1)
    return max(1, int(base_epochs * scale_factor))


def create_train_val_split(val_split=VAL_SPLIT, random_seed=RANDOM_SEED):
    torch.manual_seed(random_seed)
    np.random.seed(random_seed)
    random.seed(random_seed)
    
    trainset = datasets.CIFAR10(root="./data", train=True, download=True, 
                               transform=transforms.Compose([transforms.ToTensor()]))
    dataset_size = len(trainset)
    indices = list(range(dataset_size))
    np.random.shuffle(indices)
    
    split_size = int(np.floor(val_split * dataset_size))
    val_indices = indices[:split_size]
    train_indices = indices[split_size:]
    
    return train_indices, val_indices


def create_val_loader(val_indices, batch_size=64):
    val_transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010))
    ])
    val_dataset = datasets.CIFAR10(root="./data", train=True, download=True, transform=val_transform)
    val_subset = Subset(val_dataset, val_indices)
    return DataLoader(val_subset, batch_size=batch_size, shuffle=False)


class ASHAScheduler:
    def __init__(self, max_rung=MAX_RUNG, reduction_factor=REDUCTION_FACTOR, min_trials_per_rung=4):
        self.max_rung = max_rung
        self.reduction_factor = reduction_factor
        self.min_trials_per_rung = min_trials_per_rung
        self.rungs = defaultdict(list)
        self.pending_trials = {}
        self.completed_trials = {}
        self.trial_counter = 0
    
    def suggest_trial(self, hyperparameter_space: Dict) -> Optional[Tuple[int, Dict, int]]:
        for r in range(self.max_rung):
            completed_at_rung = [t for t in self.rungs[r] if t[3] == r]
            if len(completed_at_rung) >= self.min_trials_per_rung:
                completed_at_rung.sort(key=lambda x: x[2], reverse=True)
                top_k = max(1, len(completed_at_rung) // self.reduction_factor)
                
                for trial_id, config, result, original_rung in completed_at_rung[:top_k]:
                    if r + 1 <= self.max_rung:
                        self.pending_trials[trial_id] = (config, r + 1, 'promoted')
                        print(f"Promoting trial {trial_id}: rung {r} -> {r + 1} (acc={result:.4f})")
                
                self.rungs[r] = [t for t in self.rungs[r] if t[0] not in [t[0] for t in completed_at_rung[:top_k]]]
                break
        
        for trial_id, (config, rung, status) in list(self.pending_trials.items()):
            if status == 'promoted':
                self.pending_trials[trial_id] = (config, rung, 'running')
                return trial_id, config, rung
        
        config = sample_hyperparameters(hyperparameter_space)
        trial_id = self.trial_counter
        self.trial_counter += 1
        self.pending_trials[trial_id] = (config, 0, 'running')
        return trial_id, config, 0
    
    def report_result(self, trial_id: int, config: Dict, rung: int, result: float):
        self.rungs[rung].append((trial_id, config, result, rung))
        self.completed_trials[trial_id] = (config, rung, result)
        if trial_id in self.pending_trials:
            del self.pending_trials[trial_id]
    
    def get_best_config(self) -> Tuple[Optional[Dict], float]:
        if not self.completed_trials:
            return None, 0.0
        
        for r in range(self.max_rung, -1, -1):
            rung_trials = [t for t in self.rungs[r] if t[3] == r]
            if rung_trials:
                rung_trials.sort(key=lambda x: x[2], reverse=True)
                best_trial_id, best_config, best_result, _ = rung_trials[0]
                return best_config, best_result
        
        all_results = [(trial_id, config, result) for trial_id, (config, rung, result) in self.completed_trials.items()]
        all_results.sort(key=lambda x: x[2], reverse=True)
        _, best_config, best_result = all_results[0]
        return best_config, best_result
    
    def get_stats(self) -> Dict:
        stats = {}
        for r in range(self.max_rung + 1):
            completed = [t for t in self.rungs[r] if t[3] == r]
            stats[f'rung_{r}_completed'] = len(completed)
            if completed:
                stats[f'rung_{r}_best'] = max(t[2] for t in completed)
        stats['total_completed'] = len(self.completed_trials)
        stats['total_pending'] = len(self.pending_trials)
        return stats


def run_trial(
    trial_id: int,
    config: Dict,
    rung: int,
    max_rung: int,
    do_alignment: bool,
    finetune_mode: str,
    val_loader: DataLoader,
    temp_config_path: str,
    fixed_task_epochs: int = 50,
    fixed_alignment_epochs: int = 20,
    fixed_alignment_distance: str = 'mse'
) -> float:
    print(f"\nTrial {trial_id} (rung {rung}/{max_rung})")
    
    config['task_epochs'] = get_epochs_for_rung(fixed_task_epochs, rung, max_rung)
    config['alignment_epochs'] = fixed_alignment_epochs
    config['alignment_distance'] = fixed_alignment_distance
    config['finetune_mode'] = finetune_mode
    
    if 'lora_rank' not in config:
        config['lora_rank'] = 8
    if 'lora_alpha' not in config:
        config['lora_alpha'] = 16
    if 'lora_dropout' not in config:
        config['lora_dropout'] = 0.1
    
    with open(temp_config_path, 'w') as f:
        json.dump(config, f, indent=2)
    
    try:
        val_acc, _ = train_orca(
            do_alignment=do_alignment,
            finetune_mode=finetune_mode,
            config_path=temp_config_path,
            val_loader=val_loader,
            return_val_acc=True
        )
        
        print(f"Trial {trial_id} done: val_acc={val_acc:.4f}")
        return val_acc
    
    except Exception as e:
        print(f"Trial {trial_id} failed: {e}")
        return 0.0


def main():
    parser = argparse.ArgumentParser(description='ASHA Hyperparameter Tuning')
    parser.add_argument('--do_alignment', action='store_true',
                       help='Perform embedding alignment')
    parser.add_argument('--finetune_mode', type=str, default='fpt',
                       choices=['fpt', 'full', 'lora'],
                       help='Fine-tuning mode: fpt (frozen backbone + layer norms), full (all paramters), or lora (Low Rank)')
    parser.add_argument('--max_trials', type=int, default=MAX_TRIALS,
                       help='Maximum number of trials')
    parser.add_argument('--max_rung', type=int, default=MAX_RUNG,
                       help='Maximum rung level')
    parser.add_argument('--output', type=str, default='best_hyperparameters.json',
                       help='Output file for best hyperparameters')
    parser.add_argument('--val_split', type=float, default=VAL_SPLIT,
                       help='Validation split ratio')
    parser.add_argument('--resume', type=str, default=None,
                       help='Resume from progress file (e.g., best_hyperparameters_progress.json)')
    parser.add_argument('--task_epochs', type=int, default=25,
                       help='Fixed number of task epochs (not tuned, will be scaled by rung)')
    parser.add_argument('--alignment_epochs', type=int, default=15,
                       help='Fixed number of alignment epochs (not tuned)')
    parser.add_argument('--alignment_distance', type=str, default='mse',
                       choices=['mse', 'cosine', 'mmd', 'otdd'],
                       help='Fixed alignment distance metric (not tuned). Options: mse, cosine, mmd, otdd')
    args = parser.parse_args()
    
    finetune_mode = args.finetune_mode
    
    torch.manual_seed(RANDOM_SEED)
    np.random.seed(RANDOM_SEED)
    random.seed(RANDOM_SEED)
    
    print("ASHA tuning")
    print(f"Alignment: {args.do_alignment}, Mode: {finetune_mode}")
    print(f"Max trials: {args.max_trials}, Max rung: {args.max_rung}")
    
    print("\nCreating validation split...")
    train_indices, val_indices = create_train_val_split(val_split=args.val_split)
    val_loader = create_val_loader(val_indices, batch_size=64)
    print(f"Val set size: {len(val_indices)}")
    
    hyperparameter_space = get_hyperparameter_space(finetune_mode)
    scheduler = ASHAScheduler(max_rung=args.max_rung, reduction_factor=REDUCTION_FACTOR)
    temp_config_path = "temp_asha_config.json"
    progress_file = args.output.replace('.json', '_progress.json')
    summary_file = args.output.replace('.json', '_summary.json')
    
    if args.resume and os.path.exists(args.resume):
        print(f"Resuming from {args.resume}...")
        with open(args.resume, 'r') as f:
            progress_data = json.load(f)
        trial_results = progress_data['trial_results']
        best_overall_acc = progress_data['best_val_acc']
        trial_count = progress_data['trials_completed']
        start_time = datetime.fromisoformat(progress_data['start_time'])
        
        if 'config' in progress_data:
            if 'fixed_task_epochs' in progress_data['config']:
                if args.task_epochs == 25:
                    args.task_epochs = progress_data['config']['fixed_task_epochs']
            if 'fixed_alignment_epochs' in progress_data['config']:
                if args.alignment_epochs == 15:
                    args.alignment_epochs = progress_data['config'].get('fixed_alignment_epochs', 15)
            if 'fixed_alignment_distance' in progress_data['config']:
                if args.alignment_distance == 'mse':
                    args.alignment_distance = progress_data['config']['fixed_alignment_distance']
            if 'finetune_mode' in progress_data['config']:
                finetune_mode = progress_data['config']['finetune_mode']
            elif 'full_finetune' in progress_data['config']:
                finetune_mode = 'full' if progress_data['config']['full_finetune'] else 'fpt'
        
        scheduler = ASHAScheduler(max_rung=args.max_rung, reduction_factor=REDUCTION_FACTOR)
        for result in trial_results:
            scheduler.report_result(result['trial_id'], result['config'], result['rung'], result['val_acc'])
        scheduler.trial_counter = max([r['trial_id'] for r in trial_results]) + 1 if trial_results else 0
        
        print(f"Resumed: {trial_count} trials, best val_acc: {best_overall_acc:.4f}")
    else:
        print("\nStarting ASHA search")
        trial_count = 0
        best_overall_acc = 0.0
        trial_results = []
        start_time = datetime.now()
        
        progress_data = {
            'start_time': start_time.isoformat(),
        'config': {
            'do_alignment': args.do_alignment,
            'finetune_mode': finetune_mode,
            'max_trials': args.max_trials,
            'max_rung': args.max_rung,
            'reduction_factor': REDUCTION_FACTOR,
            'val_split': args.val_split,
            'fixed_task_epochs': args.task_epochs,
            'fixed_alignment_epochs': args.alignment_epochs,
            'fixed_alignment_distance': args.alignment_distance
        },
        'fixed_params': {
            'task_epochs': args.task_epochs,
            'alignment_epochs': args.alignment_epochs,
            'alignment_distance': args.alignment_distance
        },
        'trial_results': [],
            'best_val_acc': 0.0,
            'trials_completed': 0
        }
    
    while trial_count < args.max_trials:
        trial_suggestion = scheduler.suggest_trial(hyperparameter_space)
        if trial_suggestion is None:
            if len(scheduler.pending_trials) == 0:
                print("No more trials to run")
                break
            else:
                continue
        
        trial_id, config, rung = trial_suggestion
        
        print(f"\nRunning trial {trial_id} (rung {rung})...")
        val_acc = run_trial(
            trial_id=trial_id,
            config=config,
            rung=rung,
            max_rung=args.max_rung,
            do_alignment=args.do_alignment,
            finetune_mode=finetune_mode,
            val_loader=val_loader,
            temp_config_path=temp_config_path,
            fixed_task_epochs=args.task_epochs,
            fixed_alignment_epochs=args.alignment_epochs,
            fixed_alignment_distance=args.alignment_distance
        )
        
        scheduler.report_result(trial_id, config, rung, val_acc)
        trial_count += 1
        
        trial_result = {
            'trial_id': trial_id,
            'rung': rung,
            'val_acc': float(val_acc),
            'config': config,
            'timestamp': datetime.now().isoformat()
        }
        trial_results.append(trial_result)
        
        if val_acc > best_overall_acc:
            best_overall_acc = val_acc
            print(f"New best: {best_overall_acc:.4f}")
        
        progress_data['trial_results'] = trial_results
        progress_data['best_val_acc'] = float(best_overall_acc)
        progress_data['trials_completed'] = trial_count
        progress_data['last_update'] = datetime.now().isoformat()
        
        with open(progress_file, 'w') as f:
            json.dump(progress_data, f, indent=2)
        
        if trial_count % 5 == 0:
            best_config, best_result = scheduler.get_best_config()
            stats = scheduler.get_stats()
            print(f"\nProgress: {trial_count}/{args.max_trials} trials")
            print(f"Best: {best_result:.4f}, Stats: {stats}")
    
    end_time = datetime.now()
    duration = (end_time - start_time).total_seconds()
    
    rung_bests = {}
    for result in trial_results:
        rung = result['rung']
        if rung not in rung_bests or result['val_acc'] > rung_bests[rung]['val_acc']:
            rung_bests[rung] = result
    
    trial_results.sort(key=lambda x: x['val_acc'], reverse=True)
    
    summary = {
        'start_time': start_time.isoformat(),
        'end_time': end_time.isoformat(),
        'duration_seconds': duration,
        'config': {
            'do_alignment': args.do_alignment,
            'finetune_mode': finetune_mode,
            'max_trials': args.max_trials,
            'max_rung': args.max_rung,
            'reduction_factor': REDUCTION_FACTOR,
            'val_split': args.val_split,
            'fixed_task_epochs': args.task_epochs,
            'fixed_alignment_epochs': args.alignment_epochs,
            'fixed_alignment_distance': args.alignment_distance
        },
        'fixed_params': {
            'task_epochs': args.task_epochs,
            'alignment_epochs': args.alignment_epochs,
            'alignment_distance': args.alignment_distance
        },
        'results': {
            'total_trials': len(trial_results),
            'best_val_acc': float(best_overall_acc),
            'trials_by_rung': {str(r): len([t for t in trial_results if t['rung'] == r]) 
                              for r in range(args.max_rung + 1)},
            'best_per_rung': {
                str(rung): {
                    'trial_id': result['trial_id'],
                    'val_acc': float(result['val_acc']),
                    'config': result['config']
                }
                for rung, result in rung_bests.items()
            }
        },
        'top_10_configs': [
            {
                'rank': i + 1,
                'trial_id': t['trial_id'],
                'rung': t['rung'],
                'val_acc': float(t['val_acc']),
                'config': t['config']
            }
            for i, t in enumerate(trial_results[:10])
        ],
        'all_trials': trial_results
    }
    
    with open(summary_file, 'w') as f:
        json.dump(summary, f, indent=2)
    print(f"\nSummary saved to {summary_file}")
    
    best_config, best_result = scheduler.get_best_config()
    
    print("\nASHA tuning complete")
    print(f"Total trials: {trial_count}")
    print(f"Best val acc: {best_result:.4f}")
    print(f"Duration: {duration/60:.1f} minutes")
    print(f"\nBest hyperparameters:")
    for key, value in best_config.items():
        print(f"  {key}: {value}")
    print(f"\nFixed params:")
    print(f"  task_epochs: {args.task_epochs}")
    print(f"  alignment_epochs: {args.alignment_epochs}")
    print(f"  alignment_distance: {args.alignment_distance}")
    
    best_config_with_fixed = best_config.copy()
    best_config_with_fixed['task_epochs'] = args.task_epochs
    best_config_with_fixed['alignment_epochs'] = args.alignment_epochs
    best_config_with_fixed['alignment_distance'] = args.alignment_distance
    
    output_dict = {
        'best_val_acc': float(best_result),
        'hyperparameters': best_config_with_fixed,
        'do_alignment': args.do_alignment,
        'finetune_mode': finetune_mode,
        'num_trials': trial_count,
        'fixed_params': {
            'task_epochs': args.task_epochs,
            'alignment_epochs': args.alignment_epochs,
            'alignment_distance': args.alignment_distance
        },
        'timestamp': end_time.isoformat()
    }
    
    with open(args.output, 'w') as f:
        json.dump(output_dict, f, indent=2)
    
    print(f"\nBest hyperparameters saved to {args.output}")
    print(f"Progress log: {progress_file}")
    print(f"Summary: {summary_file}")
    
    if os.path.exists(temp_config_path):
        os.remove(temp_config_path)


if __name__ == "__main__":
    main()

