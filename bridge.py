#!/usr/bin/env python3
"""
Bridge script to convert SparseGPT pruned models to circuit_sparsity visualization format.

Usage:
    python bridge.py --sparsegpt-checkpoint ./sparse_opt125m.pt \
                     --model-name facebook/opt-125m \
                     --output-dir ./circuit_viz \
                     --tasks quote_completion,ioi

Requirements:
    pip install torch transformers datasets tqdm pickle5
"""

import argparse
import os
from pathlib import Path
from typing import Dict, List, Tuple, Any
import json

import torch
import torch.nn as nn
from transformers import AutoModelForCausalLM, AutoTokenizer
from datasets import load_dataset
from tqdm import tqdm
import numpy as np


POS_KEYWORDS = [
    "embed_positions",
    "wpe",
    "position_embedding",
    "position_embeddings",
    "positional_embedding",
    "pos_emb",
]


def _clone_to_cpu(tensor: torch.Tensor | None) -> torch.Tensor | None:
    if tensor is None:
        return None
    return tensor.detach().to("cpu", dtype=torch.float16)


def extract_embedding_weights(model) -> Dict[str, torch.Tensor | None]:
    """Extract token and positional embedding weights from the model."""
    token_emb = None
    input_embeddings = model.get_input_embeddings()
    if input_embeddings is not None and hasattr(input_embeddings, "weight"):
        token_emb = _clone_to_cpu(input_embeddings.weight)

    pos_emb = None
    for name, param in model.named_parameters():
        lname = name.lower()
        if any(keyword in lname for keyword in POS_KEYWORDS):
            pos_emb = _clone_to_cpu(param)
            break

    return {
        "token_embeddings": token_emb,
        "positional_embeddings": pos_emb,
    }


def load_prune_metrics(checkpoint_path: str | None):
    if not checkpoint_path:
        return None
    base = Path(checkpoint_path)
    if base.is_file():
        base = base.parent
    metrics_file = base / "prune_metrics.json"
    if metrics_file.exists():
        try:
            with open(metrics_file, "r") as fh:
                return json.load(fh)
        except json.JSONDecodeError:
            return None
    return None


class CircuitDataExtractor:
    """Extract circuit visualization data from sparse models."""
    
    def __init__(self, model, tokenizer, device='cuda'):
        self.model = model
        self.tokenizer = tokenizer
        self.device = device
        self.model.to(device)
        self.model.eval()
        
    def extract_sparsity_mask(self) -> Dict[str, torch.Tensor]:
        """Extract binary masks showing which weights are non-zero."""
        masks = {}
        for name, param in self.model.named_parameters():
            if 'weight' in name and param.dim() >= 2:
                masks[name] = (param.abs() > 1e-8).float()
        return masks
    
    def compute_weight_importance(self, masks: Dict[str, torch.Tensor]) -> Dict[str, torch.Tensor]:
        """Compute importance scores for each weight (magnitude-based)."""
        importance = {}
        for name, param in self.model.named_parameters():
            if name in masks:
                importance[name] = param.abs() * masks[name]
        return importance
    
    def run_task_evaluation(self, task_name: str, num_samples: int = 100) -> Dict[str, Any]:
        """Run model on specific task and collect activations."""
        
        if task_name == 'quote_completion':
            return self._eval_quote_completion(num_samples)
        elif task_name == 'ioi':
            return self._eval_ioi(num_samples)
        else:
            raise ValueError(f"Unknown task: {task_name}")
    
    def _eval_quote_completion(self, num_samples: int) -> Dict[str, Any]:
        """Evaluate on Python quote completion task."""
        samples = []
        activations = []
        
        # Generate synthetic Python quote examples
        test_cases = [
            ('x = "hello', '"'),
            ("x = 'world", "'"),
            ('print("test', '"'),
            ("name = 'alice", "'"),
        ]
        
        for i in tqdm(range(num_samples), desc="Quote completion"):
            prefix, expected = test_cases[i % len(test_cases)]
            
            inputs = self.tokenizer(prefix, return_tensors='pt').to(self.device)
            
            # Forward pass with hooks to capture activations
            layer_acts = {}
            hooks = []
            
            def make_hook(layer_name):
                def hook(module, input, output):
                    layer_acts[layer_name] = output[0].detach().cpu() if isinstance(output, tuple) else output.detach().cpu()
                return hook
            
            # Register hooks on attention and MLP layers
            for name, module in self.model.named_modules():
                if 'attn' in name or 'mlp' in name or 'fc' in name:
                    hooks.append(module.register_forward_hook(make_hook(name)))
            
            with torch.no_grad():
                outputs = self.model(**inputs, output_hidden_states=True)
                logits = outputs.logits[0, -1, :]
                
                # Get prediction
                pred_token_id = logits.argmax().item()
                pred_token = self.tokenizer.decode([pred_token_id])
            
            # Remove hooks
            for hook in hooks:
                hook.remove()
            
            samples.append({
                'input': prefix,
                'expected': expected,
                'predicted': pred_token,
                'correct': expected in pred_token,
                'logits': logits.cpu().numpy(),
            })
            
            activations.append(layer_acts)
        
        return {
            'samples': samples,
            'activations': activations,
            'accuracy': sum(s['correct'] for s in samples) / len(samples),
        }
    
    def _eval_ioi(self, num_samples: int) -> Dict[str, Any]:
        """Evaluate on Indirect Object Identification task."""
        samples = []
        activations = []
        
        # IOI template: "When John and Mary went to the store, Mary gave a bottle to"
        templates = [
            ("When {A} and {B} went to the store, {B} gave a bottle to", "{A}"),
            ("After {A} and {B} left the party, {B} told a secret to", "{A}"),
            ("{A} and {B} were friends. Then {B} gave a gift to", "{A}"),
        ]
        
        names = ['John', 'Mary', 'Alice', 'Bob', 'Charlie', 'Diana']
        
        for i in tqdm(range(num_samples), desc="IOI task"):
            template, answer_template = templates[i % len(templates)]
            name_a = names[(i * 2) % len(names)]
            name_b = names[(i * 2 + 1) % len(names)]
            
            text = template.format(A=name_a, B=name_b)
            expected = answer_template.format(A=name_a)
            
            inputs = self.tokenizer(text, return_tensors='pt').to(self.device)
            
            layer_acts = {}
            hooks = []
            
            def make_hook(layer_name):
                def hook(module, input, output):
                    layer_acts[layer_name] = output[0].detach().cpu() if isinstance(output, tuple) else output.detach().cpu()
                return hook
            
            for name, module in self.model.named_modules():
                if 'attn' in name or 'mlp' in name or 'fc' in name:
                    hooks.append(module.register_forward_hook(make_hook(name)))
            
            with torch.no_grad():
                outputs = self.model(**inputs, output_hidden_states=True)
                logits = outputs.logits[0, -1, :]
                pred_token_id = logits.argmax().item()
                pred_token = self.tokenizer.decode([pred_token_id])
            
            for hook in hooks:
                hook.remove()
            
            samples.append({
                'input': text,
                'expected': expected,
                'predicted': pred_token,
                'correct': name_a.lower() in pred_token.lower(),
                'logits': logits.cpu().numpy(),
            })
            
            activations.append(layer_acts)
        
        return {
            'samples': samples,
            'activations': activations,
            'accuracy': sum(s['correct'] for s in samples) / len(samples),
        }
    
    def identify_circuits(self, task_results: Dict[str, Any], k: int = 10) -> Dict[str, Any]:
        """Identify top-k most important circuit components for a task."""
        
        # Aggregate activation magnitudes across samples
        all_activations = task_results['activations']
        
        circuit_importance = {}
        for layer_name in all_activations[0].keys():
            # Average activation magnitude across all samples
            avg_activation = torch.stack([
                acts[layer_name].abs().mean() 
                for acts in all_activations
            ]).mean().item()
            
            circuit_importance[layer_name] = avg_activation
        
        # Get top-k components
        sorted_components = sorted(
            circuit_importance.items(), 
            key=lambda x: x[1], 
            reverse=True
        )[:k]
        
        return {
            'top_k_components': sorted_components,
            'all_importance': circuit_importance,
        }


def summarize_layer_samples(task_results: Dict[str, Any], top_k: int = 5) -> Dict[str, Dict[str, List[Dict[str, Any]]]]:
    samples = task_results.get("samples", [])
    activations = task_results.get("activations", [])
    if not samples or not activations:
        return {}

    layer_scores: Dict[str, List[Dict[str, Any]]] = {}
    for sample, layer_act in zip(samples, activations, strict=True):
        for layer_name, act in layer_act.items():
            score = float(act.abs().mean().item())
            entry = {
                "input": sample.get("input"),
                "expected": sample.get("expected"),
                "predicted": sample.get("predicted"),
                "correct": sample.get("correct"),
                "activation": score,
            }
            layer_scores.setdefault(layer_name, []).append(entry)

    summary: Dict[str, Dict[str, List[Dict[str, Any]]]] = {}
    for layer_name, entries in layer_scores.items():
        entries_sorted = sorted(entries, key=lambda x: x["activation"], reverse=True)
        summary[layer_name] = {
            "top": entries_sorted[:top_k],
            "bottom": list(reversed(entries_sorted[-top_k:])) if len(entries_sorted) >= top_k else list(reversed(entries_sorted)),
        }
    return summary


def load_sparsegpt_checkpoint(checkpoint_path: str, model_name: str, device='cuda'):
    """Load a SparseGPT pruned checkpoint."""
    
    # Load base model
    model = AutoModelForCausalLM.from_pretrained(model_name)
    
    # Load sparse weights
    if checkpoint_path and os.path.exists(checkpoint_path):
        print(f"Loading sparse checkpoint from {checkpoint_path}")
        if os.path.isdir(checkpoint_path):
            # If checkpoint is a directory produced by save_pretrained, reload from it.
            model = AutoModelForCausalLM.from_pretrained(checkpoint_path)
        else:
            checkpoint = torch.load(checkpoint_path, map_location='cpu')
            
            # Handle different checkpoint formats
            if 'model' in checkpoint:
                state_dict = checkpoint['model']
            elif 'state_dict' in checkpoint:
                state_dict = checkpoint['state_dict']
            else:
                state_dict = checkpoint
            
            model.load_state_dict(state_dict, strict=False)
    
    return model


def create_viz_data(
    model,
    tokenizer,
    tasks: List[str],
    output_dir: Path,
    experiment_name: str,
    model_name: str,
    k_values: List[int] = [5, 10, 20, 50],
    num_samples: int = 100,
    device: str = "cuda",
    prune_metrics: dict | None = None,
):
    """Generate visualization data for circuit_sparsity."""
    
    extractor = CircuitDataExtractor(model, tokenizer, device=device)
    
    # Extract sparsity information
    print("Extracting sparsity masks...")
    masks = extractor.extract_sparsity_mask()
    importance = extractor.compute_weight_importance(masks)
    
    # Compute overall sparsity
    total_params = sum(p.numel() for p in model.parameters())
    nonzero_params = sum((p.abs() > 1e-8).sum().item() for p in model.parameters())
    sparsity = 1.0 - (nonzero_params / total_params)
    
    print(f"Model sparsity: {sparsity:.2%} ({nonzero_params}/{total_params} non-zero)")
    
    embedding_payload = extract_embedding_weights(model)
    if tokenizer is not None:
        embedding_payload["tokenizer_name"] = getattr(tokenizer, "name_or_path", None)
        embedding_payload["tokenizer_vocab_size"] = getattr(tokenizer, "vocab_size", None)

    # Process each task
    for task in tasks:
        print(f"\nEvaluating task: {task}")
        
        task_results = extractor.run_task_evaluation(task, num_samples)
        
        print(f"Task accuracy: {task_results['accuracy']:.2%}")
        
        model_dir = (
            output_dir
            / "viz"
            / model_name.replace("/", "_")
            / task
            / experiment_name
        )

        layer_sample_summary = summarize_layer_samples(task_results)
        # Generate circuit data for different k values
        for k in k_values:
            print(f"  Identifying top-{k} circuit components...")
            
            circuits = extractor.identify_circuits(task_results, k)
            
            # Prepare viz_data.pkl
            viz_data = {
                'model_name': model_name,
                'experiment': experiment_name,
                'task': task,
                'k': k,
                'sparsity': sparsity,
                'masks': {name: mask.cpu().numpy() for name, mask in masks.items()},
                'importance': {name: imp.detach().cpu().numpy() for name, imp in importance.items()},
                'circuits': circuits,
                'task_results': task_results,
                'bridge_samples': layer_sample_summary,
                'metadata': {
                    'num_samples': num_samples,
                    'total_params': total_params,
                    'nonzero_params': nonzero_params,
                }
            }
            if prune_metrics is not None:
                viz_data['prune_metrics'] = prune_metrics
            if embedding_payload.get("token_embeddings") is not None:
                viz_data['embedding_data'] = embedding_payload

            # Save in circuit_sparsity expected structure
            viz_dir = model_dir / str(k)
            viz_dir.mkdir(parents=True, exist_ok=True)
            
            viz_path = viz_dir / 'viz_data.pt'
            torch.save(viz_data, viz_path)
            
            print(f"    Saved to {viz_path}")
            
            # Also save human-readable summary
            summary_path = viz_dir / 'summary.json'
            with open(summary_path, 'w') as f:
                json.dump({
                    'accuracy': task_results['accuracy'],
                    'sparsity': sparsity,
                    'top_components': circuits['top_k_components'][:5],
                    'k': k,
                }, f, indent=2)
    
    print(f"\n✓ All visualization data saved to {output_dir}")
    print(f"\nTo visualize, run:")
    print(f"  cd circuit_sparsity")
    print(f"  streamlit run app.py -- --viz-dir {output_dir / 'viz'}")


def main():
    parser = argparse.ArgumentParser(description='Bridge SparseGPT to circuit_sparsity')
    parser.add_argument('--sparsegpt-checkpoint', type=str, 
                        help='Path to SparseGPT pruned checkpoint')
    parser.add_argument('--model-name', type=str, default='facebook/opt-125m',
                        help='Base model name (e.g., facebook/opt-125m)')
    parser.add_argument('--output-dir', type=str, default='./circuit_viz',
                        help='Output directory for visualization data')
    parser.add_argument('--tasks', type=str, default='quote_completion,ioi',
                        help='Comma-separated list of tasks')
    parser.add_argument('--experiment-name', type=str, default='sparsegpt_integration',
                        help='Experiment name for organization')
    parser.add_argument('--k-values', type=str, default='5,10,20,50',
                        help='Comma-separated k values for circuit extraction')
    parser.add_argument('--num-samples', type=int, default=100,
                        help='Number of samples per task')
    parser.add_argument('--device', type=str, default='cuda',
                        help='Device to use (cuda/cpu)')
    
    args = parser.parse_args()
    
    # Parse arguments
    tasks = [t.strip() for t in args.tasks.split(',')]
    k_values = [int(k.strip()) for k in args.k_values.split(',')]
    output_dir = Path(args.output_dir)
    
    print("="*60)
    print("SparseGPT → Circuit Sparsity Bridge")
    print("="*60)
    print(f"Model: {args.model_name}")
    print(f"Checkpoint: {args.sparsegpt_checkpoint or 'None (using dense model)'}")
    print(f"Tasks: {', '.join(tasks)}")
    print(f"Output: {output_dir}")
    print("="*60)
    
    # Load model
    print("\nLoading model...")
    model = load_sparsegpt_checkpoint(
        args.sparsegpt_checkpoint, 
        args.model_name, 
        args.device
    )
    tokenizer = AutoTokenizer.from_pretrained(args.model_name)
    prune_metrics = load_prune_metrics(args.sparsegpt_checkpoint)
    
    # Generate visualization data
    create_viz_data(
        model=model,
        tokenizer=tokenizer,
        tasks=tasks,
        output_dir=output_dir,
        experiment_name=args.experiment_name,
        model_name=args.model_name,
        k_values=k_values,
        num_samples=args.num_samples,
        device=args.device,
        prune_metrics=prune_metrics,
    )
    
    print("\n✓ Pipeline complete!")


if __name__ == '__main__':
    main()
