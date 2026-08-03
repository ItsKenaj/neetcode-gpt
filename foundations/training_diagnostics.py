import torch
import torch.nn as nn
from typing import List, Dict


class Solution:

    def compute_activation_stats(self, model: nn.Module, x: torch.Tensor) -> List[Dict[str, float]]:
        # Forward pass through model layer by layer
        # After each nn.Linear, record: mean, std, dead_fraction
        # Run with torch.no_grad(). Round to 4 decimals.
        stats = []
        def get_hook(layer_stats):
            def hook(module, input, output):
                # detach output from comp grad for stat computation
                out = output.detach()
                mean = round(float(out.mean().item()), 4)
                std = round(float(out.std().item()), 4)
                dead_fraction = round(float((out <= 0).all(dim=0).float().mean().item()), 4)

                layer_stats.append({
                    "mean": mean,
                    "std": std,
                    "dead_fraction": dead_fraction
                })

            return hook

        hooks = []
        for module in model.modules():
            if isinstance(module, nn.Linear):
                hooks.append(module.register_forward_hook(get_hook(stats)))


        with torch.no_grad():
            output = model(x)
        
        for h in hooks:
            h.remove()
        
        return stats

    def compute_gradient_stats(self, model: nn.Module, x: torch.Tensor, y: torch.Tensor) -> List[Dict[str, float]]:
        # Forward + backward pass with nn.MSELoss
        # For each nn.Linear layer's weight gradient, record: mean, std, norm
        # Call model.zero_grad() first. Round to 4 decimals.
        model.zero_grad()

        output = model(x)
        criterion = nn.MSELoss()
        loss = criterion(output, y)
        loss.backward()

        stats = []
        for name, module in model.named_modules():
            if isinstance(module, nn.Linear) and module.weight.grad is not None:
                grad = module.weight.grad
                mean = round(float(grad.mean().item()), 4)
                std = round(float(grad.std().item()), 4)
                norm = round(float(grad.norm().item()), 4)
                stats.append({
                    "mean": mean,
                    "std": std,
                    "norm": norm
                })
            
        return stats

    def diagnose(self, activation_stats: List[Dict[str, float]], gradient_stats: List[Dict[str, float]]) -> str:
        # Classify network health based on the stats
        # Return: 'dead_neurons', 'exploding_gradients', 'vanishing_gradients', or 'healthy'
        # Check in priority order (see problem description for thresholds)
        for layer_stats in activation_stats:
            if layer_stats["dead_fraction"] > 0.5:
                return "dead_neurons"
        
        for layer_grad_stats in gradient_stats:
            if layer_grad_stats["norm"] > 1000:
                return "exploding_gradients"
        
        if gradient_stats[-1]["norm"] < 1e-5:
            return "vanishing_gradients"

        for layer_stats in activation_stats:
            if layer_stats["std"] < 0.1:
                return "vanishing_gradients"
            elif layer_stats["std"] > 10.0:
                return "exploding_gradients"

        return "healthy"