import torch
import torch.nn as nn
from torchtyping import TensorType

class Solution(nn.Module):
    def __init__(self):
        super().__init__()
        torch.manual_seed(0)
        # Architecture: Linear(784, 512) -> ReLU -> Dropout(0.2) -> Linear(512, 10) -> Sigmoid
        self.model = nn.Sequential(
            nn.Linear(in_features=784, out_features=512),  
            nn.ReLU(),                                   
            nn.Dropout(p=0.2),   
            nn.Linear(in_features=512, out_features=10),
            nn.Sigmoid()     
        )

    def forward(self, images: TensorType[float]) -> TensorType[float]:
        torch.manual_seed(0)
        # images shape: (batch_size, 784)
        # Return the model's prediction to 4 decimal places
        return torch.round(self.model(images), decimals=4)
