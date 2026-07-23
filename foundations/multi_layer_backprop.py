import numpy as np
from typing import List


class Solution:
    def forward_and_backward(self,
                              x: List[float],
                              W1: List[List[float]], b1: List[float],
                              W2: List[List[float]], b2: List[float],
                              y_true: List[float]) -> dict:
        # Architecture: x -> Linear(W1, b1) -> ReLU -> Linear(W2, b2) -> predictions
        # Loss: MSE = mean((predictions - y_true)^2)
        #
        # Return dict with keys:
        #   'loss':  float (MSE loss, rounded to 4 decimals)
        #   'dW1':   2D list (gradient w.r.t. W1, rounded to 4 decimals)
        #   'db1':   1D list (gradient w.r.t. b1, rounded to 4 decimals)
        #   'dW2':   2D list (gradient w.r.t. W2, rounded to 4 decimals)
        #   'db2':   1D list (gradient w.r.t. b2, rounded to 4 decimals)
        x, W1, b1, W2, b2, y_true = (np.array(x, dtype=float),
         np.array(W1, dtype=float),
         np.array(b1, dtype=float),
         np.array(W2, dtype=float),
         np.array(b2, dtype=float),
         np.array(y_true, dtype=float))
        
        N = len(y_true)
        z1 = np.dot(x, W1.T) + b1
        a1 = np.maximum(0, z1)
        z2 = np.dot(a1, W2.T) + b2
        loss = np.mean((z2 - y_true)**2)

        dz2 = 2/N * (z2 - y_true)
        da1 = dz2 @ W2
        dW2 = np.outer(dz2, a1)
        db2 = dz2
        relu_mask = z1 > 0
        dz1 = da1 * relu_mask
        dW1 = np.outer(dz1, x)
        db1 = dz1 * np.ones_like(b1)

        return {
            'loss': round(loss, 4).tolist(),
            'dW1': np.round(dW1, 4).tolist(),
            'db1': np.round(db1, 4).tolist(),
            'dW2': np.round(dW2, 4).tolist(),
            'db2': np.round(db2, 4).tolist()
        }
        

