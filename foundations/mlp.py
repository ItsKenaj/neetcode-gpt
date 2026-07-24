import numpy as np
from numpy.typing import NDArray
from typing import List


class Solution:
    def forward(self, x: NDArray[np.float64], weights: List[NDArray[np.float64]], biases: List[NDArray[np.float64]]) -> NDArray[np.float64]:
        # x: 1D input array
        # weights: list of 2D weight matrices
        # biases: list of 1D bias vectors
        # Apply ReLU after each hidden layer, no activation on output layer
        # return np.round(your_answer, 5)
        layers = len(weights)
        if layers == 1:
            return x @ weights[0] + biases[0]

        h = np.maximum(0, x @ weights[0] + biases[0])
        for i in range(1, layers - 1):
            h = np.maximum(0, h @ weights[i] + biases[i])

        
        return np.round(h @ weights[-1] + biases[-1], 5)