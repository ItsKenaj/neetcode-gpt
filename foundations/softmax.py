import numpy as np
from numpy.typing import NDArray


class Solution:

    def softmax(self, z: NDArray[np.float64]) -> NDArray[np.float64]:
        # z is a 1D NumPy array of logits
        # Hint: subtract max(z) for numerical stability before computing exp
        # return np.round(your_answer, 4)
        z -= max(z)
        exp_z = np.exp(z)
        return np.round(exp_z / np.sum(exp_z, axis=-1, keepdims=True), 4)

