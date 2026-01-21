"""
Why tensors are used over Python lists in Deep Learning
-------------------------------------------------------
Deep learning models rely on fast, large-scale numerical computations.
While Python lists are flexible, they are slow for numerical operations
and do not support GPU acceleration or automatic differentiation.

Tensors (NumPy / PyTorch) are optimized multi-dimensional arrays that:
• Enable fast, vectorized computations
• Support GPU/TPU acceleration
• Track operations for automatic gradient computation
• Provide built-in support for broadcasting, reshaping, and linear algebra

Small comparison:
List:    [1, 2, 3] + [4, 5, 6]  → Error (manual looping needed)
Tensor:  tensor([1, 2, 3]) + tensor([4, 5, 6]) → tensor([5, 7, 9])

Time comparison (1,000,000 elements on CPU):
List addition   ≈ 0.030 s  (Python loop)
Tensor addition ≈ 0.002 s  (Vectorized C/C++ backend)

Hence, tensors are the fundamental data structure in deep learning.
"""
import numpy as np
import torch

# 1D
np_1d = np.array([1, 2, 3])
pt_1d = torch.tensor([1, 2, 3])

# 2D
np_2d = np.array([[1, 2], [3, 4]])
pt_2d = torch.tensor([[1, 2], [3, 4]])

# 3D (Depth, Rows, Cols)
np_3d = np.ones((2, 3, 4)) 
pt_3d = torch.ones((2, 3, 4))

print(f"PyTorch 3D Shape: {pt_3d.shape}")
print(f"NumPy 3D Shape: {np_3d.shape}\n")

# 2. ELEMENT-WISE OPERATIONS

print("--- 2. Element-wise Operations ---")
a = torch.tensor([10, 20, 30])
b = torch.tensor([1, 2, 3])

print(f"Addition: {a + b}")
print(f"Multiplication: {a * b}") # Element-wise
print(f"Power: {a ** 2}\n")


# 3. INDEXING & SLICING
print("--- 3. Indexing & Slicing ---")
data = torch.arange(12).reshape(3, 4)
print(f"Original Matrix:\n{data}")

# Subtensor extraction (Rows 0-1, Cols 1-2)
sub = data[0:2, 1:3]
print(f"Sub-tensor (top-right block):\n{sub}")

# Boolean Masking
mask = data > 5
print(f"Elements > 5: {data[mask]}")

# Replacing values via mask
data[data < 3] = -1
print(f"After masking (values < 3 set to -1):\n{data}\n")

# 4. RESHAPING & DIMENSIONALITY
print("--- 4. Reshaping ---")
x = torch.arange(6) # [0, 1, 2, 3, 4, 5]

# .view vs .reshape
# .view() is a memory-efficient "look" at the same data
view_x = x.view(2, 3) 
# .reshape() is safer if memory is not contiguous
reshape_x = x.reshape(3, 2)

# Squeeze and Unsqueeze
y = torch.tensor([1, 2, 3])      # Shape [3]
y_unsqueezed = y.unsqueeze(0)    # Shape [1, 3] (Added row dim)
y_squeezed = y_unsqueezed.squeeze() # Back to [3]

print(f"Unsqueezed Shape: {y_unsqueezed.shape}")
print(f"Squeezed Shape: {y_squeezed.shape}\n")

# 5. BROADCASTING
print("--- 5. Broadcasting ---")
# Scaling every row of a (3, 2) matrix by a (1, 2) vector
matrix = torch.ones(3, 2)
scale = torch.tensor([10, 20]) # Shape [2] -> broadcasted to [3, 2]
result = matrix * scale

print(f"Matrix (3x2) * Scale (2):\n{result}\n")

# 6. IN-PLACE VS OUT-OF-PLACE
print("--- 6. In-place vs Out-of-place ---")
t = torch.tensor([1.0, 2.0])

# Out-of-place (creates new tensor)
t_new = t.add(5)
print(f"Original after out-of-place: {t}")

# In-place (modifies existing tensor, ends with '_')
t.add_(5)
print(f"Original after in-place: {t}")