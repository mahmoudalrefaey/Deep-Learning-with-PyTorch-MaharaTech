# Building Models & Making Predictions with PyTorch

This lesson walks through building a simple linear regression model in PyTorch from scratch to understand model construction and prediction.

---

## Goal

Use synthetic data to build a linear model that learns to predict `y` from `x`. The model should ideally learn the weight = 0.7 and bias = 0.3.

---

## Required Libraries

```python
import torch
import torch.nn as nn
````

---

## Defining the Model

We define a custom PyTorch model by subclassing `nn.Module`:

```python
class LinearRegressionModel(nn.Module):
    def __init__(self):
        super().__init__()
        self.weight = nn.Parameter(torch.randn(1, dtype=torch.float), requires_grad=True)
        self.bias = nn.Parameter(torch.randn(1, dtype=torch.float), requires_grad=True)

    def forward(self, x):
        return x * self.weight + self.bias
```

### Model Explanation

* **Parameters**: Learnable values (`weight`, `bias`)
* **Forward pass**: Implements the linear equation `y = wx + b`

---

## 🔧 PyTorch Essentials for Model Building

| Module               | Purpose                                 |
| -------------------- | --------------------------------------- |
| `torch.nn`           | Core components for neural networks     |
| `torch.nn.Parameter` | Tensors with `requires_grad=True`       |
| `torch.nn.Module`    | Base class for all models               |
| `torch.optim`        | Optimization algorithms                 |
| `forward()`          | Method defining the model's computation |

> Think of `nn.Module` as the full building, `nn.Parameter` as bricks, and `forward()` as how the building is used.

---

## Training & Predictions

* **Training**: Update model's weight and bias using training data (`x_train`, `y_train`)
* **Prediction**: Use `x_test` to generate predictions and compare with `y_test`

---

## Summary

This notebook provides a hands-on introduction to:

* Defining a simple model with learnable parameters
* Understanding PyTorch's core abstractions
* Performing forward computation and preparing for training

It's a foundational step toward training more complex models.
