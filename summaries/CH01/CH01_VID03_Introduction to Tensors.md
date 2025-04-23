# Detailed Summary: CH01_VID03_Introduction to Tensors

## Why Tensors?
- In deep learning, everything must be represented as numbers, particularly **tensors**, to be processed by machine learning models.
- Data types like:
  - **Images**: composed of pixels (e.g., RGB values),
  - **Text**: raw strings,
  - **Audio or Excel sheets**,
  cannot be directly understood by models.
- Example:
  - An image in RGB format is converted into a tensor with shape `[3, 224, 224]`, representing:
    - 3 color channels (Red, Green, Blue),
    - Height and Width (224 × 224 pixels).
- **Key Takeaway**: Data must be numerical (tensor format) to serve as input to models like CNNs or Transformers.

## Converting Text and Numbers to Tensors
- Just like images, **text must also be converted to numbers** because models can't process raw strings.
- This is typically done using encoding methods (e.g., **word embeddings**, **tokenization**).
- Torch (PyTorch) offers built-in functions for creating tensors from basic Python values (e.g., integers, lists).

## Creating Scalars (0-D Tensors)
- A **scalar** is a single number (e.g., `5`), but in PyTorch it must be wrapped in a tensor:
  ```python
  torch.tensor(5)
  ```
- Environments like **Google Colab** and **VS Code** help auto-suggest functions (IntelliSense).
- You can generate code using AI assistants (e.g., Google Gemini or ChatGPT) to help write tensor creation code.

## Tensor Creation Functions in PyTorch
- PyTorch provides many tensor creation functions:
  - `torch.tensor()`
  - `torch.IntTensor()`, `torch.FloatTensor()` for specifying data types.
- These are abstracted over lower-level C/C++ operations.
- Allows Python users to work efficiently with deep learning structures without diving into low-level languages.

## Understanding Dimensions and `.item()`
- Dimensions define the shape of a tensor:
  - Scalar → 0D
  - Vector → 1D
  - Matrix → 2D
  - Tensor → 3D or higher
- Example:
  ```python
  x = torch.tensor(5)
  print(x.dim())  # Output: 0
  ```
- To extract the raw number from a tensor (useful in predictions):
  ```python
  x.item()
  ```
  - Especially useful when returning model outputs to a UI or integration pipeline.

## Creating Vectors (1-D Tensors)
- A **vector** is a 1D tensor:
  ```python
  torch.tensor([5, 5])
  ```
- Used for simple numerical lists like number of bedrooms, ratings, etc.
- Shape and dimensionality can be checked using:
  ```python
  x.shape  # torch.Size([2])
  x.dim()  # 1
  ```

## Creating Matrices (2-D Tensors)
- A **matrix** is a 2D tensor (list of lists):
  ```python
  torch.tensor([[5, 5], [7, 7]])
  ```
- Shape:
  - `.shape` → `torch.Size([2, 2])` → 2 rows, 2 columns
  - `.dim()` → 2
- Common in tabular data, images, and layer weights in neural networks.

## Creating Higher-Dimensional Tensors
- You can create 3D or higher tensors for complex structures:
  ```python
  torch.tensor([
      [[1, 2, 3]],
      [[4, 5, 6]]
  ])
  ```
- Shape: `[2, 1, 3]` → 2 blocks, 1 row each, 3 columns.
- Useful in:
  - Computer Vision (batches of images),
  - NLP (sequence of embeddings),
  - Time-series (multiple time steps).

- Reshaping:
  ```python
  tensor.view(1, 3, 3)
  ```
  is often needed to match the required input of a model.

## Importance of Reshaping and Dimension Matching
- Sometimes, you need to **expand dimensions** to make 2D → 3D:
  ```python
  x = torch.tensor([[1, 2, 3]])
  x.view(1, 1, 3)  # adds extra dimension
  ```
- Each axis corresponds to a concept like batch size, channels, sequence length, etc.
- Matching input/output shapes is critical to avoid runtime errors.

## Summary of Tensor Types
| Type     | Dimension | Example Syntax              | Description                            |
|----------|-----------|-----------------------------|----------------------------------------|
| Scalar   | 0D        | `torch.tensor(5)`           | A single number                        |
| Vector   | 1D        | `torch.tensor([1, 2, 3])`   | A list of numbers                      |
| Matrix   | 2D        | `torch.tensor([[1, 2], [3, 4]])` | A 2D table                        |
| Tensor   | 3D+       | `torch.tensor([[[1,2]]])`   | Multidimensional data structure        |

- These types are symbolically referred to as:
  - Scalar → `x`
  - Vector → `X` (uppercase)
  - Matrix → `Q`
  - Tensor → general multi-dim symbol like `𝑇` or `𝑿`

## Final Notes
- Understanding tensors is **foundational** for working with AI/ML models.
- Properly managing **shape, dimension, and type** of tensors ensures successful model training and prediction.
- PyTorch simplifies tensor creation, reshaping, and manipulation through high-level, readable APIs.
