# MATLAB to Python Conversion

This repository demonstrates the conversion of MATLAB functions to Python with functional equivalence. The converted functions focus on photometric stereo and image processing operations, using NumPy, SciPy, and Matplotlib to replicate MATLAB functionality.

## Overview

The repository contains:
- **MATLAB implementations** (`matlab/` directory): Original MATLAB functions
- **Python implementations** (`python/` directory): Equivalent Python functions
- **Test suite** (`tests/` directory): Validation tests ensuring functional equivalence

## Functions Converted

### 1. `pad.m` → `pad.py`
Array padding function supporting multiple padding methods (constant, replicate, symmetric).

**MATLAB Example:**
```matlab
A = [1 2; 3 4];
B = pad(A, 1, 'constant');
```

**Python Example:**
```python
import numpy as np
from pad import pad

A = np.array([[1, 2], [3, 4]])
B = pad(A, 1, 'constant')
```

### 2. `findLight.m` → `findLight.py`
Estimates light source directions from images given known surface normals, assuming Lambertian reflectance.

**Key features:**
- Handles multiple images and normals
- Uses least squares estimation
- Returns normalized light direction vectors

### 3. `LambertianSFP.m` → `LambertianSFP.py`
Lambertian Shape from Photometric Stereo - recovers surface normals and albedo from multiple images under different lighting conditions.

**Key features:**
- Implements the Lambertian reflectance model
- Solves least squares for each pixel
- Returns unit normal vectors and albedo map

### 4. `integrateNormals.m` → `integrateNormals.py`
Integrates surface normals to recover depth maps using path integration or Poisson solver.

**Supported methods:**
- `'average'`: Simple path integration with averaging
- `'poisson'`: Poisson integration using DCT (Discrete Cosine Transform)

### 5. `smoothNormals.m` → `smoothNormals.py`
Smooths surface normals using Gaussian filtering while preserving unit length.

**Key features:**
- Creates Gaussian kernel based on sigma parameter
- Smooths each component separately
- Renormalizes to maintain unit vectors

## Installation

### Prerequisites
- Python 3.7 or higher
- pip package manager

### Setup
```bash
# Install required packages
pip install -r requirements.txt
```

## Running Tests

The test suite validates that Python implementations produce equivalent results to MATLAB implementations:

```bash
cd tests
python test_conversion.py
```

### Test Coverage
- Padding methods (constant, replicate, symmetric)
- 2D and 3D array handling
- Light direction estimation
- Photometric stereo reconstruction
- Normal integration
- Normal smoothing

## Conversion Approach

### Design Principles
1. **One-to-one functional equivalence**: Python functions produce the same outputs as MATLAB functions for equivalent inputs
2. **Modern Python standards**: Code follows PEP 8 style guidelines with comprehensive docstrings
3. **Library usage**: Leverages NumPy, SciPy, and Matplotlib to replicate MATLAB functionality
4. **Well-commented code**: Extensive documentation explaining algorithms and conversions

### Key Differences from MATLAB

| Aspect | MATLAB | Python |
|--------|--------|--------|
| Indexing | 1-based | 0-based |
| Array operations | Built-in | NumPy library |
| Matrix operations | Native | NumPy/SciPy |
| DCT | `dct2`, `idct2` | `scipy.fft.dctn`, `idctn` |
| Convolution | `conv2` | `scipy.ndimage.convolve` |
| Least squares | `\` operator | `np.linalg.lstsq` |

### Common Conversion Patterns

#### Array Creation
```matlab
% MATLAB
A = zeros(10, 10);
B = ones(5, 5, 3);
```

```python
# Python
A = np.zeros((10, 10))
B = np.ones((5, 5, 3))
```

#### Array Indexing
```matlab
% MATLAB (1-based)
A(1, 1)          % First element
A(1:5, :)        % First 5 rows
A(end, end)      % Last element
```

```python
# Python (0-based)
A[0, 0]          # First element
A[0:5, :]        # First 5 rows
A[-1, -1]        # Last element
```

#### Matrix Operations
```matlab
% MATLAB
C = A \ b;       % Solve linear system
```

```python
# Python
C = np.linalg.lstsq(A, b, rcond=None)[0]
```

## Project Structure

```
matlabtopython_conversion/
├── matlab/                  # MATLAB implementations
│   ├── pad.m
│   ├── findLight.m
│   ├── LambertianSFP.m
│   ├── integrateNormals.m
│   └── smoothNormals.m
├── python/                  # Python implementations
│   ├── pad.py
│   ├── findLight.py
│   ├── LambertianSFP.py
│   ├── integrateNormals.py
│   └── smoothNormals.py
├── tests/                   # Test suite
│   └── test_conversion.py
├── requirements.txt         # Python dependencies
└── README.md               # This file
```

## Usage Examples

### Complete Photometric Stereo Pipeline

```python
import numpy as np
from LambertianSFP import LambertianSFP
from smoothNormals import smoothNormals
from integrateNormals import integrateNormals

# Load images (H x W x N)
images = np.load('images.npy')

# Define light directions (N x 3)
light_dirs = np.array([
    [0, 0, 1],
    [1, 0, 0],
    [0, 1, 0],
    [0.707, 0.707, 0]
])

# Recover normals and albedo
normals, albedo = LambertianSFP(images, light_dirs)

# Smooth normals
normals_smooth = smoothNormals(normals, sigma=2.0)

# Integrate to get depth
depth = integrateNormals(normals_smooth, method='poisson')
```

## Technical Notes

### Numerical Precision
- Both implementations use double-precision floating-point arithmetic
- Relative tolerance for tests: `atol=0.01` (1% difference allowed)
- Some minor differences expected due to:
  - Different solver implementations
  - Floating-point arithmetic order of operations
  - Library-specific optimizations

### Performance Considerations
- Python implementations use vectorized NumPy operations for efficiency
- For large datasets, consider using sparse matrices (SciPy sparse)
- DCT operations in Poisson integration are computationally expensive

### Limitations
- Assumes Lambertian reflectance (no specular highlights)
- Does not handle shadows or inter-reflections
- Integration methods assume continuous surfaces

## Contributing

When adding new function conversions:
1. Add MATLAB implementation to `matlab/` directory
2. Create equivalent Python implementation in `python/` directory
3. Add comprehensive tests to `tests/test_conversion.py`
4. Update this README with documentation
5. Ensure tests pass with high accuracy

## License

This project is provided as-is for educational and research purposes.

## References

- Woodham, R.J. (1980). "Photometric method for determining surface orientation from multiple images"
- Horn, B.K.P., & Brooks, M.J. (1989). "Shape from Shading"
- NumPy Documentation: https://numpy.org/doc/
- SciPy Documentation: https://docs.scipy.org/