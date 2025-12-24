"""
smoothNormals.py - Surface normal smoothing

This module provides a function to smooth surface normals using Gaussian filtering,
equivalent to the MATLAB smoothNormals.m function.
"""

import numpy as np
from scipy.ndimage import convolve


def smoothNormals(normals, sigma=2.0):
    """
    Smooth surface normals using Gaussian filtering.
    
    Applies Gaussian smoothing to surface normals while preserving their
    unit length. Equivalent to the MATLAB smoothNormals() function.
    
    Parameters
    ----------
    normals : ndarray
        H x W x 3 array of surface normals
    sigma : float, optional
        Standard deviation of Gaussian kernel (default: 2.0)
        Larger values result in more smoothing
    
    Returns
    -------
    smoothed : ndarray
        H x W x 3 array of smoothed surface normals (unit vectors)
    
    Notes
    -----
    The function:
    1. Creates a Gaussian kernel based on sigma
    2. Smooths each normal component separately using convolution
    3. Renormalizes the resulting vectors to unit length
    
    This preserves the directional information while reducing noise.
    
    Examples
    --------
    >>> import numpy as np
    >>> normals = np.random.rand(50, 50, 3)
    >>> normals = normals / np.linalg.norm(normals, axis=2, keepdims=True)
    >>> smoothed = smoothNormals(normals, sigma=1.5)
    >>> smoothed.shape
    (50, 50, 3)
    >>> # Verify normals are unit vectors
    >>> np.allclose(np.linalg.norm(smoothed, axis=2), 1.0)
    True
    """
    height, width, _ = normals.shape
    
    # Create Gaussian kernel
    kernel_size = int(np.ceil(sigma * 3) * 2 + 1)  # Ensure odd size
    half_size = kernel_size // 2
    
    x, y = np.meshgrid(np.arange(-half_size, half_size + 1),
                       np.arange(-half_size, half_size + 1))
    
    kernel = np.exp(-(x**2 + y**2) / (2 * sigma**2))
    kernel = kernel / np.sum(kernel)  # Normalize
    
    # Smooth each normal component
    smoothed = np.zeros_like(normals)
    for i in range(3):
        smoothed[:, :, i] = convolve(normals[:, :, i], kernel, mode='reflect')
    
    # Renormalize to unit length
    normal_magnitudes = np.sqrt(np.sum(smoothed**2, axis=2, keepdims=True))
    # Avoid division by zero
    normal_magnitudes = np.maximum(normal_magnitudes, 1e-6)
    smoothed = smoothed / normal_magnitudes
    
    # Replace any invalid normals with default upward normal
    invalid_mask = normal_magnitudes.squeeze() < 1e-6
    smoothed[invalid_mask, :] = np.array([0, 0, 1])
    
    return smoothed
