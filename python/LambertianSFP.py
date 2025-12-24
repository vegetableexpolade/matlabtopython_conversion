"""
LambertianSFP.py - Lambertian Shape from Photometric Stereo

This module provides a function to recover surface normals and albedo from
multiple images under different lighting conditions, equivalent to the MATLAB
LambertianSFP.m function.
"""

import numpy as np


def LambertianSFP(images, light_dirs):
    """
    Lambertian Shape from Photometric Stereo.
    
    Recovers surface normals and albedo from multiple images under different
    lighting conditions, assuming Lambertian reflectance. Equivalent to the
    MATLAB LambertianSFP() function.
    
    Parameters
    ----------
    images : ndarray
        H x W x N array of N images of the same scene under different lighting
    light_dirs : ndarray
        N x 3 array of light directions (should be unit vectors)
    
    Returns
    -------
    normals : ndarray
        H x W x 3 array of surface normals (unit vectors)
    albedo : ndarray
        H x W array of albedo (diffuse reflectance) values
    
    Notes
    -----
    The Lambertian reflectance model: I = albedo * max(N' @ L, 0)
    
    This can be solved using least squares for each pixel:
    - Given multiple images I_1, ..., I_N with lights L_1, ..., L_N
    - For each pixel: I = L @ g, where g = albedo * normal
    - Solve for g using least squares
    - Extract albedo = ||g|| and normal = g / ||g||
    
    The algorithm:
    1. Ensures light directions are unit vectors
    2. For each pixel, solves least squares to find g = albedo * normal
    3. Extracts albedo as magnitude of g
    4. Extracts normal as normalized g
    5. Handles degenerate cases with default values
    
    Examples
    --------
    >>> import numpy as np
    >>> images = np.random.rand(50, 50, 4)  # 4 images
    >>> light_dirs = np.array([[0, 0, 1], [1, 0, 0], [0, 1, 0], [1, 1, 1]])
    >>> light_dirs = light_dirs / np.linalg.norm(light_dirs, axis=1, keepdims=True)
    >>> normals, albedo = LambertianSFP(images, light_dirs)
    >>> normals.shape
    (50, 50, 3)
    >>> albedo.shape
    (50, 50)
    """
    height, width, num_images = images.shape
    
    # Ensure light directions are unit vectors
    light_dirs = light_dirs / np.sqrt(np.sum(light_dirs**2, axis=1, keepdims=True))
    
    # Reshape images for processing (images x pixels)
    I = images.reshape(height * width, num_images).T
    
    # Initialize outputs
    N = np.zeros((height * width, 3))
    rho = np.zeros(height * width)
    
    # Solve for each pixel
    for i in range(height * width):
        intensities = I[:, i]
        
        # Check if pixel has sufficient variation
        if np.max(intensities) > 0.01 and np.std(intensities) > 0.001:
            # Solve least squares: intensities = light_dirs @ (rho * normal)
            # This gives us g = rho * normal
            g = np.linalg.lstsq(light_dirs, intensities, rcond=None)[0]
            
            # Extract albedo and normal
            rho[i] = np.linalg.norm(g)
            
            if rho[i] > 1e-6:
                N[i, :] = g / rho[i]
            else:
                N[i, :] = np.array([0, 0, 1])  # Default upward normal
        else:
            # Insufficient data, use default
            N[i, :] = np.array([0, 0, 1])
            rho[i] = 0
    
    # Reshape outputs
    normals = N.reshape(height, width, 3)
    albedo = rho.reshape(height, width)
    
    # Ensure normals are unit vectors
    normal_magnitudes = np.sqrt(np.sum(normals**2, axis=2, keepdims=True))
    normal_magnitudes = np.maximum(normal_magnitudes, 1e-6)  # Avoid division by zero
    normals = normals / normal_magnitudes
    
    return normals, albedo
