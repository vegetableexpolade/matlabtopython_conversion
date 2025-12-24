"""
findLight.py - Light source direction estimation

This module provides a function to estimate light source directions from
images given known surface normals, equivalent to the MATLAB findLight.m function.
"""

import numpy as np


def findLight(images, normals):
    """
    Estimate light source directions from images.
    
    This function estimates the light source directions for a set of images
    given known surface normals, assuming Lambertian reflectance.
    Equivalent to the MATLAB findLight() function.
    
    Parameters
    ----------
    images : ndarray
        H x W x N array where N is the number of images
    normals : ndarray
        H x W x 3 array of surface normals, or M x 3 where M = H*W
        Surface normals should be unit vectors or will be normalized.
    
    Returns
    -------
    light_dirs : ndarray
        N x 3 array of light directions (unit vectors)
    
    Notes
    -----
    The function uses least squares to estimate light directions assuming
    Lambertian reflectance: I = rho * max(N' * L, 0)
    
    The algorithm:
    1. Reshapes images to column vectors
    2. Normalizes surface normals
    3. For each image, solves least squares: I = N @ (rho * L)
    4. Returns normalized light direction vectors
    
    Examples
    --------
    >>> import numpy as np
    >>> images = np.random.rand(100, 100, 3)  # 3 images
    >>> normals = np.random.rand(100, 100, 3)
    >>> light_dirs = findLight(images, normals)
    >>> light_dirs.shape
    (3, 3)
    """
    height, width, num_images = images.shape
    
    # Reshape images to column vectors (pixels x images)
    I = images.reshape(height * width, num_images)
    
    # Handle normals input - convert 3D to 2D if necessary
    if normals.ndim == 3:
        N = normals.reshape(height * width, 3)
    else:
        N = normals.copy()
    
    # Remove invalid pixels (where normal magnitude is too small)
    normal_magnitudes = np.sqrt(np.sum(N**2, axis=1))
    valid_mask = normal_magnitudes > 0.1
    I_valid = I[valid_mask, :]
    N_valid = N[valid_mask, :]
    
    # Normalize normals to unit vectors
    N_norm = N_valid / np.sqrt(np.sum(N_valid**2, axis=1, keepdims=True))
    
    # Initialize light directions
    light_dirs = np.zeros((num_images, 3))
    
    # Estimate each light direction independently
    for i in range(num_images):
        intensities = I_valid[:, i]
        
        # Remove pixels with very low intensity
        max_intensity = np.max(intensities)
        bright_mask = intensities > max_intensity * 0.1
        
        if np.sum(bright_mask) > 10:
            N_bright = N_norm[bright_mask, :]
            I_bright = intensities[bright_mask]
            
            # Solve for light direction using least squares
            # I = (N' @ L) * rho, we solve for L assuming rho is absorbed in L
            # This is equivalent to: N_bright @ light_dir = I_bright
            light_dir = np.linalg.lstsq(N_bright, I_bright, rcond=None)[0]
            
            # Normalize light direction
            light_dir_norm = np.linalg.norm(light_dir)
            if light_dir_norm > 1e-6:
                light_dirs[i, :] = light_dir / light_dir_norm
            else:
                # Default to upward pointing light if solution is degenerate
                light_dirs[i, :] = np.array([0, 0, 1])
        else:
            # Default to upward pointing light if insufficient data
            light_dirs[i, :] = np.array([0, 0, 1])
    
    return light_dirs
