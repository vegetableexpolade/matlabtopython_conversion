"""
integrateNormals.py - Surface normal integration

This module provides a function to integrate surface normals to recover depth,
equivalent to the MATLAB integrateNormals.m function.
"""

import numpy as np
from scipy.fft import dctn, idctn


def integrateNormals(normals, method='average'):
    """
    Integrate surface normals to recover depth.
    
    Integrates surface normals to estimate the depth map (height field).
    Equivalent to the MATLAB integrateNormals() function.
    
    Parameters
    ----------
    normals : ndarray
        H x W x 3 array of surface normals
    method : str, optional
        Integration method: 'average' (default) or 'poisson'
        - 'average': Simple path integration using averaging
        - 'poisson': Poisson integration using DCT
    
    Returns
    -------
    depth : ndarray
        H x W depth map
    
    Notes
    -----
    The function uses gradient integration to reconstruct the surface.
    
    For the 'average' method:
    - Computes gradients from normals: p = -nx/nz, q = -ny/nz
    - Integrates from top-left corner using path integration
    - Averages contributions from left and top paths
    
    For the 'poisson' method:
    - Solves the Poisson equation using DCT
    - More robust to noise but computationally more expensive
    
    Examples
    --------
    >>> import numpy as np
    >>> normals = np.random.rand(50, 50, 3)
    >>> normals = normals / np.linalg.norm(normals, axis=2, keepdims=True)
    >>> depth = integrateNormals(normals, method='average')
    >>> depth.shape
    (50, 50)
    """
    height, width, _ = normals.shape
    
    # Extract normal components
    nx = normals[:, :, 0]
    ny = normals[:, :, 1]
    nz = normals[:, :, 2]
    
    # Avoid division by zero
    nz = np.where(np.abs(nz) < 1e-6, 1e-6, nz)
    
    # Compute gradients: p = -nx/nz, q = -ny/nz
    p = -nx / nz
    q = -ny / nz
    
    if method == 'average':
        # Simple path integration using averaging
        depth = np.zeros((height, width))
        
        # Integrate from top-left corner
        # Along first row
        for j in range(1, width):
            depth[0, j] = depth[0, j-1] + p[0, j-1]
        
        # Along first column
        for i in range(1, height):
            depth[i, 0] = depth[i-1, 0] + q[i-1, 0]
        
        # Fill in the rest using average of two paths
        for i in range(1, height):
            for j in range(1, width):
                from_left = depth[i, j-1] + p[i, j-1]
                from_top = depth[i-1, j] + q[i-1, j]
                depth[i, j] = (from_left + from_top) / 2
        
    elif method == 'poisson':
        # Poisson integration using DCT
        # Compute divergence of gradient field
        px = np.gradient(p, axis=1)
        qy = np.gradient(q, axis=0)
        div = px + qy
        
        # Solve Poisson equation using DCT (Discrete Cosine Transform)
        D = dctn(div, type=2, norm='ortho')
        
        # Create frequency domain denominator
        x, y = np.meshgrid(np.arange(width), np.arange(height))
        denom = 2 * (np.cos(np.pi * x / width) + np.cos(np.pi * y / height) - 2)
        denom[0, 0] = 1  # Avoid division by zero at DC component
        
        Z = D / denom
        Z[0, 0] = 0  # Set DC component to zero (mean depth = 0)
        
        depth = idctn(Z, type=2, norm='ortho')
    else:
        raise ValueError(f'Unknown integration method: {method}')
    
    return depth
