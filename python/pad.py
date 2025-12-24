"""
pad.py - Array padding function

This module provides a function to pad arrays with specified values,
equivalent to the MATLAB pad.m function.
"""

import numpy as np


def pad(img, padsize, method='constant'):
    """
    Pad an array with specified values.
    
    This function is equivalent to the MATLAB pad() function, providing
    padding functionality for 2D and 3D arrays using various methods.
    
    Parameters
    ----------
    img : ndarray
        Input array (can be 2D or 3D)
    padsize : int or tuple of int
        Size of padding on each side. If scalar, same padding is applied
        to rows and columns. If tuple (rows, cols), different padding for each.
    method : str, optional
        Padding method: 'constant' (default), 'replicate', or 'symmetric'
        - 'constant': Pad with zeros
        - 'replicate': Pad by repeating edge values
        - 'symmetric': Pad by mirroring edge values
    
    Returns
    -------
    padded : ndarray
        Padded array with same dtype as input
    
    Examples
    --------
    >>> import numpy as np
    >>> A = np.array([[1, 2], [3, 4]])
    >>> B = pad(A, 1, 'constant')
    >>> B.shape
    (4, 4)
    
    Notes
    -----
    This function replicates the behavior of the MATLAB pad() function,
    using NumPy for efficient array operations.
    """
    # Convert scalar padsize to tuple
    if np.isscalar(padsize):
        padsize = (padsize, padsize)
    
    # Get dimensions
    if img.ndim == 2:
        img = img[:, :, np.newaxis]  # Add channel dimension
        squeeze_output = True
    else:
        squeeze_output = False
    
    rows, cols, channels = img.shape
    
    # Calculate output size
    new_rows = rows + 2 * padsize[0]
    new_cols = cols + 2 * padsize[1]
    
    # Initialize output array
    if method == 'constant':
        # Pad with zeros
        padded = np.zeros((new_rows, new_cols, channels), dtype=img.dtype)
        # Copy original image to center
        padded[padsize[0]:padsize[0]+rows, padsize[1]:padsize[1]+cols, :] = img
        
    elif method == 'replicate':
        # Pad by replicating edge values
        padded = np.zeros((new_rows, new_cols, channels), dtype=img.dtype)
        # Copy original image to center
        padded[padsize[0]:padsize[0]+rows, padsize[1]:padsize[1]+cols, :] = img
        
        # Replicate edges
        # Top and bottom
        for i in range(padsize[0]):
            padded[i, padsize[1]:padsize[1]+cols, :] = img[0, :, :]
            padded[new_rows-i-1, padsize[1]:padsize[1]+cols, :] = img[-1, :, :]
        
        # Left and right (including corners)
        for j in range(padsize[1]):
            padded[:, j, :] = padded[:, padsize[1], :]
            padded[:, new_cols-j-1, :] = padded[:, padsize[1]+cols-1, :]
        
    elif method == 'symmetric':
        # Pad by mirroring edge values
        padded = np.zeros((new_rows, new_cols, channels), dtype=img.dtype)
        # Copy original image to center
        padded[padsize[0]:padsize[0]+rows, padsize[1]:padsize[1]+cols, :] = img
        
        # Mirror edges
        # Top
        if padsize[0] > 0:
            mirror_rows = min(padsize[0], rows)
            padded[:mirror_rows, padsize[1]:padsize[1]+cols, :] = \
                np.flip(img[:mirror_rows, :, :], axis=0)
        
        # Bottom
        if padsize[0] > 0:
            mirror_rows = min(padsize[0], rows)
            padded[-mirror_rows:, padsize[1]:padsize[1]+cols, :] = \
                np.flip(img[-mirror_rows:, :, :], axis=0)
        
        # Left
        if padsize[1] > 0:
            mirror_cols = min(padsize[1], cols)
            padded[:, :mirror_cols, :] = \
                np.flip(padded[:, padsize[1]:padsize[1]+mirror_cols, :], axis=1)
        
        # Right
        if padsize[1] > 0:
            mirror_cols = min(padsize[1], cols)
            padded[:, -mirror_cols:, :] = \
                np.flip(padded[:, padsize[1]+cols-mirror_cols:padsize[1]+cols, :], axis=1)
    else:
        raise ValueError(f'Unknown padding method: {method}')
    
    # Remove extra dimension if input was 2D
    if squeeze_output:
        padded = padded.squeeze(axis=2)
    
    return padded
