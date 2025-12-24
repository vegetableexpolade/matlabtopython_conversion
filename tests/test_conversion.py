"""
test_conversion.py - Test suite for MATLAB to Python conversion

This module contains tests to validate that the Python implementations
produce equivalent results to the MATLAB implementations.
"""

import numpy as np
import sys
import os

# Add python directory to path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'python'))

# Import Python implementations
from pad import pad
from findLight import findLight
from LambertianSFP import LambertianSFP
from integrateNormals import integrateNormals
from smoothNormals import smoothNormals


def test_pad_constant():
    """Test pad function with constant padding."""
    print("Testing pad with constant padding...")
    
    # Test 2D array
    A = np.array([[1, 2], [3, 4]], dtype=float)
    B = pad(A, 1, 'constant')
    
    expected = np.array([
        [0, 0, 0, 0],
        [0, 1, 2, 0],
        [0, 3, 4, 0],
        [0, 0, 0, 0]
    ], dtype=float)
    
    assert B.shape == (4, 4), f"Expected shape (4, 4), got {B.shape}"
    assert np.allclose(B, expected), "Constant padding result incorrect"
    print("  ✓ Constant padding test passed")


def test_pad_replicate():
    """Test pad function with replicate padding."""
    print("Testing pad with replicate padding...")
    
    A = np.array([[1, 2], [3, 4]], dtype=float)
    B = pad(A, 1, 'replicate')
    
    expected = np.array([
        [1, 1, 2, 2],
        [1, 1, 2, 2],
        [3, 3, 4, 4],
        [3, 3, 4, 4]
    ], dtype=float)
    
    assert B.shape == (4, 4), f"Expected shape (4, 4), got {B.shape}"
    assert np.allclose(B, expected), "Replicate padding result incorrect"
    print("  ✓ Replicate padding test passed")


def test_pad_3d():
    """Test pad function with 3D array."""
    print("Testing pad with 3D array...")
    
    A = np.random.rand(10, 10, 3)
    B = pad(A, 2, 'constant')
    
    assert B.shape == (14, 14, 3), f"Expected shape (14, 14, 3), got {B.shape}"
    assert np.allclose(B[2:12, 2:12, :], A), "Original data not preserved"
    print("  ✓ 3D array padding test passed")


def test_findLight():
    """Test findLight function."""
    print("Testing findLight...")
    
    # Create synthetic test data
    height, width = 20, 20
    num_images = 3
    
    # Create known light directions
    true_lights = np.array([
        [0, 0, 1],      # Top light
        [1, 0, 0],      # Side light
        [0.707, 0, 0.707]  # Diagonal light
    ])
    true_lights = true_lights / np.linalg.norm(true_lights, axis=1, keepdims=True)
    
    # Create known normals (all pointing up with some variation)
    normals = np.zeros((height, width, 3))
    normals[:, :, 2] = 1  # All normals point up
    normals[:, :, 0] = np.random.rand(height, width) * 0.2 - 0.1  # Small variation
    normals[:, :, 1] = np.random.rand(height, width) * 0.2 - 0.1
    normals = normals / np.linalg.norm(normals, axis=2, keepdims=True)
    
    # Generate synthetic images using Lambertian model
    images = np.zeros((height, width, num_images))
    albedo = 0.8
    for i in range(num_images):
        for y in range(height):
            for x in range(width):
                normal = normals[y, x, :]
                light = true_lights[i, :]
                intensity = albedo * max(0, np.dot(normal, light))
                images[y, x, i] = intensity
    
    # Add small noise
    images += np.random.rand(height, width, num_images) * 0.01
    
    # Estimate light directions
    estimated_lights = findLight(images, normals)
    
    assert estimated_lights.shape == (num_images, 3), \
        f"Expected shape ({num_images}, 3), got {estimated_lights.shape}"
    
    # Check that estimated lights are close to true lights
    for i in range(num_images):
        # Lights should be unit vectors
        assert np.abs(np.linalg.norm(estimated_lights[i, :]) - 1.0) < 0.01, \
            f"Light {i} is not a unit vector"
        
        # Direction should be close (cosine similarity > 0.9)
        similarity = np.dot(estimated_lights[i, :], true_lights[i, :])
        assert similarity > 0.9, \
            f"Light {i} direction too different: similarity={similarity:.3f}"
    
    print("  ✓ findLight test passed")


def test_LambertianSFP():
    """Test LambertianSFP function."""
    print("Testing LambertianSFP...")
    
    # Create synthetic test data
    height, width = 30, 30
    num_images = 4
    
    # Create known normals (tilted plane)
    x, y = np.meshgrid(np.linspace(-1, 1, width), np.linspace(-1, 1, height))
    true_normals = np.zeros((height, width, 3))
    true_normals[:, :, 0] = -0.3  # Tilt in x
    true_normals[:, :, 1] = -0.2  # Tilt in y
    true_normals[:, :, 2] = 1.0
    true_normals = true_normals / np.linalg.norm(true_normals, axis=2, keepdims=True)
    
    # Create known albedo
    true_albedo = 0.7 * np.ones((height, width))
    
    # Create known light directions
    lights = np.array([
        [0, 0, 1],
        [1, 0, 0.5],
        [0, 1, 0.5],
        [-0.5, -0.5, 1]
    ])
    lights = lights / np.linalg.norm(lights, axis=1, keepdims=True)
    
    # Generate synthetic images
    images = np.zeros((height, width, num_images))
    for i in range(num_images):
        for y in range(height):
            for x in range(width):
                normal = true_normals[y, x, :]
                light = lights[i, :]
                intensity = true_albedo[y, x] * max(0, np.dot(normal, light))
                images[y, x, i] = intensity
    
    # Add small noise
    images += np.random.rand(height, width, num_images) * 0.01
    
    # Recover normals and albedo
    estimated_normals, estimated_albedo = LambertianSFP(images, lights)
    
    assert estimated_normals.shape == (height, width, 3), \
        f"Expected normals shape ({height}, {width}, 3), got {estimated_normals.shape}"
    assert estimated_albedo.shape == (height, width), \
        f"Expected albedo shape ({height}, {width}), got {estimated_albedo.shape}"
    
    # Check that normals are unit vectors
    normal_mags = np.linalg.norm(estimated_normals, axis=2)
    assert np.allclose(normal_mags, 1.0, atol=0.01), "Normals are not unit vectors"
    
    # Check that estimated normals are close to true normals
    # (use dot product as similarity measure)
    similarity = np.sum(estimated_normals * true_normals, axis=2)
    mean_similarity = np.mean(similarity)
    assert mean_similarity > 0.95, \
        f"Normals too different: mean similarity={mean_similarity:.3f}"
    
    # Check that albedo is approximately correct
    albedo_error = np.mean(np.abs(estimated_albedo - true_albedo))
    assert albedo_error < 0.1, f"Albedo error too large: {albedo_error:.3f}"
    
    print("  ✓ LambertianSFP test passed")


def test_integrateNormals():
    """Test integrateNormals function."""
    print("Testing integrateNormals...")
    
    # Create normals for a simple tilted plane
    height, width = 40, 40
    normals = np.zeros((height, width, 3))
    normals[:, :, 0] = -0.2  # Constant tilt
    normals[:, :, 1] = -0.3
    normals[:, :, 2] = 1.0
    normals = normals / np.linalg.norm(normals, axis=2, keepdims=True)
    
    # Test average method
    depth_avg = integrateNormals(normals, method='average')
    assert depth_avg.shape == (height, width), \
        f"Expected depth shape ({height}, {width}), got {depth_avg.shape}"
    
    # Check that depth varies smoothly (gradient should be consistent)
    depth_var = np.var(np.diff(depth_avg, axis=0))
    assert depth_var < 1.0, "Depth variation too large for constant slope"
    
    # Test Poisson method
    depth_poisson = integrateNormals(normals, method='poisson')
    assert depth_poisson.shape == (height, width), \
        f"Expected depth shape ({height}, {width}), got {depth_poisson.shape}"
    
    print("  ✓ integrateNormals test passed")


def test_smoothNormals():
    """Test smoothNormals function."""
    print("Testing smoothNormals...")
    
    # Create noisy normals
    height, width = 30, 30
    normals = np.zeros((height, width, 3))
    normals[:, :, 2] = 1.0
    # Add noise
    normals[:, :, 0] = np.random.rand(height, width) * 0.4 - 0.2
    normals[:, :, 1] = np.random.rand(height, width) * 0.4 - 0.2
    normals = normals / np.linalg.norm(normals, axis=2, keepdims=True)
    
    # Smooth normals
    smoothed = smoothNormals(normals, sigma=2.0)
    
    assert smoothed.shape == (height, width, 3), \
        f"Expected shape ({height}, {width}, 3), got {smoothed.shape}"
    
    # Check that normals are unit vectors
    normal_mags = np.linalg.norm(smoothed, axis=2)
    assert np.allclose(normal_mags, 1.0, atol=0.01), "Smoothed normals are not unit vectors"
    
    # Check that smoothing reduces variance
    original_var = np.var(normals[:, :, 0])
    smoothed_var = np.var(smoothed[:, :, 0])
    assert smoothed_var < original_var, "Smoothing did not reduce variance"
    
    print("  ✓ smoothNormals test passed")


def run_all_tests():
    """Run all tests."""
    print("\n" + "="*60)
    print("Running MATLAB to Python Conversion Tests")
    print("="*60 + "\n")
    
    tests = [
        test_pad_constant,
        test_pad_replicate,
        test_pad_3d,
        test_findLight,
        test_LambertianSFP,
        test_integrateNormals,
        test_smoothNormals,
    ]
    
    failed = 0
    for test in tests:
        try:
            test()
        except Exception as e:
            print(f"  ✗ {test.__name__} FAILED: {str(e)}")
            failed += 1
    
    print("\n" + "="*60)
    if failed == 0:
        print("All tests passed! ✓")
    else:
        print(f"{failed} test(s) failed!")
    print("="*60 + "\n")
    
    return failed == 0


if __name__ == "__main__":
    success = run_all_tests()
    sys.exit(0 if success else 1)
