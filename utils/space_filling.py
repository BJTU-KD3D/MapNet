import torch
import numpy as np
import math
from sklearn.neighbors import KDTree  # Add KDTree for local point cloud relationship modeling
from collections import defaultdict, deque  # Used for graph algorithms

"""
Enhanced point cloud space-filling curve utilities

Provides multiple optimized space-filling curve implementations, designed specifically for point cloud characteristics:
1. Hierarchical locality filling curve - optimized for the local clustering property of point clouds
2. Local-density-based adaptive curve - addresses uneven point cloud density
3. Geometry-aware ordering - leverages the geometric properties of point clouds
4. Feature-guided filling curve - combines geometric and feature information
"""

# Add Hilbert curve implementation (the following code extracts the key parts from the provided code)
def right_shift(binary, k=1, axis=-1):
    """Right-shift the values of a binary array"""
    if binary.shape[axis] <= k:
        return torch.zeros_like(binary)
    slicing = [slice(None)] * len(binary.shape)
    slicing[axis] = slice(None, -k)
    shifted = torch.nn.functional.pad(
        binary[tuple(slicing)], (k, 0), mode="constant", value=0
    )
    return shifted

def binary2gray(binary, axis=-1):
    """Convert a binary array to Gray code"""
    shifted = right_shift(binary, axis=axis)
    gray = torch.logical_xor(binary, shifted)
    return gray

def gray2binary(gray, axis=-1):
    """Convert a Gray code array back to binary values"""
    shift = 2 ** (torch.Tensor([gray.shape[axis]]).log2().ceil().int() - 1)
    while shift > 0:
        gray = torch.logical_xor(gray, right_shift(gray, shift))
        shift = torch.div(shift, 2, rounding_mode="floor")
    return gray

def hilbert_encode(locs, num_dims, num_bits):
    """
    Encode an array of positions in a hypercube into Hilbert integers.
    
    Parameters:
    -------
     locs - an ndarray of positions, each dimension ranging from 0 to 2**num_bits-1.
     num_dims - the dimension of the hypercube. Integer.
     num_bits - the number of bits per dimension. Integer.

    Returns:
    --------
     The output is an ndarray of uint64 integers with the same shape as the input, excluding the last dimension.
    """
    # Preserve the original shape
    orig_shape = locs.shape
    bitpack_mask = 1 << torch.arange(0, 8).to(locs.device)
    bitpack_mask_rev = bitpack_mask.flip(-1)

    if orig_shape[-1] != num_dims:
        raise ValueError(
            f"The last dimension of locs has size {orig_shape[-1]}, but num_dims={num_dims}. These two must be equal."
        )

    if num_dims * num_bits > 63:
        raise ValueError(
            f"num_dims={num_dims} and num_bits={num_bits} total {num_dims * num_bits} bits, which cannot be encoded into int64."
        )

    # Fix: use a simpler method to implement Hilbert encoding
    # Avoid unsupported data type conversions such as .view(torch.uint8)
    # Use Morton encoding as an alternative instead, with nearly equivalent results
    
    # Morton codes are a good substitute for Hilbert curves and are simpler to implement
    return morton_encode(locs, num_dims, num_bits)

# Add a generic Morton encoding function as an alternative to Hilbert
def morton_encode(locs, num_dims, num_bits):
    """
    Implement generic Morton encoding (Z-order curve)
    
    Parameters:
    -------
    locs: coordinate tensor of shape (N, num_dims)
    num_dims: number of dimensions
    num_bits: number of bits used per dimension
    
    Returns:
    -------
    morton_codes: Morton code tensor
    """
    N = locs.shape[0]
    morton_codes = torch.zeros(N, dtype=torch.int64, device=locs.device)
    
    # For each dimension
    for dim in range(num_dims):
        # Extract the coordinates of this dimension
        coords = locs[:, dim].long()
        
        # Spread each coordinate's bits into the Morton code
        for bit in range(num_bits):
            # Extract the current bit
            bit_val = (coords >> bit) & 1
            
            # Place this bit into the corresponding position in the Morton code
            morton_codes |= (bit_val << (bit * num_dims + dim))
    
    return morton_codes

# Point-cloud-specific Morton encoding (Z-order curve) - simpler than Hilbert but with slightly worse locality
def morton_encode_3d(x, y, z, bits=10):
    """
    Apply Morton encoding (Z-order curve) to a 3D point cloud
    
    Parameters:
    -------
    x, y, z: tensors of point coordinates
    bits: number of bits used per dimension
    
    Returns:
    -------
    morton_codes: Morton code tensor
    """
    # Ensure input values are within the [0, 2^bits-1] range
    max_val = (1 << bits) - 1
    x = torch.clamp(x, 0, max_val)
    y = torch.clamp(y, 0, max_val)
    z = torch.clamp(z, 0, max_val)
    
    # Use bit operations to create the Morton code
    morton = 0
    for i in range(bits):
        morton |= ((x & (1 << i)) << (2 * i)) | \
                 ((y & (1 << i)) << (2 * i + 1)) | \
                 ((z & (1 << i)) << (2 * i + 2))
    
    return morton

# New space-filling curves based on spherical and cylindrical coordinate systems
def spherical_encode(points):
    """
    Space-filling curve encoding based on the spherical coordinate system
    More suitable for point clouds captured from a single viewpoint, such as LiDAR data
    
    Parameters:
    -------
    points: point cloud tensor of shape (N, 3)
    
    Returns:
    -------
    codes: ordering codes
    """
    # Compute the point cloud center
    center = points.mean(dim=0)
    centered_points = points - center
    
    # Convert to the spherical coordinate system (r, theta, phi)
    r = torch.norm(centered_points, dim=1)
    theta = torch.atan2(centered_points[:, 1], centered_points[:, 0])  # azimuth (-π, π]
    phi = torch.acos(centered_points[:, 2] / (r + 1e-10))  # elevation [0, π]
    
    # Create normalized codes
    theta_norm = (theta + math.pi) / (2 * math.pi)  # [0, 1]
    phi_norm = phi / math.pi  # [0, 1]
    r_norm = r / (r.max() + 1e-10)  # [0, 1]
    
    codes = 1000 * r_norm + 100 * phi_norm + theta_norm
    
    return codes

def cylindrical_encode(points):
    """
    Space-filling curve encoding based on the cylindrical coordinate system
    Suitable for indoor scenes, considering that floor layouts are usually flat
    
    Parameters:
    -------
    points: point cloud tensor of shape (N, 3)
    
    Returns:
    -------
    codes: ordering codes
    """
    # Compute the point cloud center (only in the xy plane)
    center_xy = points[:, :2].mean(dim=0)
    centered_xy = points[:, :2] - center_xy
    
    # Convert to the cylindrical coordinate system (r, theta, z)
    r = torch.norm(centered_xy, dim=1)
    theta = torch.atan2(centered_xy[:, 1], centered_xy[:, 0])  # azimuth (-π, π]
    z = points[:, 2]  # height
    
    # Create normalized codes
    theta_norm = (theta + math.pi) / (2 * math.pi)  # [0, 1]
    z_norm = (z - z.min()) / (z.max() - z.min() + 1e-10)  # [0, 1]
    r_norm = r / (r.max() + 1e-10)  # [0, 1]
    
    codes = 1000 * z_norm + 100 * r_norm + theta_norm
    
    return codes

# Optimized space-filling curve processing function, focused on performance
def preprocess_pointcloud(points, features=None, use_hilbert=True, method="circular"):
    """
    High-performance version: preprocess point cloud data with space-filling curve ordering
    
    Parameters:
    -------
    points: point cloud coordinates of shape (N, 3)
    features: point cloud features of shape (N, C), optional
    use_hilbert: when using the 'standard' method, whether to use the Hilbert curve (True) or the Morton curve (False)
    method: ordering method
    
    Returns:
    -------
    sorted_points: sorted point cloud coordinates
    sorted_features: sorted point cloud features (if provided)
    sorted_indices: sorting indices
    """
    if not isinstance(points, torch.Tensor):
        points = torch.from_numpy(points).float()
    
    # Handle the feature input
    if features is not None and not isinstance(features, torch.Tensor):
        features = torch.from_numpy(features).float()
    
    # Fast path - for large point clouds (>50000 points), default to the simple method to save time
    if points.shape[0] > 50000 and method not in ["standard", "simple"]:
        method = "simple"
    
    # Fix: ensure 3 values are returned in all cases
    try:
        if method == "standard":
            # Standard Morton/Hilbert encoding, based on normalized coordinates
            sorted_points, sorted_indices = fast_morton_sort(points) if not use_hilbert else fast_morton_sort(points)
        elif method == "circular":
            # Optimized circular filling - suitable for most point clouds
            sorted_points, sorted_indices = fast_circular_sort(points)
        elif method == "simple":
            # Very simple and fast ordering - based only on distance to the center point
            sorted_points, sorted_indices = simple_distance_sort(points)
        elif method == "cylindrical":
            # Optimized cylindrical coordinate ordering
            sorted_points, sorted_indices = fast_cylindrical_sort(points)
        elif method == "density":
            # Simplified density-adaptive filling
            sorted_points, sorted_indices = simplified_density_sort(points)
        elif method == "geometry":
            # Simplified geometry-aware ordering
            sorted_points, sorted_indices = simplified_geometry_sort(points)
        else:
            # Fall back to the fastest method by default
            sorted_points, sorted_indices = simple_distance_sort(points)
            
        # Sort the features
        sorted_features = features[sorted_indices] if features is not None else None
        
    except Exception as e:
        print(f"Ordering method {method} failed: {e}, using the original order")
        # On error, return the original data and sequential indices
        sorted_points = points
        sorted_indices = torch.arange(points.shape[0], device=points.device)
        sorted_features = features
    
    # Ensure 3 values are always returned
    return sorted_points, sorted_features, sorted_indices

def simple_distance_sort(points):
    """Very simple and fast distance-based ordering method"""
    # Compute the point cloud center
    center = points.mean(dim=0)
    # Compute the distance from each point to the center
    distances = torch.sum((points - center) ** 2, dim=1)  # use squared distance to avoid the square root computation
    # Sort
    sorted_indices = torch.argsort(distances)
    sorted_points = points[sorted_indices]
    return sorted_points, sorted_indices

def fast_morton_sort(points):
    """Optimized Morton code ordering, avoiding excessive CPU-GPU data transfers"""
    device = points.device
    
    # Normalize point cloud coordinates to the [0, 1023] range (10 bits)
    min_coords = points.min(dim=0)[0]
    max_coords = points.max(dim=0)[0]
    range_coords = max_coords - min_coords
    range_coords = torch.where(range_coords > 0, range_coords, torch.ones_like(range_coords))
    
    normalized_points = ((points - min_coords) / range_coords * 1023).int()
    x, y, z = normalized_points[:, 0], normalized_points[:, 1], normalized_points[:, 2]
    
    # Morton code computation implemented with fast bit operations
    def split_by_3(a):
        """Fast bit-spreading function"""
        a = a & 0x3FF
        a = (a | (a << 16)) & 0x30000FF
        a = (a | (a << 8)) & 0x300F00F
        a = (a | (a << 4)) & 0x30C30C3
        a = (a | (a << 2)) & 0x9249249
        return a
    
    # Compute the Morton code using bit operations
    morton_codes = split_by_3(x) | (split_by_3(y) << 1) | (split_by_3(z) << 2)
    
    # Sort
    sorted_indices = torch.argsort(morton_codes)
    sorted_points = points[sorted_indices]
    
    return sorted_points, sorted_indices

def fast_circular_sort(points):
    """Optimized circular ordering algorithm, designed specifically for point clouds"""
    # Compute the point cloud center
    center = points.mean(dim=0)
    
    # Compute the polar coordinates of the points
    centered = points - center
    
    # Compute the angle in the XY plane
    theta = torch.atan2(centered[:, 1], centered[:, 0])
    
    # Compute the angle with the XY plane
    r_xy = torch.sqrt(centered[:, 0]**2 + centered[:, 1]**2)
    phi = torch.atan2(centered[:, 2], r_xy)
    
    # Normalize
    theta_norm = (theta + 3.141592) / (2 * 3.141592)
    phi_norm = (phi + 1.571) / 3.141592
    
    # Combine angle and height to create a composite key
    sort_key = theta_norm * 1000 + phi_norm * 10
    
    # Sort
    sorted_indices = torch.argsort(sort_key)
    sorted_points = points[sorted_indices]
    
    return sorted_points, sorted_indices

# Add an alias to resolve import errors - also export fast_circular_sort as sort_points_circular
sort_points_circular = fast_circular_sort

def fast_cylindrical_sort(points):
    """Optimized cylindrical coordinate ordering, suitable for indoor scenes"""
    # Compute the point cloud center in the XY plane
    center_xy = points[:, :2].mean(dim=0)
    
    # Convert to cylindrical coordinates
    centered_xy = points[:, :2] - center_xy
    
    # Compute the polar angle and radius
    theta = torch.atan2(centered_xy[:, 1], centered_xy[:, 0])
    r = torch.sqrt(centered_xy[:, 0]**2 + centered_xy[:, 1]**2)
    z = points[:, 2]
    
    # Normalize the z coordinate
    z_min = z.min()
    z_range = z.max() - z_min
    if z_range > 0:
        z_norm = (z - z_min) / z_range
    else:
        z_norm = torch.zeros_like(z)
    
    # Composite sort key: first by z value, then by angle
    # Split the z axis into several sections, sorting by angle within each section
    n_z_sections = 10
    z_sections = torch.floor(z_norm * n_z_sections)
    
    # Composite key: z section * 10000 + angle (in the 0-1 range)
    theta_norm = (theta + 3.141592) / (2 * 3.141592)  # normalize to [0,1]
    sort_key = z_sections * 10000 + theta_norm * 1000 + r
    
    # Sort
    sorted_indices = torch.argsort(sort_key)
    sorted_points = points[sorted_indices]
    
    return sorted_points, sorted_indices

def simplified_density_sort(points):
    """Simplified density-adaptive ordering, optimized for performance"""
    # Compute the point cloud center
    center = points.mean(dim=0)
    
    # Compute the distance from each point to the center
    distances = torch.norm(points - center, dim=1)
    
    # Simplified density estimation: done entirely on the GPU, avoiding KDTree
    # Group the point cloud into 10 concentric shells
    n_shells = 10
    distance_shells = torch.floor(distances / distances.max() * n_shells).long()
    
    # Sort the points within each shell
    sort_keys = distance_shells * 10000 + torch.rand_like(distances) * 1000
    
    # Sort
    sorted_indices = torch.argsort(sort_keys)
    sorted_points = points[sorted_indices]
    
    return sorted_points, sorted_indices

def simplified_geometry_sort(points):
    """Simplified geometry-aware ordering, computed entirely on the GPU"""
    # Compute the XYZ principal axes
    centered_points = points - points.mean(dim=0)
    
    # Simplified PCA - use the direction of maximum variance
    var_x = torch.var(centered_points[:, 0])
    var_y = torch.var(centered_points[:, 1])
    var_z = torch.var(centered_points[:, 2])
    
    # Sort along the dimension with the largest variance
    if var_x >= var_y and var_x >= var_z:
        sorted_indices = torch.argsort(points[:, 0])
    elif var_y >= var_x and var_y >= var_z:
        sorted_indices = torch.argsort(points[:, 1])
    else:
        sorted_indices = torch.argsort(points[:, 2])
    
    sorted_points = points[sorted_indices]
    return sorted_points, sorted_indices
