import torch
import torch.nn.functional as F


def bilinear_sampling(image, grid):
    """
    Perform bilinear sampling on the image using the provided grid.

    Args:
        image (torch.Tensor): Input image tensor with shape (B, C, H, W).
        grid (torch.Tensor): Sampling grid with shape (B, Z, X, 2) where the last dimension
                             represents the normalized 2D coordinates.

    Returns:
        torch.Tensor: BEV feature map with shape (B, C, Z, X).
    """
    # Normalize grid values to be in the range [-1, 1] for F.grid_sample
    B, Z, X, _ = grid.shape
    grid = grid.view(B, Z * X, 2)
    grid = 2.0 * grid - 1.0  # assuming grid values are in [0, 1]
    grid = grid.view(B, Z, X, 2)

    # Perform bilinear sampling
    bev_features = F.grid_sample(image, grid, align_corners=True)

    return bev_features


def create_sampling_grid(K, T, Z, X, bev_resolution, bev_start_position):
    """
    Create a sampling grid to project 3D coordinates into 2D image plane.

    Args:
        K (torch.Tensor): Camera intrinsic matrix with shape (3, 3).
        T (torch.Tensor): Transformation matrix from world coordinates to camera coordinates with shape (4, 4).
        Z (int): Number of grid points along the Z-axis.
        X (int): Number of grid points along the X-axis.
        bev_resolution (tuple): Resolution of the BEV grid (Z_resolution, X_resolution).
        bev_start_position (tuple): Start position of the BEV grid (Z_start, X_start).

    Returns:
        torch.Tensor: Sampling grid with shape (1, Z, X, 2) for one image.
    """
    # Create 3D grid coordinates
    z_range = torch.arange(0, Z * bev_resolution[0], bev_resolution[0]) + bev_start_position[0]
    x_range = torch.arange(0, X * bev_resolution[1], bev_resolution[1]) + bev_start_position[1]
    z, x = torch.meshgrid(z_range, x_range)
    ones = torch.ones_like(z)
    grid_3d = torch.stack([x, ones, z, ones], dim=-1).view(-1, 4).T  # Shape: (4, Z*X)

    # Transform to camera coordinates
    grid_3d_camera = torch.matmul(T, grid_3d).T  # Shape: (Z*X, 4)
    grid_3d_camera = grid_3d_camera[:, :3]  # Ignore homogeneous coordinate

    # Project to 2D image plane
    grid_2d = torch.matmul(K, grid_3d_camera.T).T  # Shape: (Z*X, 3)
    grid_2d = grid_2d[:, :2] / grid_2d[:, 2:]  # Normalize by depth

    # Normalize to [0, 1] for grid_sample
    H, W = image.shape[2:]
    grid_2d[:, 0] /= W
    grid_2d[:, 1] /= H

    return grid_2d.view(1, Z, X, 2)


# Example usage
B, C, H, W = 1, 3, 256, 256  # Example dimensions
Z, X = 128, 128  # BEV dimensions
bev_resolution = (0.5, 0.5)  # Example resolution
bev_start_position = (0, 0)  # Example start position

image = torch.randn(B, C, H, W)  # Example input image
K = torch.tensor([[1000, 0, W / 2], [0, 1000, H / 2], [0, 0, 1]])  # Example intrinsic matrix
T = torch.eye(4)  # Example transformation matrix

grid = create_sampling_grid(K, T, Z, X, bev_resolution, bev_start_position)
bev_features = bilinear_sampling(image, grid)

print(bev_features.shape)  # Should be (B, C, Z, X)
