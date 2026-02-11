import numpy as np

def apply_homogeneous_transform(T, points):
    """
    Apply 4x4 homogeneous transform T to 3D point(s).
    """
    # Your code here
    points = np.asarray(points)
    T = np.asarray(T)
    single_input = False

    if points.ndim == 1:
            points = points[None, :]
            single_input = True
            
    ones = np.ones((points.shape[0], 1), dtype=points.dtype)
    points_h = np.concatenate([points, ones], axis=1)
    transformed_h = points_h @ T.T
    transformed = transformed_h[:, :3]

    return transformed[0] if single_input else transformed