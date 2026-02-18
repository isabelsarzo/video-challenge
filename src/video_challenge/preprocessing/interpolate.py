import numpy as np

def interpolate(data_3d):
    """
    Interpolates missing (NaN) values along the time axis (frames).
    data_3d shape: (frames, joints, coordinates)
    """
    filled_data = np.copy(data_3d)
    _, n_joints, n_coords = filled_data.shape
    
    for j in range(n_joints):
        for c in range(n_coords):
            y = filled_data[:, j, c]
            nans = np.isnan(y)
            
            # If there are NaNs, but not ALL are NaNs, interpolate
            if np.any(nans) and not np.all(nans):
                # Get indices of NaNs and valid values
                nan_indices = nans.nonzero()[0]
                valid_indices = (~nans).nonzero()[0]
                
                # Linearly interpolate missing frames
                filled_data[nans, j, c] = np.interp(nan_indices, valid_indices, y[~nans])
            
            # If the entire array is NaN (joint never detected), fill with zeros to prevent math errors
            elif np.all(nans):
                filled_data[:, j, c] = 0.0
                
    return filled_data