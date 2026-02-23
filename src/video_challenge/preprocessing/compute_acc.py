import numpy as np

from scipy.signal import savgol_filter

from src.video_challenge.preprocessing.interpolate import interpolate

def process_single_file(file_path, fps=30, window_length=9):
    """
    Reads a single .npy file, cleans NaNs, computes acceleration, and returns the acceleration array.

    args:
        file_path (str): Path to the input .npy file containing landmark data.
        fps (int): Frames per second of the original video, used to calculate time intervals.
        window_length (int): Length of the filter window for Savitzky-Golay. Must be odd and >= polyorder + 2.
    
    returns:
        np.ndarray: A 3D array of shape (150, 33, 3) containing the computed acceleration for each landmark.
    """
    dt = 1.0 / fps
    
    # Load Data -> (150, 33, 5)
    lmk_arr = np.load(file_path)
    
    # Slice to keep only x, y, z -> (150, 33, 3)
    xyz_data = lmk_arr[:, :, :3]
    
    # Handle NaNs via interpolation
    clean_xyz = interpolate(xyz_data)
    
    # Compute Acceleration (Savitzky-Golay)
    acceleration = savgol_filter(
        clean_xyz, 
        window_length=window_length,
        polyorder=2, 
        deriv=2, 
        delta=dt, 
        axis=0,
        mode='nearest'
    )

    return acceleration