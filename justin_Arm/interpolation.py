import numpy as np
from scipy.interpolate import interp1d


def interpolate_trajectories(data, target_length):
    """
    Interpolates a numpy array from (X, Y, 7) to (X, target_length, 7).

    Parameters:
    data (numpy.ndarray): Input data of shape (X, Y, 7)
    target_length (int): The desired length of the interpolated trajectories

    Returns:
    numpy.ndarray: Interpolated data of shape (X, target_length, 7)
    """
    X, Y, joints = data.shape
    interpolated_data = np.zeros((X, target_length, joints))

    # Define the original and target time points
    original_time_points = np.linspace(0, 1, Y)
    target_time_points = np.linspace(0, 1, target_length)

    for i in range(X):
        for j in range(joints):
            # Interpolate for each trajectory and joint
            interpolator = interp1d(original_time_points, data[i, :, j], kind="linear")
            interpolated_data[i, :, j] = interpolator(target_time_points)

    return interpolated_data


# Example usage
data = np.load("q_paths_4123.npy")
target_length = 100
interpolated_data = interpolate_trajectories(data, target_length)

print(interpolated_data.shape)  # Should print (984, 100, 7)
