import numpy as np
import torch
from scipy.interpolate import interp1d

from justin_arm.obstacle_distance import img2dist_img, img2interpolation_fun


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


def create_state_action_array(q_values):
    """
    Generate an array that stores the difference between the current waypoint and the next waypoint for each joint.

    Parameters:
    q_values (numpy.ndarray): An array of shape (n_waypoints, n_joints) containing the waypoints for each joint.

    Returns:
    numpy.ndarray: An array of shape (n_waypoints, n_joints) containing the actions (differences) between waypoints.
    """
    n_waypoints, n_joints = q_values.shape
    actions = np.zeros((n_waypoints, n_joints))

    for i in range(n_waypoints - 1):
        actions[i, :] = q_values[i + 1, :] - q_values[i, :]

    # The last row remains zero as initialized

    state_action = np.concatenate([q_values, actions], axis=-1)
    # print("State action shape: ", state_action.shape)
    # print("Actions shape: ", actions.shape)
    return (
        torch.tensor(state_action, dtype=torch.float32),
        torch.tensor(actions, dtype=torch.float32),
    )


def condition_start_end_per_trajectory(q_path):

    # Store the dimensions of q_path in here
    n_waypoints, n_joints = q_path.shape
    # Start condition is the 7th joint of the first waypoint
    start_condition = q_path[0][:]
    # End condition is the 7th joint of the last waypoint
    end_condition = q_path[-1][:]
    return {
        0: torch.tensor(start_condition, dtype=torch.float32),
        n_waypoints - 1: torch.tensor(end_condition, dtype=torch.float32),
    }


def robot_env_dist(q, robot, img, n_voxels=64):
    voxel_size = 2.5 / n_voxels
    limits = np.array([[-1.25, +1.25], [-1.25, +1.25], [-1.25, +1.25]])
    dimg = img2dist_img(img=img, voxel_size=voxel_size, add_boundary=False)
    dist_fun = img2interpolation_fun(img=dimg, limits=limits, order=0)
    x_spheres = robot.get_spheres(q=q)
    return dist_fun(x=x_spheres) - robot.spheres.r


def analyze_distance(distance):
    # Show the shape of the ndarray:
    print("++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++")
    # Show true if the ndarray contains negative values:
    print(f"np.any(distance < 0): {np.any(distance < 0)}")
    # Show the negative values:
    print(f"distance[distance < 0]: {distance[distance < 0]}")
    # Show the number of negative values:
    print(f"np.sum(distance < 0): {np.sum(distance < 0)}")
    # Retrieve the indices within the df of the entries where the distance is negative:
    print(f"np.where(distance < 0): {np.where(distance < 0)}")

    # Add up the values of the entries which are negative:
    print(f"np.sum(distance[distance < 0]): {np.sum(distance[distance < 0])}")
    print("++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++")

    if np.any(distance < 0):
        collision_score = np.sum(distance[distance < 0])
    else:
        collision_score = 0
    print(f"Collision_score: {collision_score}")
    return collision_score
