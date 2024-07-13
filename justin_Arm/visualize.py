from random import sample
from turtle import color

import matplotlib

matplotlib.use("Agg")  # or 'Qt5Agg', depending on your system
import matplotlib.pyplot as plt
import numpy as np
from mpl_toolkits.mplot3d import Axes3D
from rokin import robots, vis

from justin_arm.helper import interpolate_trajectories


# First plot: 3D Path for the different joints over 20 Waypoints
# Create a colormap
def plot_trajectory_per_frames(q_paths):
    """
    Plots the trajectory of the 8 frames of JustinArm for a single path.

    Args:
        q_paths (ndarray): Array of joint configurations for a single path.
            Shape of the array should be (waypoints, n_joints).

    Returns:
        None
    """

    robot = robots.JustinArm07()
    frames = robot.get_frames(q_paths)
    # Get shape of frames array
    trajectory_length, frame_num, _, _ = frames.shape
    colors = plt.cm.jet(np.linspace(0, 1, trajectory_length))
    # First plot: Trajectory of the 8 Frames of JustinArm
    plt.ion()  # Enable interactive mode

    fig1 = plt.figure()
    ax1 = fig1.add_subplot(111, projection="3d")

    # Iterate over each robot (8 robots)
    for robot_idx in range(frame_num):
        # Initialize lists to store coordinates for each robot's end gripper path
        x_coords = []
        y_coords = []
        z_coords = []

        # Iterate over each waypoint (20 waypoints)
        for waypoint_idx in range(trajectory_length):
            # Extract the homogeneous matrix for the current robot and waypoint
            matrix = frames[waypoint_idx, robot_idx, :, :]
            # Extract the X, Y, Z coordinates (assuming they are in the last row and first three columns)
            x = matrix[0, 3]
            y = matrix[1, 3]
            z = matrix[2, 3]

            # Append the coordinates to the respective lists
            x_coords.append(x)
            y_coords.append(y)
            z_coords.append(z)

        # Plot the path for the current robot with black line and scatter points with color gradient
        ax1.plot(x_coords, y_coords, z_coords, zorder=5, label=f"Frame {robot_idx+1}")
        ax1.scatter(
            x_coords,
            y_coords,
            z_coords,
            c=colors,
            zorder=5,
            label=f"Frame {robot_idx+1}",
        )

    # Add labels and legend
    ax1.set_xlabel("X")
    ax1.set_ylabel("Y")
    ax1.set_zlabel("Z")
    ax1.set_title("Trajectory of the 8 Frames of JustinArm")


def plot_multiple_trajectories(q_paths, sampling_num):
    """
    Plots multiple trajectories of the end-effector given a sample number.

    Parameters:
        q_paths (ndarray): Array of joint configurations for each trajectory.
        sampling_num (int): Number of trajectories to sample and plot.

    Returns:
        None
    """

    # Define robot and colormap
    robot = robots.JustinArm07()
    colors = plt.cm.jet(np.linspace(0, 1, 20))

    # Get shape of q_paths array
    num_trajectories, waypoints, _ = q_paths.shape

    # Second plot: End-effector trajectory for multiple paths
    fig2 = plt.figure()
    ax2 = fig2.add_subplot(111, projection="3d")

    # Retrieve trajectory_num amount of samples from frames across the first dimension
    # Randomly sample indices
    random_indices = np.random.choice(num_trajectories, sampling_num, replace=False)
    # Create a new array with the sampled rows
    sampled_paths = q_paths[random_indices, :, :]

    # Iterate over each path (20 paths)
    for i in range(sampling_num):
        x_coords = []
        y_coords = []
        z_coords = []
        frames = robot.get_frames(sampled_paths[i])

        # Iterate over each waypoint
        for waypoint_idx in range(waypoints):
            # Extract the homogeneous matrix for the current robot and waypoint
            matrix = frames[waypoint_idx, 7, :, :]
            # Extract the X, Y, Z coordinates (assuming they are in the last row and first three columns)
            x = matrix[0, 3]
            y = matrix[1, 3]
            z = matrix[2, 3]

            # Append the coordinates to the respective lists
            x_coords.append(x)
            y_coords.append(y)
            z_coords.append(z)

        # Plot the path for the current robot with black line and scatter points with color gradient
        ax2.scatter(x_coords, y_coords, z_coords, c=colors, zorder=10)

    # Add labels and legend
    ax2.set_xlabel("X")
    ax2.set_ylabel("Y")
    ax2.set_zlabel("Z")
    ax2.set_title("End-effector trajectory for multiple paths")
    plt.show()


def plot_q_values_per_trajectory(q_paths):
    """
    Plot the Q-values over the waypoints for each joint in the robot.

    Args:
        q_paths (ndarray): Array of Q-values for each joint over different trajectories.
            Shape of the array should be (num_paths, waypoints, joints).

    Returns:
        None
    """
    # Third plot: Q-value plots for each joint
    fig3, axs = plt.subplots(4, 2, figsize=(12, 12))  # 4 rows, 2 columns (8 subplots)

    # Flatten the axs array for easy iteration, ignoring the last subplot
    axs = axs.flatten()
    if q_paths.ndim == 2:
        q_paths = np.expand_dims(q_paths, axis=0)

    num_paths, waypoints, joints = q_paths.shape

    # Create subplots for each joint
    for j in range(joints):
        ax = axs[j]
        ax.set_xlabel("Waypoints")
        ax.set_ylabel(f"Q value of Joint {j + 1}")
        for i in range(num_paths):
            ax.plot(q_paths[i, :, j], color="black", zorder=10)
            ax.scatter(
                range(q_paths.shape[1]),
                q_paths[i, :, j],
                c=plt.cm.jet(np.linspace(0, 1, waypoints)),
                zorder=10,
            )

    # Remove the 8th (extra) subplot
    fig3.delaxes(axs[-1])

    # Adjust layout to prevent overlap
    plt.tight_layout()

    # Show the third plot
    plt.show()


# # Load numpy ndarray:
# image_4123 = np.load("justin_arm/data/image_4123.npy")
# paths = np.load("justin_arm/data/q_paths_4123.npy")
# print(f"Paths:{paths.shape}")


# # Load robot
# robot = robots.JustinArm07()
# frames = robot.get_frames(paths[0])

# # Interpolate the trajectories
# paths = interpolate_trajectories(paths, 100)

# # Test the functions
# plot_trajectory_per_frames(paths[0])
# plot_q_values_per_trajectory(paths[:10])
# plot_multiple_trajectories(paths, 20)
