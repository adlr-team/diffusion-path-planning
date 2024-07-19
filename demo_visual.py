import os
import pickle
import sys

import numpy as np
from rokin import robots, vis
from wzk import sql2

from justin_arm.helper import analyze_distance, robot_env_dist
from justin_arm.visualize import (
    plot_multiple_trajectories,
    plot_q_values_per_trajectory,
    plot_trajectory_per_frames,
)

n_voxels = 64
world_shape = (n_voxels, n_voxels, n_voxels)
limits = np.array([[-1.25, +1.25], [-1.25, +1.25], [-1.25, +1.25]])

n_waypoints = 20  # start + 20 inner points + end
n_dim = 3
n_dof = 7
n_worlds = 12500

# Load data:
# pickle load:
raw_path = pickle.load(open("justin_arm/data/paths_raw_4123.pkl", "rb"))
world = pickle.load(open("justin_arm/data/world_4123.pkl", "rb"))


paths = np.load("justin_arm/data/q_paths_4123.npy")
image = np.load("justin_arm/data/image_4123.npy")


robot = robots.JustinArm07()

# Randomly sample a path from paths and visualize it:
# Select a random index along the first axis
random_index = np.random.choice(paths.shape[0])
# Extract the corresponding path
random_path = paths[random_index]


distance = robot_env_dist(q=random_path, robot=robot, img=image[0])
score = analyze_distance(distance)


# Plot the trajectory of the 8 frames of JustinArm
plot_multiple_trajectories(q_paths=random_path, sampling_num=1)
plot_q_values_per_trajectory(q_paths=random_path)


# alternative: three_pv - pyvista; mc meshcat
# vis.three_mc.animate_path(
#     robot=robot,
#     q=paths_4123[10],
#     kwargs_robot=dict(color="red", alpha=0.2),
#     kwargs_world=dict(img=image_4123[0], limits=limits, color="yellow"),
# )


vis.three_pv.animate_path(
    robot=robot,
    q=random_path,
    kwargs_robot=dict(color="red"),
    kwargs_world=dict(img=image[0], limits=limits, color="yellow"),
)


input()
# move through animation with arrow keys
