import pickle

import numpy as np
from helper import analyze_distance, robot_env_dist
from rokin import robots, vis
from wzk import sql2

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


distance = robot_env_dist(q=paths[1], robot=robot, img=image[0])
score = analyze_distance(distance)


# alternative: three_pv - pyvista; mc meshcat
# vis.three_mc.animate_path(
#     robot=robot,
#     q=paths_4123[10],
#     kwargs_robot=dict(color="red", alpha=0.2),
#     kwargs_world=dict(img=image_4123[0], limits=limits, color="yellow"),
# )


vis.three_pv.animate_path(
    robot=robot,
    q=paths[1],
    kwargs_robot=dict(color="red"),
    kwargs_world=dict(img=image[0], limits=limits, color="yellow"),
)


input()
# move through animation with arrow keys
