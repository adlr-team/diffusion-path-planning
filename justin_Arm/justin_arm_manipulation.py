import pickle

import numpy as np
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
raw_path_4123 = pickle.load(open("justin_arm/paths_raw_4123.pkl", "rb"))
world_4123 = pickle.load(open("justin_arm/world_4123.pkl", "rb"))


paths_4123 = np.load("justin_arm/q_paths_4123.npy")
image_4123 = np.load("justin_arm/image_4123.npy")


robot = robots.JustinArm07()
# alternative: three_pv - pyvista; mc meshcat
vis.three_mc.animate_path(
    robot=robot,
    q=paths_4123[10],
    kwargs_robot=dict(color="red", alpha=0.2),
    kwargs_world=dict(img=image_4123[0], limits=limits, color="yellow"),
)
input()

# vis.three_pv.animate_path(robot=robot, q=q_paths[0],
#                           kwargs_world=dict(img=obstacle_images[2], limits=limits))

# move through animation with arrow keys
