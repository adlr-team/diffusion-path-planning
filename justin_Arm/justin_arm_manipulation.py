import numpy as np
from rokin import robots, vis
from wzk import sql2
import pickle

n_voxels = 64
world_shape = (n_voxels, n_voxels, n_voxels)
limits = np.array([[-1.25, +1.25], [-1.25, +1.25], [-1.25, +1.25]])

n_waypoints = 20  # start + 20 inner points + end
n_dim = 3
n_dof = 7
n_worlds = 12500

# Load data:
# pickle load:
raw_path_4123 = pickle.load(open("paths_raw_4123.pkl", "rb"))
world_4123 = pickle.load(open("world_4123.pkl", "rb"))


paths_4123 = np.load("q_paths_4123.npy")
image_4123 = np.load("image_4123.npy")


robot = robots.JustinArm07()
# alternative: three_pv - pyvista; mc meshcat
vis.three_mc.animate_path(
    robot=robot,
    q=q_paths[1],
    kwargs_robot=dict(color="red", alpha=0.2),
    kwargs_world=dict(img=obstacle_images[2], limits=limits, color="yellow"),
)
input()

# vis.three_pv.animate_path(robot=robot, q=q_paths[0],
#                           kwargs_world=dict(img=obstacle_images[2], limits=limits))

# move through animation with arrow keys
