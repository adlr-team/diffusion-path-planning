from diffuser.utils.rendering import MazeRenderer

import gymnasium as gym
env = gym.make('PointMaze_Medium_Diverse_G-v3', render_mode='human')
print(env.maze.maze_map)

renderer = MazeRenderer(env)
observation, success = env.reset()
rgb_array = env.render()

print(observation)
print(success)
renderer.renders(observation["observation"].reshape(1, 4))

print(observation)
print(observation["observation"].reshape(1, 4))