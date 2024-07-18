[Colab Link](https://colab.research.google.com/drive/152ADFBIZr6j0fMgLpPGvfBC0XEKoY76f?usp=sharing)

[Colab Christophe](https://colab.research.google.com/drive/1QDwWmu8kRxZFI6nRI6nbOu31eYmYEywE?usp=sharing)

## Getting started

To get started, follow these steps:

1. Install Poetry (only for the first execution):
    ```shell
    pip install poetry
    ```

2. Install project dependencies (only for the first execution):
    ```shell
    poetry install
    ```

3. Activate the Virtual Environment:
    ```shell
    poetry shell
    ```


4. Add Dependencies (Optional): If you need to add new dependencies to your project, you can use the following command:
    ```shell
    poetry add <package-name>
    ```

4. Update Dependencies (Optional): To update the dependencies to the latest versions according to the constraints in your pyproject.toml file, use:
    ```shell
    poetry update
    ```




    # Implementation Notes
- This is based on the maze2d branch from Janner
- I added  device = "cpu" to the config.py file, line 24 and 67. Fix them when you want CUDA
- In Maze2D branch, the temporal.py and helpers.py are the same as the main branch, but the diffusion.py is slightly different
- In the script folder, planmaze2d and train are relevant for us.
- setup.py has the Parser.
- Take arguments from config/maze2d.py without using the Parser
- Comment out gym in preprocessing.py
- Comment out d4rl and 23-25 and import gym at datasets/d4rl.py --> convert everthing to "import gymnasium as gym"
- diffuser_test_new has the new gym robotics environment, diffuser_test_new_wo_gym is the last working version that does not have gym. (pip install gymnasium-robotics)
- import gymnasium as gym (did not work on Windows) (Could not build wheels for mujoco, which is required to install pyproject.toml-based projects)
- Switched to Colab
- pip install mujoco and gymnasium
- arrays.py update the device: DEVICE = 'cuda:0' to DEVICE = 'cpu' for now
- Comment out #import mujoco_py as mjc in rendering.py! Do not install mujoco-py (a different thing from mujoco)
- Get rid of d4rl too (d4rl.py line 23-25 comment out)
- Changing the environment requires properties to be changed too. I get 'PointMazeEnv' object has no attribute 'maze_arr' and similar errors and they need to be modified according to the new library.
- I commented out the MujocoRenderer class inside rendering.py because we do not use it and it uses the old mujoco binding (mujoco-py)