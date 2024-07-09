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


def create_state_action_array(q_values):
    # q_values is an array of the size (num_trajectories,n_waypoints, n_joints)
    # I want generate an array which for each trajectory stores the difference between the current waypoitn and the next waypoint  for a given joint of n_joints. the last row the action is zero:
    # num_trajectories,n_waypoints-1, n_joints
    num_trajectories, n_waypoints, n_joints = q_values.shape
    actions = np.zeros((num_trajectories, n_waypoints, n_joints))
    print(actions.shape)
    print(q_values.shape)
    for i in range(num_trajectories):
        for j in range(n_waypoints - 1):
            actions[i, j] = q_values[i, j + 1] - q_values[i, j]
    # Now we stack both stack them along the joints axis:
    # so the output size is (num_trajectories,n_waypoints, 2*n_joints)
    # Concatenate along the joints axis (axis=2)
    state_action = np.concatenate([q_values[:, :, :], actions], axis=2)
    return state_action


def condition_start_end(q_path):
    # retrieve the start and end condition of the trajecotry in the form of the q values of the 7th joint on the 0 and the last waypoint for a given trajectory:
    print(q_path.shape)
    # Start condition is the 7th joint of the first waypoint
    start_condition = q_path[0][:]
    # End condition is the 7th joint of the last waypoint
    end_condition = q_path[-1][:]

    # Now concatenate 7 additional 0 to every start and end:

    # TODO: do we only condition on the endgripper q-values or all of them?

    start_condition = np.concatenate([start_condition, np.zeros(7)])
    end_condition = np.concatenate([end_condition, np.zeros(7)])

    return start_condition, end_condition


# # Example usage
# data = np.load("justin_arm/q_paths_4123.npy")
# print(f"Data shape: {data.shape}")
# state_action = create_state_action_array(data)
# print(f"State action shape: {state_action.shape}")
# print(f"State action shape: {state_action[0].shape}")
# # get me the 20th of the bottom 7: (20, 14)
# print(f"State action: {state_action[0][19].shape}")
# print(f"State action: {state_action[0][19][7:]}")

# # target_length = 100
# # interpolated_data = interpolate_trajectories(data, target_length)

# # print(interpolated_data.shape)  # Should print (984, 100, 7)


# # Example usage
# start_condition, end_condition = condition_start_end(data[0])
# print()
# print(f"Start condition: {start_condition.shape}")
# print(f"End condition: {end_condition}")
