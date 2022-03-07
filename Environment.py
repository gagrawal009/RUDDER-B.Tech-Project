import numpy as np
import torch
import tqdm
from torch.utils.data import Dataset
from matplotlib import pyplot as plt
from widis_lstm_tools.nn import LSTMLayer

"""
2D Grid Environment 

Default-
Grid Size: 13x13
Reward position: (9,9)
Initial position: (7,7)        
"""

class Environment2D(Dataset):
    def __init__(self, n_samples: int, max_timestep: int, n_positions: int, rnd_gen: np.random.RandomState):
        """Our simple 1D environment as PyTorch Dataset"""
        super(Environment2D, self).__init__()
        n_dims = 2
        n_actions = 4

        zero_position = (int(np.ceil(n_positions / 2.)), (int(np.ceil(n_positions / 2.))))
        coin_position = (zero_position[0] + 2, zero_position[1] + 2)

        # Generate random action sequences
        actions = np.asarray(rnd_gen.randint(low=0, high=n_actions, size=(n_samples, max_timestep)), dtype=np.int)
        actions_onehot = np.identity(n_actions, dtype=np.float32)[actions]

        # Generate observations from action sequences
        # actions[:] = (actions * 2) - 1

        observations = np.full(fill_value=zero_position[0], shape=(n_samples, max_timestep, n_dims), dtype=np.int)
        observation_to_onehot = np.full(fill_value=zero_position[0], shape=(n_samples, max_timestep), dtype=np.int)

        for t in range(max_timestep-1):
            action = get_actions(actions, n_dims, t)
            for ac in range(n_dims):
              observations[:, t+1, ac] = np.clip(observations[:, t, ac] + action[ac], 0, n_positions-1) 

            observation_to_onehot[:, t+1] = observations[:,t+1,0]*n_positions + observations[:,t+1,1] 

        observations_onehot = np.identity(n_positions**n_dims, dtype=np.float32)[observation_to_onehot]

        # Calculate rewards (sum over coin position for all timesteps)
        rewards = np.zeros(shape=(n_samples, max_timestep), dtype=np.float32)
        rewards[:, -1] = observations_onehot[:, :, get_key(coin_position, n_positions)].sum(axis=1)

        self.actions = actions_onehot
        self.observations = observations_onehot
        self.rewards = rewards

    def __len__(self):
        return self.rewards.shape[0]
    
    def __getitem__(self, idx):
        return self.observations[idx], self.actions[idx], self.rewards[idx]


def get_actions(action_arr, n_dims, t):
  action_list = []
  var = action_arr[:,t]
  for i in range(n_dims):
    temp = np.where(var==2*i, -1, np.where(var==2*i+1, 1, 0))
    action_list.append(temp) 
  return action_list

def get_key(state, pivot):
  return state[0]*pivot + state[1]

def get_coord(key, pivot):
  return (int(state/pivot), state%pivot)



n_positions = 13
env = Environment2D(n_samples=1000, max_timestep=50, n_positions=13, rnd_gen=rnd_gen)
env_loader = torch.utils.data.DataLoader(env, batch_size=8, num_workers=4)








# Prepare some random generators for later
rnd_gen = np.random.RandomState(seed=123)
_ = torch.manual_seed(123)
