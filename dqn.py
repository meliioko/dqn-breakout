import torch
import torch.nn as nn
import torch.optim as optim
from collections import deque
from tqdm import tqdm
import copy
import random
import numpy as np
import gym

class DQN(nn.Module):
    def __init__(self, action_size):
        super(DQN, self).__init__()

        self.conv1 = nn.Conv2d(4, 32, kernel_size=8, stride=4)  # Assuming input_shape is (channels, height, width)
        self.conv2 = nn.Conv2d(32, 64, kernel_size=4, stride=2)
        self.conv3 = nn.Conv2d(64, 64, kernel_size=3, stride=1)

        # Compute the size of the output of the last conv layer
        def conv2d_size_out(size, kernel_size=3, stride=1):
            return (size - (kernel_size - 1) - 1) // stride + 1

        convw = conv2d_size_out(conv2d_size_out(conv2d_size_out(84, 8, 4), 4, 2), 3, 1)
        convh = conv2d_size_out(conv2d_size_out(conv2d_size_out(84, 8, 4), 4, 2), 3, 1)
        linear_input_size = convw * convh * 64

        self.fc = nn.Sequential(
            nn.Flatten(),
            nn.Linear(linear_input_size, 512),
            nn.ReLU(),
            nn.Linear(512, action_size)
        )


    def forward(self, x):

        x = torch.relu(self.conv1(x))
        x = torch.relu(self.conv2(x))
        x = torch.relu(self.conv3(x))

        return self.fc(x)

class DuelingDQN(nn.Module):
    def __init__(self, action_size):
        super(DuelingDQN, self).__init__()

        # Convolutional layers are the same
        self.conv1 = nn.Conv2d(4, 32, kernel_size=8, stride=4)
        self.conv2 = nn.Conv2d(32, 64, kernel_size=4, stride=2)
        self.conv3 = nn.Conv2d(64, 64, kernel_size=3, stride=1)

        # Compute the size of the output of the last conv layer
        def conv2d_size_out(size, kernel_size=3, stride=1):
            return (size - (kernel_size - 1) - 1) // stride + 1

        convw = conv2d_size_out(conv2d_size_out(conv2d_size_out(84, 8, 4), 4, 2), 3, 1)
        convh = conv2d_size_out(conv2d_size_out(conv2d_size_out(84, 8, 4), 4, 2), 3, 1)
        linear_input_size = convw * convh * 64

        # Separate streams for V(s) and A(s, a)
        self.fc_value = nn.Sequential(
            nn.Flatten(),
            nn.Linear(linear_input_size, 512),
            nn.ReLU(),
            nn.Linear(512, 1)  # This outputs V(s)
        )

        self.fc_advantage = nn.Sequential(
            nn.Flatten(),
            nn.Linear(linear_input_size, 512),
            nn.ReLU(),
            nn.Linear(512, action_size)  # This outputs A(s, a)
        )

    def forward(self, x):
        x = torch.relu(self.conv1(x))
        x = torch.relu(self.conv2(x))
        x = torch.relu(self.conv3(x))

        value = self.fc_value(x)
        advantage = self.fc_advantage(x)

        # Combine value and advantage streams
        q_values = value + (advantage - advantage.mean(dim=1, keepdim=True))

        return q_values

class PrioritizedReplayBuffer:
    def __init__(self, size):
        self.buffer = np.empty((size), dtype=object)
        self.priorities = np.zeros((size))
        self.max_priority = 1.0
        self.size = size
        self.len = 0
        self.pos = 0
    
    def add(self, experience, priority=None):
        if priority is None:
            priority = self.max_priority

        self.buffer[self.pos] = experience
        self.priorities[self.pos] = priority

        if self.len < self.size:
            self.len += 1

        self.pos = (self.pos + 1) % self.size
    
    def sample(self, batch_size, alpha, beta):
        scaled_priorities = self.priorities ** alpha
        sample_probs = scaled_priorities / np.sum(scaled_priorities)
        sampled_indices = np.random.choice(len(self.buffer), batch_size, p=sample_probs)
        samples = [self.buffer[i] for i in sampled_indices]
        is_weights = (1 / self.len * sample_probs[sampled_indices]) ** beta
        is_weights /= is_weights.max()
        return samples, sampled_indices, is_weights
    
    def update_priorities(self, indices, errors, epsilon):
        for idx, error in zip(indices, errors):
            self.priorities[idx] = min(self.max_priority, (abs(error) + epsilon))

class ReplayBuffer:
    def __init__(self, size):
        self.buffer = deque([], size)

    def add(self, experiment):
        self.buffer.append(experiment)
    
    def sample(self, n):
        return random.sample(self.buffer, n)
    


def update_target_network(target, source):
    target.load_state_dict(source.state_dict())

def preprocess_state(state, device):
    return torch.tensor(np.asarray(state)).float().div(255).unsqueeze(0).to(device)  # Scales to [0,1]


def play_train(M, env, epsilon, epsilon_frames, epsilon_min, gamma, Q_weights=None, N=40000 ,max_step= 10000, explo_start=30000):
    """The main function to train the DQN

    Args:
        M (int): The number of frames to train on.
        env (gym.env): The environement to train on.
        epsilon (float): The exploration factor.
        epsilon_decay (float): The exploration factor decay.
        epsilon_min (float): The minimum of the exploration factor.
        gamma (float): The importance of future reward in the Q computation.
        Q_weights (torch.state_duct, optional): Previous state dict of the model, to continue the training. Defaults to None.
        D (deque, optional): _description_. Defaults to None.
        N (int, optional): Size if the memory replay buffer_. Defaults to 40000.
        max_step (int, optional): The maximum frames in one episode. Defaults to 10000.

    Returns:
        dict: A dict containing the loss list and rewards during training
    """
    # Training loop
    action_size = env.action_space.n  # Number of actions
    state_size = env.observation_space.shape[0]  # State size

    Q = DuelingDQN(action_size)
    if Q_weights is not None:
        Q.load_state_dict(torch.load(Q_weights))

    Q_hat = copy.deepcopy(Q)
    
    # Check if a GPU is available
    if torch.cuda.is_available():
        device = torch.device("cuda")
        print("Using GPU:", torch.cuda.get_device_name(0))
    else:
        device = torch.device("cpu")
        print("Using CPU")
    

    Q.to(device)
    Q_hat.to(device)

    optimizer = optim.Adam(Q.parameters(), lr=0.00025)
    criterion = nn.MSELoss()

    D = ReplayBuffer(N)

    frames = 0
    reward_list = []
    episode = 0

    last_update = explo_start
    last_save = explo_start

    pbar = tqdm(total=M)
    while(frames < M):
        total_reward = 0
        state = env.reset()[0]
        state, _, _, _, _ = env.step(1)

        state = preprocess_state(state, device)# Add batch dimension
        for _ in range(max_step):
            # Epsilon-greedy action selection
            if np.random.rand() <= epsilon:
                action = random.randrange(action_size)
            else:
                with torch.no_grad():  # No need to track gradients here
                    act_values = Q(state)
                    action = act_values.max(1)[1].item()  # Choose the action with the highest Q-value

            # Execute action in environment and observe next state and reward
            next_state, reward, done, _, _ = env.step(action)
            total_reward += reward
            frames += 1
            pbar.update(1)


            next_state = preprocess_state(next_state, device)

            # Store transition in D (experience replay buffer)
            D.add((state, action, reward, next_state, done))

            state = next_state

            # Check if the episode is done
            if done :
                episode += 1
                reward_list.append(total_reward)
                break



            # Train using a random minibatch from D
            if frames > explo_start:
                minibatch = D.sample(32)
                # Extract tensors from the minibatch
                states = torch.cat([s for s, a, r, ns, d in minibatch]).to(device)
                actions = torch.tensor([a for s, a, r, ns, d in minibatch], device=device).long()
                rewards = torch.tensor([r for s, a, r, ns, d in minibatch], device=device).float()
                next_states = torch.cat([ns for s, a, r, ns, d in minibatch]).to(device)
                dones = torch.tensor([d for s, a, r, ns, d in minibatch], device=device).float()


                # Compute Q values for current states
                Q_values = Q(states)
                # Select the Q value for the action taken, which are the ones we want to update
                Q_values = Q_values.gather(1, actions.unsqueeze(1)).squeeze(1)

                # Compute the Q values for next states using the target network
                with torch.no_grad():
                    next_state_values = Q_hat(next_states).max(1)[0]
                    # If done is true, we want to ignore the next state value
                    next_state_values[dones == 1] = 0.0
                    # Compute the target Q values
                    target_Q_values = rewards + (gamma * next_state_values)

                # Zero the parameter gradients
                optimizer.zero_grad()
                # Compute loss
                loss = criterion(Q_values, target_Q_values)
                loss = loss.mean()
                # Backward pass
                loss.backward()
                torch.nn.utils.clip_grad_norm_(Q.parameters(), max_norm=1.0)
                optimizer.step()

        # Update epsilon
        if epsilon > epsilon_min and frames > explo_start:
            epsilon = max(epsilon_min, 1 - (frames - explo_start) / epsilon_frames)

        # Update target network
        if frames - last_update > 10000:
            Q_hat.load_state_dict(Q.state_dict())
            last_update = frames

        if frames - last_save > 300000:
            torch.save(Q.state_dict(), f'models/Q_{version}_{frames}.pt')
            np.save(f'rewards/reward_{version}_{frames}.npy', np.asarray(reward_list))
            last_save = frames

    pbar.close()
    torch.save(Q.state_dict(), f'Q_{version}_final.pt')
    return reward_list


if __name__ == "__main__":
    version = 'DuelV1'
    env = gym.make("Breakout-v4", obs_type='grayscale', render_mode='rgb_array', full_action_space=False, frameskip=4)
    env = gym.wrappers.AtariPreprocessing(env=env, frame_skip=1, terminal_on_life_loss=True)
    env = gym.wrappers.FrameStack(env=env, num_stack=4)
    #env = gym.wrappers.RecordVideo(env, 'videos', episode_trigger= lambda x : x % 2000 == 0 and x > 300)

    reward_list = play_train(M=2000000, env=env, epsilon=0.1, epsilon_frames=100000, epsilon_min=0.1, gamma=0.99, Q_weights=None, N=45000 ,max_step=100000, explo_start=30000)
    np.save(f'rewards/reward_{version}_final.npy', np.asarray(reward_list))
