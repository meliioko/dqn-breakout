import torch
from dqn import DQN, preprocess_state
import gym
import random
import numpy as np
import os
from tqdm import tqdm

def test(model, nb_episode, name, epsilon=0.1):
    device = torch.device('cuda')
    env = gym.make("Breakout-v4", obs_type='grayscale', render_mode='rgb_array', full_action_space=False, frameskip=4)
    env = gym.wrappers.AtariPreprocessing(env=env, frame_skip=1, terminal_on_life_loss=False, noop_max=30)
    env = gym.wrappers.FrameStack(env=env, num_stack=4)
    env = gym.wrappers.RecordVideo(env, f'test/videos_{name}', episode_trigger=lambda x : x % 50 == 0)

    device = torch.device("cuda")
    Q = DQN(env.action_space.n)
    Q.load_state_dict(torch.load(model))
    Q.to(device)

    state = preprocess_state(env.reset()[0], device)
    done = False
    total_reward = 0
    state, reward, done, _, b = env.step(action=1)


    state = preprocess_state(state, device)
    rewards = []
    episode = 0
    pbar = tqdm(total=nb_episode)
    while episode < nb_episode:
        state = preprocess_state(env.reset()[0], device)
        done = False
        total_reward = 0
        state, reward, done, _, b = env.step(action=1)
        state = preprocess_state(state, device)

        while not done:
            if random.random() < epsilon:
                action = random.randrange(env.action_space.n)
            else:
                with torch.no_grad():
                    act_values = Q(state)
                    action = act_values.max(1)[1].item()
            
            state, reward, done, _, _ = env.step(action=action)
            total_reward += reward
            if done:
                rewards.append(total_reward)
                pbar.update(1)
                episode += 1
                break                
            state = preprocess_state(state=state, device=device)

    np.save(f'test/{name}_rewards.npy', np.asarray(rewards))

if __name__ == '__main__':
    for folder in os.listdir('results'):
        test(f'results/{folder}/Q.pt', 100, folder, epsilon=0.1)