from IPython.display import clear_output
import matplotlib.pyplot as plt
from envs import WindyGridWorld
from agents import Sarsa_Agent
from time import sleep
from tqdm import tqdm
import numpy as np

def train_windygridworld(env, agent, episodes=150):
    if env is None:
        env = WindyGridWorld()
    if agent is None:
        agent = Sarsa_Agent(env.states_n, env.actions_n,epsilon_decay=False)

    steps = []
    returns = []
    for episode in range(episodes):
        env.reset()
        state = env.state
        action = agent.act(state)
        done = False
        step_n = 0
        return_episode = 0
        while not done:
            new_state,reward,done = env.step(action)
            return_episode += reward
            new_action = agent.update(new_state,reward,state,action,done)
            state, action = new_state, new_action
            step_n += 1
            if done:
                steps.append(step_n)
                returns.append(return_episode)
                clear_output(wait=True)
                plt.title("Steps:" + str(step_n) + " Return:"+str(return_episode))
                plt.plot(list(range(len(steps))),steps)
                plt.plot(list(range(len(steps))),returns)
                plt.legend(["Steps", "Returns"])
                plt.show()

def train_gym(env, agent, episodes=150):
    if env is None:
        raise ValueError("No Environment is given.")
    if agent is None:
        raise ValueError("No agent is given.")

    steps = []
    returns = []
    for episode in range(episodes):
        state = env.reset()
        action = agent.act(state)
        done = False
        step_n = 0
        return_episode = 0
        while not done:
            new_state, reward, done, _ = env.step(action)
            return_episode += reward
            new_action = agent.update(new_state,reward,state,action,done)
            state, action = new_state, new_action
            step_n += 1
            if done:
                steps.append(step_n)
                returns.append(return_episode)
                clear_output(wait=True)
                plt.title("Steps:" + str(step_n) + " Return:"+str(return_episode))
                plt.plot(list(range(len(steps))),steps)
                plt.plot(list(range(len(steps))),returns)
                plt.legend(["Steps", "Returns"])
                plt.show()


def play(env, agent, episodes=2):
    for episode in range(episodes):
        env.reset()
        state = env.state
        action = agent.act(state)
        done = False
        step = 0
        actions = [action]
        while not done:
            new_state, reward, done = env.step(action)
            step += 1
            new_action = agent.update(new_state, reward, state, action, done)
            actions.append(new_action)
            state, action = new_state, new_action
            env.render()
            if done:
                print(f"Done in {step} steps.")
                print(f"Actions: {actions}")
                for a in actions:
                    print(env.actions[a], end=", ")
                print()
                sleep(8)
            sleep(0.1)
            clear_output(wait=True)

def play_taxi(env, agent, passengers=2, wait_btw_frames=1):
    for episode in range(passengers):
        state = env.reset()
        frames = []
        done = False
        step = 0
        while not done:
            action = agent.act(state)
            new_state, reward, done, _ = env.step(action)
            frames.append({
                'frame': env.render(mode='ansi'),
                'state': state,
                'action': action,
                'reward': reward
            })
            step += 1
            state = new_state
        taxi_print_frames(frames, wait_btw_frames=wait_btw_frames, episode=episode)


def taxi_print_frames(frames, wait_btw_frames, episode):
    for i, frame in enumerate(frames):
        clear_output(wait=True)
        print(frame['frame'])
        print(f"Passenger #: {episode + 1}")
        print("-----------")
        print(f"Timestep: {i + 1}")
        print(f"State: {frame['state']}")
        print(f"Action: {frame['action']}")
        print(f"Reward: {frame['reward']}")
        sleep(wait_btw_frames)

def gym_render(img, status=False, rewards=[], reward_chart=True):
    message = "Episode over" if status else ""
    if reward_chart:
        fig, (ax1, ax2) = plt.subplots(1, 2)
    else:
        fig, ax1 = plt.subplots(1, 1)
    ax1.axis("off")
    _ = ax1.imshow(img)
    if reward_chart:
        _ = ax2.plot(rewards)

    plt.title(message)
    plt.show()
    clear_output(wait=True)

def play_gym(env, agent, episodes=2):
    all_rewards = []
    for i in tqdm(range(episodes)):
        state = env.reset()
        agent.new_episode()
        done = False

        rewards = []
        while not done:
            # TODO: Agent has to choose an action here
            action = agent.act(state)
            state_, reward, done, _ = env.step(action)
            rewards.append(reward)
            # agent.learn(state, action,state_,reward,done)
            gym_render(env.render(mode="rgb_array"), status=done, rewards=all_rewards, reward_chart=False)
            state = state_
            if done:
                all_rewards.append(np.sum(rewards))
    env.close()

def play_gym2(env, agent, episodes=2):
    for episode in range(episodes):
        state = env.reset()
        done = False
        step = 0
        while not done:
            new_state, reward, done, _ = env.step(agent.act(state))
            step += 1
            if step % 2 == 0:
                fig, ax = plt.subplots(figsize=(12, 12))
                ax.imshow(env.render(mode="rgb_array"))
                plt.show()
                sleep(0.1)
                clear_output(wait=True)
