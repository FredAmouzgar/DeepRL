from IPython.display import clear_output
import matplotlib.pyplot as plt
from envs import WindyGridWorld
from agents import Sarsa_Agent
from time import sleep

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

def play_gym(env, agent, episodes=2):
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
                sleep(0.01)
                clear_output(wait=True)