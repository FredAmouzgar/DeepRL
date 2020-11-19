from os import system, name
from time import sleep

try:
    import gym
except:
    system('pip install gym')
    import gym

def clear():
    # for windows
    if name == 'nt':
        _ = system('cls')

    # for mac and linux(here, os.name is 'posix')
    else:
        _ = system('clear')

env = gym.make("Taxi-v3")
_ = env.seed(40)
actions = {"s":0, "n":1, "e":2, "w":3, "p":4, "d":5}
s, r, done, info = None, None, False, None
s = env.reset()
print("State:", s)

print("Init state:", s)
while not done:
    clear()
    print(f"State:{s} - Reward:{r} - done: {done} - Info: {info}")
    print("Info on how the state was calculated: https://github.com/openai/gym/blob/c4d0af393ef9fba641bd3ebbbf1f60d291c8475d/gym/envs/toy_text/taxi.py#L128")
    env.render()
    print("Passanger: Blue Letter, Destination: Purple Letter")
    print("s: South - n: North\ne: East - w: West\np: Pickup - d: Dropoff\nexit: Terminate the game")
    a = input("Input an action (of 6 possible actions): ")
    if a == "exit": exit(1)
    try:
        a = actions[a]
    except:
        print("**** Invalid Action.\n")
        sleep(2)
        continue
    s, r, done, info = env.step(a)

print(f"Final State:{s} - Final Reward:{r} - done: {done}")
