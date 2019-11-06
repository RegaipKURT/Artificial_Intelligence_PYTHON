import gym

env = gym.make("Taxi-v2").env

"""
blue= passenger
purple=destination
yellow=empty taxi
green = full taxi
"""

env.reset() #reset and return new environment
time_step = 0
total_reward = 0
list_visualize = []
total_list = []

for i in range(5):
    while True:

        time_step +=1

        #choose action
        action = env.action_space.sample()

        #perform action and get reward
        state, reward, done, info = env.step(action)
        
        #total reward
        total_reward += reward

        #print and visualize
        list_visualize.append({"frame":env.render(mode="ansi"),
        "state": state, "action": action, "reward":reward, "total_reward": total_reward
        })
        
        if done:
            total_list.append(total_reward)
            break
    
import time

for i, frame in enumerate(list_visualize):
    print(frame["frame"])
    print("Time  Step:", i+1)
    print("State:", frame["state"])
    print("Action:", frame["action"])
    print("Reward:", frame["reward"])
    print("Total Reward", frame["total_reward"])
    #time.sleep(2)

print(total_list)