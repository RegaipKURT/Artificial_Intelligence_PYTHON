import gym
import numpy as np
import random
import matplotlib.pyplot as plt
import time

#Environments
env = gym.make("Taxi-v2").env

states = env.observation_space.n
actions = env.action_space.n

#Q-TABLE
q_table = np.zeros([states, actions])

#Hyperparameters
alpha = 0.1
gama = 0.9
epsilon = 0.1

#plotting metrics
reward_list = []
dropout_list = []

#episodes
episode_number = 10000
for i in range(1, episode_number):
    #initialize environments
    state = env.reset()
    
    reward_count = 0
    dropouts = 0

    while True:
        
        #exploit and explore / to find action (according to epsilon)
        if random.uniform(0,1) < epsilon:
            action = env.action_space.sample()

        else:
            action = np.argmax(q_table[state])
        

        #take an action, process and take reward / observation
        next_state, reward, done, _ = env.step(action)
        
        #q-learning function
        old_value = q_table[state,action]
        next_max = np.max(q_table[next_state])

        next_value = (1-alpha)*old_value + alpha*(reward+gama*next_max)


        #update q-table
        q_table[state,action] = next_value

        #update state
        state = next_state
        
        #find number of wrong dropouts
        if reward == -10:
            dropouts = dropouts + 1


        if done:
            break

        reward_count += reward
        """
        #harita üzerinde hareketleri görmek için burayı aktif edin.
        #viualize actions on map
        env.render()
        time.sleep(0.1)
        """
    if i %10 == 0:
        #10 turda bir yazdıralım ekrana sonuçları
        #reward eksi çıkıyor hala, bunun sebebi zaman kaybı, yani fazla dolaşma
        dropout_list.append(dropouts)
        reward_list.append(reward_count)
        print("Episode:{}, Reward: {}, Wrong Dropout: {}".format(i, reward_count, dropouts))

    
    

fig, axs = plt.subplots(1,2)

axs[0].plot(reward_list)
axs[0].set_xlabel("Episode")
axs[0].set_ylabel("Reward")

axs[1].plot(dropout_list)
axs[1].set_xlabel("Episode")
axs[1].set_ylabel("Number of Wrong Dropouts")

axs[0].grid(True)
axs[1].grid(True)

plt.savefig("Egitim_grafik.png")
plt.show()
