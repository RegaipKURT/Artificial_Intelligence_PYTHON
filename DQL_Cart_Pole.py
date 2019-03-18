#%%
#Deep Q Learning Algoritması ile gym kütüphanesinden caart pole oyunu.
import gym
import numpy as np
from collections import deque
from keras import Sequential
from keras.layers import Dense
from keras.optimizers import Adam
import random

class DQLAgent:
    
    def __init__(self, env):
        #hyperparameters
        self.state_size = env.observation_space.shape[0]
        self.action_size = env.action_space.n
        
        self.gama = 0.95
        self.learning_rate = 0.001

        self.epsilon = 1
        self.epsilon_decay = 0.995
        self.epsilon_min = 0.01

        self.memory = deque(maxlen=1000)

        self.model = self.built_model()

    def built_model(self):
        #neural network par for deep q learning
        
        model = Sequential()
        model.add(Dense(48, input_dim=self.state_size, activation="tanh"))
        model.add(Dense(self.action_size, activation="linear"))
        model.compile(loss="mse", optimizer=Adam(lr=self.learning_rate))
        return model

    def remember(self, state, action, reward, next_state, done):
        #storage
        self.memory.append((state, action, reward, next_state, done))

    def act(self, state):
        #acting according to state (and explore or explode)
        if random.uniform(0,1) <= self.epsilon:           
            return env.action_space.sample()
        else:
            act_values = self.model.predict(state)
            return np.argmax(act_values[0])

    def replay(self, batch_size):
        #training
        if len(self.memory) < batch_size:
            return
        minibatch = random.sample(self.memory, batch_size)
        for state, action, reward, next_state, done in minibatch:
            
            if done:
                target = reward
            else:
                target = reward + self.gama*np.amax(self.model.predict(next_state)[0])
            
            train_target = self.model.predict(state)
            train_target[0][action] = target

            self.model.fit(state, train_target, verbose=0)

            #modelimizi her eğittiğimizde kaydediyoruz ki daha sonra rahat kullanalım
            #self.model.save("trained_model.h5",overwrite=True, include_optimizer=True)

    def Adaptive_EGreedy(self):
        #adjust epsilon value
        if self.epsilon > self.epsilon_min:
            self.epsilon *= self.epsilon_decay
        else:
            pass


if __name__ == "__main__":

    #initialize env and agent
    env = gym.make("CartPole-v0")
    agent = DQLAgent(env)
    episode = 50
    batch_size = 32

    for e in range(episode):
        
        #initialize environment
        state = env.reset()    

        state = np.reshape(state, [1,4])

        time = 0
        
        while True:
            #act
            action = agent.act(state)

            #step
            next_state, reward, done, _ = env.step(action)
            next_state = np.reshape(next_state, [1,4])

            #remember / storage
            agent.remember(state, action, reward, next_state, done)
            
            #update state
            state = next_state

            #replay
            agent.replay(batch_size)
            
            #epsilon adjust
            agent.Adaptive_EGreedy()
            
            time +=1

            if done:
                print("Episode: {}, Time: {}".format(e, time))
                break
        
#%%    test      
import time
#from keras.models import load_model

trained_model = agent

a=0
while a < 20:
    state = env.reset()
    state = np.reshape(state, [1,4])
    time_t = 0
    while True:
        env.render()
        action = trained_model.act(state)
        next_state, reward, done, _  = env.step(action)
        next_state = np.reshape(next_state, [1,4])
        state = next_state
        time_t +=1
        print(time_t)
        time.sleep(0.01)
        if done:
            break

    print("Done")
    a += 1

