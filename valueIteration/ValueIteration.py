from os import terminal_size
import matplotlib

matplotlib.use("TkAgg")
import gym
import gridworld
from gym import wrappers, logger
import numpy as np
import copy
import torch
import numpy.random as rd
import random

class ValueIteration(object):
    """Value Iteration to be applied to th intelligent agent"""

    def __init__(self, env):
        self.action_space = env.action_space
        self.observation_space = env.observation_space
        self.statedic, self.mdp = env.getMDP()
        self.states = len(self.statedic)
        self.V = rd.randn(1, self.states).flatten()  # initialise value function
        self.p = random.choices([i for i in range(4)], k=self.states)  #initialise policy
        self.ind_dict = {}  # very important dictionary: allows us to associate the indices of the statedict with the indices of the mdp
        for i, k in enumerate(self.mdp):
            ind = self.statedic[k]
            self.ind_dict[ind] = i
        
        self.terminal_states = []
        for i, k in enumerate(self.statedic):
            if k not in self.mdp.keys():
                self.terminal_states.append(i)
        self.V[self.terminal_states] = 0

    def valueIteration(self, eps, gamma):
        delta = 0
        ind_dict = self.ind_dict 
        self.gamma = gamma  
        v = self.V 
        for s in range(self.states):
            # first iteration to define delta
            if s in self.terminal_states:
                pass
            else:
                sum = [0]*4
                state_index = ind_dict[s]  # this is the state_index in the mdp list associated with state s as defined in statedict
                _, transitions = list(self.mdp.items())[state_index]
                possibleActions = list(transitions.keys())
                for a in possibleActions:
                    newStatesLength = len(transitions[a])
                    newStates = [transitions[a][i][1] for i in range(newStatesLength)]
                    newStatesInt = [self.statedic[str(new_state)] for new_state in  newStates]
                    for i in range(newStatesLength):
                        new_state_int = newStatesInt[i]
                        sum[a] += transitions[a][i][0] * (transitions[a][i][2] + gamma*self.V[new_state_int])
                self.V[s] = np.max(sum)
            delta = np.linalg.norm(v-self.V)
        delta = np.linalg.norm(v-self.V)

        while delta >= eps:
            v = self.V 
            for s in range(self.states):
                if s in self.terminal_states:
                    pass
                else:
                    sum = [0]*4
                    state_index = ind_dict[s]  # this is the state_index in the mdp list associated with state s as defined in statedict
                    _, transitions = list(self.mdp.items())[state_index]
                    possibleActions = list(transitions.keys())
                    for a in possibleActions:
                        newStatesLength = len(transitions[a])
                        newStates = [transitions[a][i][1] for i in range(newStatesLength)]
                        newStatesInt = [self.statedic[str(new_state)] for new_state in  newStates]
                        for i in range(newStatesLength):
                            new_state_int = newStatesInt[i]
                            sum[a] += transitions[a][i][0] * (transitions[a][i][2] + gamma*self.V[new_state_int])
                self.V[s] = np.max(sum)
            delta = np.linalg.norm(v-self.V)

        for s in range(self.states):
            if s in self.terminal_states:
                pass
            else:
                sum = [0]*4
                state_index = self.ind_dict[s]  
                _, transitions = list(self.mdp.items())[state_index]
                possibleActions = list(transitions.keys())
                for a in possibleActions:                    
                    newStatesLength = len(transitions[a])
                    newStates = [transitions[a][i][1] for i in range(newStatesLength)]
                    newStatesInt = [self.statedic[str(new_state)] for new_state in  newStates]
                    for i in range(newStatesLength):
                        new_state_int = newStatesInt[i]
                        sum[a] += transitions[a][i][0] * (transitions[a][i][2] + self.gamma*self.V[new_state_int])
            self.p[s] = np.argmax(sum)
        
        return self.V, self.p

    def act(self, obs, eps, gamma):
        value, policy = self.valueIteration(eps, gamma)
        p = policy[obs]
        v = value[obs]
        return v, p


# define a method or a seperate class "act" to choose an action for each observation

if __name__ == '__main__':


    env = gym.make("gridworld-v0")
    env.setPlan("gridworldPlans/plan6.txt", {0: -0.001, 3: 1, 4: 1, 5: -1, 6: -1})
    # don't worry about Plan4 and 7 never finishing an episode: visualise the first episode and you'll see why (robot is stuck, can't get past pink barrier)
    # Not sure what's wrong with Plan9 though

    env.seed(0)  # Initialise le seed du pseudo-random
    env.render()  # permet de visualiser la grille du jeu 
    env.render(mode="human") #visualisation sur la console

    # Execution avec un Agent
    agent2 = ValueIteration(env=env)

    observation_space = env.observation_space

    statedic, mdp = env.getMDP()
    ind_dict = {}
    for i, k in enumerate(mdp):
        ind = statedic[k]
        ind_dict[ind] = i

    episode_count = 10 # I changed the episode count as well as the rendering condition since there is no learning happening during or in between the episodes (the 'learning')
                       # happens each time .act is called, independently of previous outcomes, and always yields the same policy and value function (value iteration)
    reward = 0
    done = False
    rsum = 0
    FPS = 0.0001
    for i in range(episode_count):
        obs = env.reset()
        state = statedic[str(obs.tolist())]
        #env.verbose = (i % 100 == 0 and i > 0)  # afficher 1 episode sur 100
        env.render(FPS)
        # if env.verbose:
        #     env.render(FPS)
        j = 0
        rsum = 0
        while True:
            state = statedic[str(obs.tolist())]
            value, action = agent2.act(state, 1e-2, 0.99)
            obs, reward, done, _ = env.step(action)
            rsum += reward
            j += 1
            # delete:
            env.render(FPS) 
            # if env.verbose:
            #     env.render(FPS)
            if done:
                print("Episode : " + str(i) + " rsum=" + str(rsum) + ", " + str(j) + " actions")
                break

    print("done")
    env.close()