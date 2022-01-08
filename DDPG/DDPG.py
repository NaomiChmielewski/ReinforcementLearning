import argparse
import sys
import matplotlib
from torch import random
#matplotlib.use("Qt5agg")
matplotlib.use("TkAgg")
import gym
import gridworld
import torch
from torch import gather
from torch.nn import SmoothL1Loss
import torch.nn.functional as F
from torch import autograd
from utils import *
from core import *
from memory import *
from torch.utils.tensorboard import SummaryWriter
from torch.autograd import Variable
#import highway_env
from matplotlib import pyplot as plt
import yaml
from datetime import datetime
import random as rd
from torch import optim
import copy
from torch.distributions import Categorical
from critic_network import Actor, Critic

# tensorboard runs in root directory for some reason: choose /Desktop
class RandomAgent(object):
    """The world's simplest agent!"""

    def __init__(self, env, opt, gae=False):
        self.opt=opt
        self.env=env
        self.gae=gae
        if opt.fromFile is not None:
            self.load(opt.fromFile)
        self.obs_dim = env.observation_space.shape[0]
        self.action_space = env.action_space
        print('action space', self.action_space)
        self.num_ouputs = self.action_space.shape[0]
        self.featureExtractor = opt.featExtractor(env)
        self.test=False
        self.nbEvents=0
        self.gamma = 0.95
        self.upper_bound = env.action_space.high[0]
        self.lower_bound = env.action_space.low[0]
        self.actor_local = Actor([30, 30], self.obs_dim, self.action_space)
        # self.actor_local = torch.nn.Sequential(
        #     torch.nn.Linear(self.obs_dim, 30),
        #     torch.nn.LayerNorm(30),
        #     torch.nn.ReLU(),
        #     torch.nn.Linear(30, 30),
        #     torch.nn.LayerNorm(30),
        #     torch.nn.ReLU(),
        #     torch.nn.Linear(30, self.num_ouputs),
        #     torch.nn.Tanh()
        # )
        self.critic_local = Critic([30,30], self.obs_dim, self.action_space)
        # self.critic_local = torch.nn.Sequential(
        #     torch.nn.Linear(self.obs_dim, 30),
        #     torch.nn.LayerNorm(30),
        #     torch.nn.ReLU(),
        #     torch.nn.Linear(30, 30),
        #     torch.nn.LayerNorm(30),
        #     torch.nn.ReLU(),
        #     torch.nn.Linear(30, self.num_ouputs)
        # )
        self.actor_target = copy.deepcopy(self.actor_local)
        #print('number of actions', self.num_ouputs)
        self.critic_target = copy.deepcopy(self.critic_local)
        self.mem_size = 1000000
        self.batch_size = 1000
        self.buffer = Memory(mem_size=self.mem_size)
        self.critic_optimizer = optim.Adam(self.critic_local.parameters(), lr=3e-3)#, weight_decay=1e-2) # 3e-3
        self.actor_optimizer = optim.Adam(self.actor_local.parameters(), lr=1e-3) # 3e-4
        self.noise = Orn_Uhlen(n_actions=1, sigma=0.2)
        self.tau = 0.1 # (le prof appelle ca rho)
        self.t_step = 0

    def act(self, obs):
        state = torch.from_numpy(obs).float().squeeze()
        policy = self.actor_local(state).data #item()?
        noise = self.noise.sample()
        act = policy + noise
        act = act.clamp(self.lower_bound, self.upper_bound)
        return act

    # sauvegarde du modèle
    def save(self,outputDir):
        pass

    # chargement du modèle.
    def load(self,inputDir):
        pass

    # apprentissage de l'agent. Dans cette version rien à faire
    def learn(self):
        # Si l'agent est en mode de test, on n'entraîne pas
        if self.test:
            return 
        idx, w, batch = self.buffer.sample(self.batch_size)
        states = torch.FloatTensor([x[0] for x in batch]).squeeze(1) 
        actions = torch.FloatTensor([x[1] for x in batch]).resize_((self.batch_size,1))
        rewards = (torch.FloatTensor([x[2] for x in batch])/1000.0).unsqueeze(1)
        next_states = torch.FloatTensor([x[3] for x in batch]).squeeze(1)
        dones = torch.FloatTensor([x[4] for x in batch])

        next_action_batch = self.actor_target(next_states)

        next_state_action_values = self.critic_target(next_states, next_action_batch.detach())
        expected_values = rewards + (1.0-dones)*self.gamma * next_state_action_values
        

        self.critic_optimizer.zero_grad()
        state_action_batch = self.critic_local(states, actions)
        q_moy = state_action_batch.mean()
        value_loss = F.mse_loss(state_action_batch, expected_values.detach())
        value_loss.backward()
        self.critic_optimizer.step()

        self.actor_optimizer.zero_grad()
        #predicted_action = self.actor_local(states).squeeze(1).detach()
        policy_loss = -self.critic_local(states, self.actor_local(states)) 
        policy_loss = policy_loss.mean()
        policy_loss.backward()
        self.actor_optimizer.step()        

        self.soft_update(self.actor_target, self.actor_local, self.tau)
        self.soft_update(self.critic_target, self.critic_local, self.tau)

        logger.direct_write('Actor Loss', policy_loss, self.t_step)
        logger.direct_write('Critic Loss', value_loss, self.t_step)
        logger.direct_write('Q', q_moy, self.t_step)

        self.t_step += 1

        return self, policy_loss, value_loss

    # enregistrement de la transition pour exploitation par learn ulterieure
    def store(self,ob, action, new_ob, reward, done, it):
        # Si l'agent est en mode de test, on n'enregistre pas la transition
        if not self.test:

            # si on atteint la taille max d'episode en apprentissage, alors done ne devrait pas etre a true (episode pas vraiment fini dans l'environnement)
            if it == self.opt.maxLengthTrain:
                print("undone")
                done=False
            if not self.gae:
                tr = (ob, action, reward, new_ob, done)
                self.buffer.store(tr)

    def soft_update(self, target, source, tau):
        for target_param, param in zip(target.parameters(), source.parameters()):
            target_param.data.copy_(target_param.data * (1.0 - tau) + param.data * tau)

    # retoune vrai si c'est le moment d'entraîner l'agent.
    # Dans cette version retourne vrai tous les freqoptim evenements
    # Mais on pourrait retourner vrai seulement si done pour s'entraîner seulement en fin d'episode
    def timeToLearn(self,done):
        if self.test:
            return False
        self.nbEvents+=1
        return self.nbEvents%self.opt.freqOptim == 0

if __name__ == '__main__':
    env, config, outdir, logger = init('/Users/naomi/Desktop/Uni/M2A/2021-2022/RLD/TME4env/configs/config_random_pendulum.yaml', "RandomAgent")

    freqTest = config["freqTest"]
    freqSave = config["freqSave"]
    nbTest = config["nbTest"]
    env.seed(config["seed"])
    np.random.seed(config["seed"])
    episode_count = config["nbEpisodes"]

    agent = RandomAgent(env,config)


    rsum = 0
    mean = 0
    verbose = True
    itest = 0
    reward = 0
    done = False
    time_to_learn = 0
    step_count = 0
    for i in range(episode_count):
        checkConfUpdate(outdir, config)

        rsum = 0
        #agent.nbEvents = 0
        ob = env.reset()

        # On souhaite afficher l'environnement (attention à ne pas trop afficher car çà ralentit beaucoup)
        if i % int(config["freqVerbose"]) == 0:
            verbose = True
        else:
            verbose = False

        # C'est le moment de tester l'agent
        if i % freqTest == 0 and i >= freqTest:  ##### Same as train for now
            print("Test time! ")
            mean = 0
            agent.test = True

        # On a fini cette session de test
        if i % freqTest == nbTest and i > freqTest:
            print("End of test, mean reward=", mean / nbTest)
            itest += 1
            logger.direct_write("rewardTest", mean / nbTest, itest)
            agent.test = False

        # C'est le moment de sauver le modèle
        if i % freqSave == 0:
            agent.save(outdir + "/save_" + str(i))

        j = 0
        if verbose:
            env.render()

        new_ob = agent.featureExtractor.getFeatures(ob)
        while True:
            if verbose:
                env.render()

            ob = new_ob
            action= agent.act(ob)
            new_ob, reward, done, _ = env.step(action)
            new_ob = agent.featureExtractor.getFeatures(new_ob)

            j+=1

            # Si on a atteint la longueur max définie dans le fichier de config
            if ((config["maxLengthTrain"] > 0) and (not agent.test) and (j == config["maxLengthTrain"])) or ( (agent.test) and (config["maxLengthTest"] > 0) and (j == config["maxLengthTest"])):
                done = True
                print("forced done!")

            agent.store(ob, action, new_ob, reward, done,j)
            rsum += reward

            if agent.timeToLearn(done):
                print('learning')
                for _ in range(10):
                    agent.learn()
                # if not agent.test:
                #     actor_loss = agent.learn()[1].detach()
                #     critic_loss = agent.learn()[2].detach()
                #     logger.direct_write('Actor Loss', actor_loss, i)
                #     logger.direct_write('Critic Loss', critic_loss, i)
                agent.buffer = Memory(mem_size=1000)
            if done:
                print(str(i) + " rsum=" + str(rsum) + ", " + str(j) + " actions ")
                logger.direct_write("reward", rsum, i)
                #agent.nbEvents = 0
                mean += rsum
                rsum = 0

                break

    env.close()
