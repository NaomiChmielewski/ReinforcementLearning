import argparse
import sys
import matplotlib
from torch import random
from torch.distributions.utils import logits_to_probs
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
        self.featureExtractor = opt.featExtractor(env)
        self.test=False
        self.nbEvents=0

        self.actor=nn.Sequential(
            nn.Linear(self.obs_dim,64),
            nn.Tanh(),
            nn.Linear(64,64),
            nn.Tanh(),
            nn.Linear(64,self.action_space.n),
            nn.Softmax(dim=-1)
        )
        self.critic=nn.Sequential(
            nn.Linear(self.obs_dim,64),
            nn.Tanh(),
            nn.Linear(64,64),
            nn.Tanh(),
            nn.Linear(64,1)
        )
        self.critic_target = copy.deepcopy(self.critic)
        self.mem_size = 1000
        self.buffer = Memory(mem_size=self.mem_size)
        self.optimizer = optim.Adam(self.critic.parameters(), lr=2e-4) # best values so far: 8e-3 with 5e-3
        self.actor_optimizer = optim.Adam(self.actor.parameters(), lr=2e-4)
        self.t_step = 0
        self.gamma = 0.999
        self.lambd = 0.99
        self.batch_size = 1000
        self.K = 10
        self.C = 1000
        self.clip = 0.2 
        self.kl = False
        self.clipped = False
        self.beta = 1.0
        self.delta = 0.01

        self.states = []
        self.actions = []
        self.log_probs = []
        self.rewards = []
        self.dones = []
        self.new_states = []
        self.values = []
        
        self.actor_count = 0
        self.critic_count = 0

    def act(self, obs):

        policy_dist = self.actor(torch.FloatTensor(obs))
        # state = torch.from_numpy(obs).float().squeeze()
        # policy = self.actor(state)
        dist = Categorical(policy_dist)
        action = dist.sample()

        if not self.test:
            self.log_probs.append(dist.log_prob(action))
            self.actions.append(action.detach())
            self.states.append(torch.FloatTensor(obs))
            self.values.append(self.critic_target(torch.FloatTensor(obs)).detach())

        return action.item()

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

        if self.clipped:
            value = self.critic_target(torch.FloatTensor(self.new_states[-1])).item()

            returns = np.zeros_like(self.rewards)
            gae = np.zeros_like(self.rewards)
            adv = 0.
            for t in reversed(range(len(self.rewards))):
                if t == len(self.rewards)-1:
                    returns[t] = self.rewards[t] + self.gamma * (1-self.dones[t]) * value
                    delta = self.rewards[t] + self.gamma * (1-self.dones[t]) * value - self.values[t].item()
                else:
                    returns[t] = self.rewards[t] + self.gamma * (1-self.dones[t]) * returns[t+1]
                    delta = self.rewards[t] + self.gamma * (1-self.dones[t]) * self.values[t+1].item() - self.values[t].item()

                adv = adv * self.gamma * self.lambd * (1-self.dones[t]) + delta
                gae[t] = adv

            target = torch.FloatTensor(returns)
            gae = torch.FloatTensor(gae)
            gae = (gae - gae.mean()) / (gae.std() + 1e-10)
            
            old_states = torch.stack(self.states, dim=0).squeeze().detach()
            old_actions = torch.stack(self.actions, dim=0).squeeze().detach()
            old_log_probs = torch.stack(self.log_probs, dim=0).squeeze().detach()

            state_values = self.critic(old_states).view(-1)
            old_dist = self.actor(old_states).view((-1, self.action_space.n))

            for _ in range(self.K):

                policy_dist = self.actor(old_states)
                dist = Categorical(policy_dist)
                log_probs = dist.log_prob(old_actions)
                ratios = torch.exp(log_probs - old_log_probs.detach())

                surr1 = ratios * gae.detach()
                surr2 = torch.clamp(ratios, 1 - self.clip, 1 + self.clip) * gae.detach()

                actor_loss = (-torch.min(surr1, surr2)).mean()
                # Entropy?

                self.actor_count += 1
                self.actor_loss = actor_loss
                self.actor_optimizer.zero_grad()
                actor_loss.backward(retain_graph=True)
                self.actor_optimizer.step() 

                self.C += 1

            new_dist = self.actor(old_states).view((-1, self.action_space.n))
            kl_div = F.kl_div(new_dist, old_dist.view((-1, self.action_space.n)), reduction='batchmean')
            self.kl_loss = kl_div
            
            critic_loss = F.smooth_l1_loss(target, state_values)
            self.critic_loss = critic_loss
            self.critic_count += 1
            self.optimizer.zero_grad()
            critic_loss.backward()
            self.optimizer.step()

            self.states = []
            self.actions = []
            self.log_probs = []
            self.rewards = []
            self.dones = []
            self.new_states = []
            self.values = []

            if self.t_step%self.C == 0:
                print('UPDATE target')
                for target_param, local_param in zip(self.critic_target.parameters(), self.critic.parameters()):
                    target_param.data.copy_(local_param.data)

        elif self.kl:
            value = self.critic(torch.FloatTensor(self.new_states[-1])).item()

            returns = np.zeros_like(self.rewards)
            gae = np.zeros_like(self.rewards)
            adv = 0.
            for t in reversed(range(len(self.rewards))):
                if t == len(self.rewards)-1:
                    returns[t] = self.rewards[t] + self.gamma * (1-self.dones[t]) * value
                    delta = self.rewards[t] + self.gamma * (1-self.dones[t]) * value - self.values[t].item()
                else:
                    returns[t] = self.rewards[t] + self.gamma * (1-self.dones[t]) * returns[t+1]
                    delta = self.rewards[t] + self.gamma * (1-self.dones[t]) * self.values[t+1].item() - self.values[t].item()

                adv = adv * self.gamma * self.lambd * (1-self.dones[t]) + delta
                gae[t] = adv

            target = torch.FloatTensor(returns)
            gae = torch.FloatTensor(gae)
            gae = (gae - gae.mean()) / (gae.std() + 1e-10)
            
            old_states = torch.stack(self.states, dim=0).squeeze().detach()
            old_actions = torch.stack(self.actions, dim=0).squeeze().detach()
            old_log_probs = torch.stack(self.log_probs, dim=0).squeeze().detach()

            state_values = self.critic(old_states).view(-1)
            old_dist = self.actor(old_states).view((-1, self.action_space.n))

            for _ in range(self.K):

                policy_dist = self.actor(old_states)
                dist = Categorical(policy_dist)
                log_probs = dist.log_prob(old_actions)

                ratios = torch.exp(log_probs - old_log_probs.detach())

                surr1 = (ratios * gae.detach()).mean()
                surr2 = F.kl_div(dist.probs, old_dist.detach(), reduction='batchmean')

                actor_loss = -(surr1-self.beta * surr2)
                # Entropy?
                self.actor_count += 1
                self.actor_loss = actor_loss
                self.actor_optimizer.zero_grad()
                actor_loss.backward()
                self.actor_optimizer.step()

            policy_dist = self.actor(old_states)
            dist = Categorical(policy_dist)
            kl_div = F.kl_div(dist.probs.view((-1, self.action_space.n)), target=old_dist.view((-1, self.action_space.n)), reduction='batchmean')
            self.kl_loss = kl_div
            if kl_div>=1.5*self.delta:
                self.beta*=2
            if kl_div<=self.delta/1.5:
                self.beta*=0.5

            critic_loss = F.smooth_l1_loss(target, state_values.view(-1))
            self.critic_loss = critic_loss
            self.critic_count += 1
            self.optimizer.zero_grad() 
            critic_loss.backward()
            self.optimizer.step()

            self.states = []
            self.actions = []
            self.log_probs = []
            self.rewards = []
            self.dones = []
            self.new_states = []
            self.values = []

            if self.t_step%self.C == 0:
                print('UPDATE target')
                for target_param, local_param in zip(self.critic_target.parameters(), self.critic.parameters()):
                    target_param.data.copy_(local_param.data)
        else:
            value = self.critic(torch.FloatTensor(self.new_states[-1])).item()

            returns = np.zeros_like(self.rewards)
            gae = np.zeros_like(self.rewards)
            adv = 0.
            for t in reversed(range(len(self.rewards))):
                if t == len(self.rewards)-1:
                    returns[t] = self.rewards[t] + self.gamma * (1-self.dones[t]) * value
                    delta = self.rewards[t] + self.gamma * (1-self.dones[t]) * value - self.values[t].item()
                else:
                    returns[t] = self.rewards[t] + self.gamma * (1-self.dones[t]) * returns[t+1]
                    delta = self.rewards[t] + self.gamma * (1-self.dones[t]) * self.values[t+1].item() - self.values[t].item()

                adv = adv * self.gamma * self.lambd * (1-self.dones[t]) + delta
                gae[t] = adv

            target = torch.FloatTensor(returns)
            gae = torch.FloatTensor(gae)
            gae = (gae - gae.mean()) / (gae.std() + 1e-10)
            
            old_states = torch.stack(self.states, dim=0).squeeze().detach()
            old_actions = torch.stack(self.actions, dim=0).squeeze().detach()
            old_log_probs = torch.stack(self.log_probs, dim=0).squeeze().detach()

            state_values = self.critic(old_states).view(-1)
            old_dist = self.actor(old_states).view((-1, self.action_space.n))

            for _ in range(self.K):

                policy_dist = self.actor(old_states)
                dist = Categorical(policy_dist)
                log_probs = dist.log_prob(old_actions)

                ratios = torch.exp(log_probs - old_log_probs.detach())

                surr1 = (ratios * gae.detach()).mean()

                actor_loss = -surr1
                # Entropy?
                self.actor_count += 1
                self.actor_loss = actor_loss
                self.actor_optimizer.zero_grad()
                actor_loss.backward()
                self.actor_optimizer.step()

            policy_dist = self.actor(old_states)
            dist = Categorical(policy_dist)
            kl_div = F.kl_div(dist.probs.view((-1, self.action_space.n)), target=old_dist.view((-1, self.action_space.n)), reduction='batchmean')
            self.kl_loss = kl_div

            critic_loss = F.smooth_l1_loss(target, state_values.view(-1))
            self.critic_loss = critic_loss
            self.critic_count += 1
            self.optimizer.zero_grad() 
            critic_loss.backward()
            self.optimizer.step()

            self.states = []
            self.actions = []
            self.log_probs = []
            self.rewards = []
            self.dones = []
            self.new_states = []
            self.values = []

            if self.t_step%self.C == 0:
                print('UPDATE target')
                for target_param, local_param in zip(self.critic_target.parameters(), self.critic.parameters()):
                    target_param.data.copy_(local_param.data)               

    # enregistrement de la transition pour exploitation par learn ulterieure
    def store(self,ob, action, new_obs, reward, done, it):
        # Si l'agent est en mode de test, on n'enregistre pas la transition
        if not self.test:

            # si on atteint la taille max d'episode en apprentissage, alors done ne devrait pas etre a true (episode pas vraiment fini dans l'environnement)
            if it == self.opt.maxLengthTrain:
                print("undone")
                done=False
           
            self.rewards.append(reward)
            self.dones.append(float(done))
            self.new_states.append(new_obs)

    def evaluate(self, episode_obs, episode_acts):
        episode_obs = torch.tensor(episode_obs, dtype=torch.float)
        episode_acts = torch.tensor(episode_acts, dtype=torch.int64).resize_((len(episode_obs),1))
        V = self.critic_target(episode_obs).squeeze().gather(1, episode_acts)
        return V.squeeze()

    # retoune vrai si c'est le moment d'entraîner l'agent.
    # Dans cette version retourne vrai tous les freqoptim evenements
    # Mais on pourrait retourner vrai seulement si done pour s'entraîner seulement en fin d'episode
    def timeToLearn(self,done):
        if self.test:
            return False
        self.nbEvents+=1
        return self.nbEvents%self.opt.freqOptim == 0

if __name__ == '__main__':
    env, config, outdir, logger = init('/Users/naomi/Desktop/Uni/M2A/2021-2022/RLD/TME4env/configs/config_random_cartpole.yaml', "RandomAgent")
    
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
    t=time.time()
    for i in range(episode_count):
        checkConfUpdate(outdir, config)

        rsum = 0
        ob = env.reset()

        # On souhaite afficher l'environnement (attention à ne pas trop afficher car çà ralentit beaucoup)
        if i % int(config["freqVerbose"]) == 0:
            verbose = True
        else:
            verbose = False

        verbose=False

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

        j = 0
        if verbose:
            env.render()

        new_obs = agent.featureExtractor.getFeatures(ob)
        
        while True:
            if verbose:
                env.render()

            ob = new_obs
            
            action= agent.act(ob)
            new_obs, reward, done, _ = env.step(action)
           
            new_obs = agent.featureExtractor.getFeatures(new_obs)
            agent.store(ob, action, new_obs, reward, done,j)
            
            j+=1

            # Si on a atteint la longueur max définie dans le fichier de config
            if ((config["maxLengthTrain"] > 0) and (not agent.test) and (j == config["maxLengthTrain"])) or ( (agent.test) and (config["maxLengthTest"] > 0) and (j == config["maxLengthTest"])):
                done = True
                print("forced done!")

            
            rsum += reward

            if agent.timeToLearn(done):
                agent.learn()
                logger.direct_write("actor loss", agent.actor_loss, agent.actor_count)
                logger.direct_write("critic loss", agent.critic_loss, agent.critic_count)
                logger.direct_write("KL loss", agent.kl_loss, agent.critic_count)

            if done:
                print(str(i) + " rsum=" + str(rsum) + ", " + str(j) + " actions ")
                logger.direct_write("reward", rsum, i)
                mean += rsum
                rsum = 0

                break

                       
      
    env.close()