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
from torch.distributions import Normal
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
#from critic_network import Actor, Critic
from SACnetworks import SoftQNetwork, PolicyNetwork

# tensorboard runs in root directory for some reason: choose /Desktop
class RandomAgent(object):
    """The world's simplest agent!"""

    def __init__(self, env, opt):
        self.opt=opt
        self.env=env
        if opt.fromFile is not None:
            self.load(opt.fromFile)
        self.action_range = [env.action_space.low[0], env.action_space.high[0]]
        self.obs_dim = env.observation_space.shape[0]
        self.action_dim = env.action_space.shape[0]
        # self.action_space = env.action_space
        # self.num_ouputs = self.action_space.shape[0]
        self.featureExtractor = opt.featExtractor(env)
        self.test=False
        self.nbEvents=0
        self.gamma = 0.95
        self.update_step = 0
        self.delay_step = 1 # 2
        self.q_lr = 0.01
        self.policy_lr = 0.001
        self.a_lr = 0.001
        self.q_net1_local = SoftQNetwork(self.obs_dim, self.action_dim)
        self.q_net2_local = SoftQNetwork(self.obs_dim, self.action_dim)
        self.q_net1_target = copy.deepcopy(self.q_net1_local)
        self.q_net2_target = copy.deepcopy(self.q_net2_local)
        self.policy_net = PolicyNetwork(self.obs_dim, self.action_dim)
        self.mem_size = 1000000
        self.batch_size = 1000
        self.buffer = Memory(mem_size=self.mem_size)
        self.q1_optimizer = optim.Adam(self.q_net1_local.parameters(), lr=self.q_lr)
        self.q2_optimizer = optim.Adam(self.q_net2_local.parameters(), lr=self.q_lr)
        self.policy_optimizer = optim.Adam(self.policy_net.parameters(), lr=self.policy_lr)
        self.alpha = 0.2
        self.tau = 0.1 # (le prof appelle ca rho)
        self.target_entropy = -torch.prod(torch.Tensor(self.env.action_space.shape)).item()
        self.log_alpha = torch.zeros(1, requires_grad=True)
        self.alpha_optim = optim.Adam([self.log_alpha], lr=self.a_lr)

    def act(self, obs):
        state = torch.from_numpy(obs).float().squeeze()
        mean, log_std = self.policy_net.forward(state)
        std = log_std.exp()

        normal = Normal(mean, std)
        z = normal.sample()
        action = torch.tanh(z)
        action = action.numpy()
        return self.rescale_action(action)

    def rescale_action(self, action):
        return action * (self.action_range[1] - self.action_range[0]) / 2.0 + \
            (self.action_range[1] + self.action_range[0]) / 2.0

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
        rewards = (torch.FloatTensor([x[2] for x in batch])/100.0).unsqueeze(1)
        next_states = torch.FloatTensor([x[3] for x in batch]).squeeze(1)
        dones = torch.FloatTensor([x[4] for x in batch])

        next_actions, next_log_pi = self.policy_net.sample(next_states)
        next_q1 = self.q_net1_target(next_states, next_actions)
        next_q2 = self.q_net2_target(next_states, next_actions)
        next_q_target = torch.min(next_q1, next_q2) - self.alpha * next_log_pi
        expected_q = rewards + (1.0 - dones) * self.gamma * next_q_target

        #curr_q1 = self.q_net1_local.forward(states, actions)
        curr_q1 = self.q_net1_local(states, actions)
        q1_moy = curr_q1.mean()
        #curr_q2 = self.q_net2_local.forward(states, actions)
        curr_q2 = self.q_net2_local(states, actions)
        q1_loss = F.mse_loss(curr_q1, expected_q.detach())
        q2_loss = F.mse_loss(curr_q2, expected_q.detach())

        self.q1_optimizer.zero_grad()
        q1_loss.backward()
        self.q1_optimizer.step()

        self.q2_optimizer.zero_grad()
        q2_loss.backward()
        self.q2_optimizer.step()

        new_actions, log_pi = self.policy_net.sample(states)
        if self.update_step % self.delay_step == 0:
            min_q = torch.min(
                self.q_net1_local.forward(states, new_actions),
                self.q_net2_local.forward(states, new_actions)
            )
            policy_loss = (self.alpha * log_pi - min_q).mean()
            logger.direct_write('Actor Loss', policy_loss, self.update_step)

            self.policy_optimizer.zero_grad()
            policy_loss.backward()
            self.policy_optimizer.step()

            for target_param, local_param in zip(self.q_net1_target.parameters(), self.q_net1_local.parameters()):
                target_param.data.copy_(self.tau * local_param + (1-self.tau) * target_param)

            for target_param, local_param in zip(self.q_net2_target.parameters(), self.q_net2_local.parameters()):
                target_param.data.copy_(self.tau * local_param + (1-self.tau) * target_param)

        alpha_loss =(self.log_alpha * (-log_pi.detach() - self.target_entropy)).mean()
        #alpha_loss =(self.log_alpha * (-log_pi - self.target_entropy).detach()).mean()
        self.alpha_optim.zero_grad()
        alpha_loss.backward()
        self.alpha_optim.step()
        self.alpha = self.log_alpha.exp()

        logger.direct_write('Critic Loss', q1_loss, self.update_step)
        #logger.direct_write('Entropy', alpha_loss, self.update_step)
        logger.direct_write('Q', q1_moy, self.update_step)

        self.update_step += 1

        return self

    # enregistrement de la transition pour exploitation par learn ulterieure
    def store(self,ob, action, new_ob, reward, done, it):
        # Si l'agent est en mode de test, on n'enregistre pas la transition
        if not self.test:

            # si on atteint la taille max d'episode en apprentissage, alors done ne devrait pas etre a true (episode pas vraiment fini dans l'environnement)
            if it == self.opt.maxLengthTrain:
                print("undone")
                done=False

            tr = (ob, action, reward, new_ob, done)
            self.buffer.store(tr)

    # retoune vrai si c'est le moment d'entraîner l'agent.
    # Dans cette version retourne vrai tous les freqoptim evenements
    # Mais on pourrait retourner vrai seulement si done pour s'entraîner seulement en fin d'episode
    def timeToLearn(self,done):
        if self.test:
            return False
        self.nbEvents+=1
        return self.nbEvents%self.opt.freqOptim == 0

if __name__ == '__main__':
    env, config, outdir, logger = init('/Users/naomi/Desktop/Uni/M2A/2021-2022/RLD/TME8/configs/config_random_pendulum.yaml', "RandomAgent")

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
                # agent.buffer = Memory(mem_size=1000)
            if done:
                print(str(i) + " rsum=" + str(rsum) + ", " + str(j) + " actions ")
                logger.direct_write("reward", rsum, i)
                #agent.nbEvents = 0
                mean += rsum
                rsum = 0

                break

    env.close()
