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

# tensorboard runs in root directory for some reason: choose /Desktop
class RandomAgent(object):
    """The world's simplest agent!"""

    def __init__(self, env, opt):
        self.opt=opt
        self.env=env
        if opt.fromFile is not None:
            self.load(opt.fromFile)
        self.obs_dim = env.observation_space.shape[0]
        self.action_space = env.action_space
        self.featureExtractor = opt.featExtractor(env)
        self.test=False
        self.nbEvents=0
        self.gamma = 0.9
        self.qnetwork_local = NN(self.obs_dim, env.action_space.n, layers=[200])
        self.qnetwork_target = copy.deepcopy(self.qnetwork_local)
        self.buffer = Memory(mem_size=1000)
        self.optimizer = optim.Adam(self.qnetwork_local.parameters(), lr=3e-4)
        self.eps = 0.1
        self.t_step = 0
        self.C = 20

    def update_it(self):
        self.eps *= 0.99999

    def act(self, obs):
        state = torch.from_numpy(obs).float().unsqueeze(0)
        qvals = self.qnetwork_local.forward(state)
        
        if rd.random() > self.eps:
            return torch.argmax(qvals).item()
        else:
            return self.action_space.sample()

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
        idx, w, batch = self.buffer.sample(1)
        states = torch.FloatTensor([x[0] for x in batch]).squeeze()
        actions = torch.FloatTensor([x[1] for x in batch]).type(torch.int64)#.resize_((1,1))
        rewards = torch.FloatTensor([x[2] for x in batch])
        next_states = torch.FloatTensor([x[3] for x in batch])
        dones = torch.FloatTensor([x[4] for x in batch])

        predicted_targets = self.qnetwork_local(states)[actions]
        labels_next = self.qnetwork_local(next_states).max(1)[0].detach()  #change to qnetwork_target
        labels = rewards + self.gamma*labels_next*(1-dones)
        loss = F.smooth_l1_loss(predicted_targets,labels)
        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()
        if self.t_step%self.C == 0:
            for target_param, local_param in zip(self.qnetwork_target.parameters(), self.qnetwork_local.parameters()):
                target_param.data.copy_(local_param.data)
        self.t_step += 1
        return self, loss

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
            #self.lastTransition=tr #ici on n'enregistre que la derniere transition pour traitement immédiat, mais on pourrait enregistrer dans une structure de buffer (c'est l'interet de memory.py)

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
    for i in range(episode_count):
        checkConfUpdate(outdir, config)

        rsum = 0
        agent.nbEvents = 0
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

            agent.update_it()
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
                agent.learn()
                loss = agent.learn()[1].detach()
                logger.direct_write('Loss', loss, i)
            if done:
                print(str(i) + " rsum=" + str(rsum) + ", " + str(j) + " actions ")
                logger.direct_write("reward", rsum, i)
                agent.nbEvents = 0
                mean += rsum
                rsum = 0

                break

    env.close()
