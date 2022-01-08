import numpy as np
import pandas as pd
from pandas.core.accessor import delegate_names
import random as rd
import matplotlib.pyplot as plt

df = pd.read_csv('Bureau/M2A/2020-TMEs/RLD/TME1-2/CTR.csv', header = None, sep=':', index_col=0)
print(df.head())
dimension_list = ['dim'+str(i) for i in range(1,6)]
ad_list = ['ad'+str(i) for i in range(1,11)]
df[dimension_list] = df[1].str.split(';', expand=True).astype(float)
df[ad_list] = df[2].str.split(';', expand=True).astype(float)
df = df.drop([1,2], axis=1)
print(df.head())

def Random(data):
    """Stratégie random"""
    N = data.shape[0]
    selection_list = []
    total_reward = 0
    for n in range(N):
        selection = rd.choice(ad_list)
        ad = int(selection[2:])
        selection_list.append(ad)
        reward = data.values[n, ad+4]
        total_reward += reward
    return selection_list, total_reward

def StaticBest(data):
    """Stratégie Static Best"""
    data_cumsum = data[ad_list].cumsum()
    selection_list = []
    total_reward = 0
    for i in range(data.shape[0]):
        selection = data_cumsum.iloc[i].idxmax()
        ad = int(selection[2:])
        selection_list.append(ad)
        reward = data.values[i, ad+4]
        total_reward += reward
    return selection_list, total_reward

def Optimal(data):
    """Stratégie optimale"""
    selection_list = list(data[ad_list].values.argmax(1)+1)
    total_reward = sum(data[ad_list].values.max(1))
    return selection_list, total_reward

def UCB(data):
    """Algorithme UCB"""
    N = data.shape[0]
    d = 10
    selection_list = []
    number_of_selections = [0]*d
    sums_of_rewards = [0]*d
    total_reward = 0
    
    for n in range(N):
        ad = 0
        max_upper_bound = 0
        for i in range(d):
            if number_of_selections[i] == 0:
                upper_bound = 1e400
            else:
                average_reward = sums_of_rewards[i]/number_of_selections[i]
                delta_i = np.sqrt(2*np.log(n+1)/number_of_selections[i])
                upper_bound = average_reward + delta_i

            if upper_bound > max_upper_bound:
                max_upper_bound = upper_bound
                ad = i
        selection_list.append(ad+1)
        number_of_selections[ad]+=1
        reward = data.values[n, ad+5]
        sums_of_rewards[ad] += reward
        total_reward += reward

    return selection_list, number_of_selections, sums_of_rewards, total_reward

def LinUCB(data, alpha):
    N = data.shape[0]  # nombre d'iterations
    annonces = 10  # nombre de choix (les annonces)
    d = 5  # dimension du feature vector
    A = [np.identity(d) for a in range(annonces)]  # initier matrice A pour toutes les annonces
    b = [np.zeros(d) for a in range(annonces)]  #initier vecteur b pour toutes les annonces
    selection_list = []
    number_of_selections = [0]*annonces
    sums_of_rewards = [0]*annonces
    total_reward = 0
    for i in range(N):
        context = np.array(data[dimension_list].iloc[i])
        p = np.zeros(annonces)
        for a in range(annonces):
            A_inv = np.linalg.inv(A[a])
            theta = np.dot(A_inv, b[a])
            p[a] = float(np.dot(theta.T, context) + alpha*np.sqrt(np.dot(context.T, np.dot(A_inv, context))))
        candidates = np.argwhere(p == np.amax(p))
        candidates_list =  candidates.flatten().tolist()
        chosen_candidate = np.random.choice(candidates_list)
        selection_list.append(chosen_candidate+1)
        number_of_selections[chosen_candidate] +=1
        reward = data.values[i, chosen_candidate+5]
        sums_of_rewards[chosen_candidate] += reward
        total_reward += reward
        A[chosen_candidate] += np.dot(context, context.T)
        b[chosen_candidate] += data.values[i, chosen_candidate+5]*context
    
    return selection_list, number_of_selections, sums_of_rewards, total_reward

fig, axs = plt.subplots(3, 2, sharex=True)
fig.suptitle('Selection Histrograms for Different Methods')

rd_select, rd_reward = Random(df)
axs[0, 0].hist(rd_select, density=True)
axs[0, 0].set_title('Random Selection')

stat_select, stat_reward = StaticBest(df)
axs[0, 1].hist(stat_select, density=True)
axs[0, 1].set_title('StaticBest Selection')

opt_select, opt_reward = Optimal(df)
axs[1, 0].hist(opt_select, density=True)
axs[1, 0].set_title('Optimal Selection')

ucb_select, ucb_numbers, ucb_sums, ucb_reward = UCB(df)
axs[1, 1].hist(ucb_select, density=True)
axs[1, 1].set_title('UCB Selection')

alpha1 = 1 + np.sqrt(np.log(2/0.05)/2)
linucb_select_1, linucb_numbers_1, linucb_sums_1, linucb_reward_1 = LinUCB(df, alpha1)
axs[2, 0].hist(linucb_select_1, density=True)
axs[2, 0].set_title('LinUCB with delta = 0.05')

alpha2 = 1 + np.sqrt(np.log(2/0.01)/2)
linucb_select_2, linucb_numbers_2, linucb_sums_2, linucb_reward_2 = LinUCB(df, alpha2)
axs[2, 1].hist(linucb_select_2, density=True)
axs[2, 1].set_title('LinUCB with delta = 0.01')

for ax in axs.flat:
    ax.set(xlabel='Ad', ylabel='frequency of selections')

plt.show()

for i in range(10):
    print("\n Times ad {} was selected through Random Strategy: {}".format(i+1, rd_select.count(i)))
    print("\n Times ad {} was selected through StaticBest Strategy: {}".format(i+1, stat_select.count(i)))
    print("\n Times ad {} was selected through Opimal Strategy: {}".format(i+1, opt_select.count(i)))
    print("\n Times ad {} was selected in UCB: {}".format(i+1, ucb_numbers[i]))
    print("\n Times ad {} was selected in LinUCB with delta = 0.05: {}".format(i+1, linucb_numbers_1[i]))
    print("\n Times ad {} was selected in LinUCB with delta = 0.01: {}".format(i+1, linucb_numbers_2[i]))
print("\n Total reward with Random Strategy:", rd_reward)
print("\n Total reward with StaticBest Strategy:", stat_reward)
print("\n Total reward with Optimal Strategy:", opt_reward)
print("\n Total reward with UCB:", ucb_reward)
print("\n Total reward with LinUCB for delta = 0.05:", linucb_reward_1)
print("\n Total reward with LinUCB for delta = 0.01:", linucb_reward_2)