from math import inf
from time import time
from random import choice, randrange, random
from itertools import count
import pickle 
import numpy as np
from random import random, choice
from problem import read_data, q_learning_e, ZCRobot

# simulation function 

robot, picks, drops, bounds, map_ = read_data()
rewards = [-2, 20, 50, 10000, -1000]

def simulate(env_ctor, n_iterations=inf, duration=inf, **q_learning_params):
    '''A helper function to train for a fixed number of iterations or fixed time'''
    for param in ('q', 'n'): q_learning_params[param] = q_learning_params.get(param, {})
    start_time = time()
    i = count()
    while time() < start_time + duration and next(i) < n_iterations:
        env = ZCRobot(robot, picks, drops, bounds, rewards, map_,  0.3)
        q, n = q_learning_e(env, **q_learning_params)
    return q_learning_params['q'], q_learning_params['n']



# ----------------------------------------------- #

# load pre trained model
path = []

q, n = {}, {}

with open('../pre-model/q_f.pkl', 'rb') as handle:
    q = pickle.load(handle)

with open('../pre-model/n_f.pkl', 'rb') as handle:
    n = pickle.load(handle)


simulate("Problem", q = q, n = n, n_iterations = 1, get_path = True, path = path, can_explore = False)
 
print(len(path))

#simulate("Problem", q = q, n = n, n_iterations = 10000, get_path = True, path = path, can_explore = False)

path = [] 

#simulate("Problem", q = q, n = n, n_iterations = 1, get_path = True, path = path, can_explore = False)



path_pre = []

with open('../pre-model/p_f.pkl', 'rb') as handle:
    path_pre = pickle.load(handle)["a"]

print(len(path_pre))
