import csv
import pickle 
from random import random

def read_data():
  data = list(csv.reader(open("../assets/final.csv")))
 # data = np.ndarray.tolist(np.transpose(np.array(data)))
  robot = ()
  picks = []
  drops = []
  for i in range(len(data)):
    if '0' in data[i] : robot = (i, data[i].index('0')); break

  for i in range(len(data)):
    for j in range(len(data[i])):
      if data[i][j] == '1' : picks.append((i,j))

  for i in range(len(data)):
    for j in range(len(data[i])):
      if data[i][j] == '3' : drops.append((i,j))

  bounds = (len(data), len(data[0]))

  return robot, tuple(picks), tuple(drops), bounds, data

class Environment:
    '''
    Abstract base class for an (interactive) environment formulation.
    It declares the expected methods to be used to solve it.
    All the methods declared are just placeholders that throw errors if not overriden by child "concrete" classes!
    '''
    
    def __init__(self):
        '''Constructor that initializes the problem. Typically used to setup the initial state.'''
        self.state = None
    
    def actions(self):
        '''Returns an iterable with the applicable actions to the current environment state.'''
        raise NotImplementedError
    
    def apply(self, action):
        '''Applies the action to the current state of the environment and returns the new state from applying the given action to the current environment state; not necessarily deterministic.'''
        raise NotImplementedError
    
    @classmethod
    def new_random_instance(cls):
        '''Factory method to a problem instance with a random initial state.'''
        raise NotImplementedError


def q_learning_e(env, q={}, n={}, f=lambda q, n: (q+1)/(n+1), alpha=lambda n: 0.5 ,epsilon = 0.1, error=1e-6, can_explore = True, get_path = False, path = {}):
    '''Q-learning implementation that trains on an environment till no more actions can be taken'''
    while env.state is not None:
        state = env.state
        action =  choice(env.actions()) \
          if random() < epsilon and can_explore else \
          max((action_ for action_ in env.actions()), key=lambda action_: q.get((env.state, action_), 0))
        if get_path : path.append((state, action))
        n[(state, action)] = n.get((state, action), 0) + 1
        reward = env.apply(action)
        if reward < -50000 : return q, n
        q[(state, action)] = q.get((state, action), 0) \
                           + alpha(n[state, action]) \
                           * (reward
                              + env.discount * max((q.get((env.state, next_action), 0) for next_action in env.actions()), default=0)
                              - q.get((state, action), 0))
    return q, n

class ZCRobot(Environment):
    '''ZC World'''
    
    def __init__(self, robot, picks, drops, bounds, rewards, map_, discount):
        self.state = (robot, picks, drops, False)
        self.map = map_
        self.init_state = self.state
        self.bounds = bounds
        self.rewards = rewards
        self.discount = discount
    
    def set_state(self, state) :
        if state is None : 
          self.state = list(self.state)
          self.state = None 
          return 
        state = list(state)
        self.state = list(self.state)
        self.state = tuple(state)

    def actions(self):
        if self.state is None: return []
        if self.state[0] in self.state[1] and self.state[3] == False: return ['pick'] # if not carries pick
        if self.state[0] in self.state[2] and self.state[3]: return ['drop'] # if carries drop
        if self.map[self.state[0][0]][self.state[0][1]] == '4': return ['out'] # if went out of the road
        return ['up', 'down', 'left', 'right', 'up-right', 'up-left', 'down-right', 'down-left']
    
    def apply(self, action):
        step = 1
        #print(self.state, action)
        up = lambda position: (max(position[0] - 1, 0), position[1])
        down = lambda position: (min(position[0] + 1, self.bounds[0] - 1), position[1])
        left = lambda position: (position[0], max(position[1] - 1, 0))
        right = lambda position: (position[0], min(position[1] + 1, self.bounds[1] - 1))
        up_right = lambda position: (max(position[0] - 1, 0), min(position[1] + 1, self.bounds[1] - 1))
        up_left = lambda position: (max(position[0] - 1, 0), max(position[1] - 1, 0))
        down_right = lambda position: (min(position[0] + 1, self.bounds[0] - 1), min(position[1] + 1, self.bounds[1] - 1))
        down_left = lambda position: (min(position[0] + 1, self.bounds[0] - 1), max(position[1] - 1, 0))

        robot, picks, drops, carries = self.state
        move_reward = self.rewards[0] if carries else self.rewards[0] * 5
        if   action == 'up': self.state = (up(robot), picks, drops, carries)  ; return move_reward
        elif action == 'down': self.state = (down(robot), picks, drops, carries) ; return move_reward
        elif action == 'left': self.state = (left(robot), picks, drops, carries) ; return move_reward
        elif action == 'right': self.state = (right(robot), picks, drops, carries) ; return move_reward
        elif action == 'up-right': self.state = (up_right(robot), picks, drops, carries) ; return move_reward
        elif action == 'up-left': self.state = (up_left(robot), picks, drops, carries) ; return move_reward
        elif action == 'down-right': self.state = (down_right(robot), picks, drops, carries) ; return move_reward
        elif action == 'down-left': self.state = (down_left(robot), picks, drops, carries) ; return move_reward
        
        
        elif action == 'pick':
          new_pos = []
          n_rem = 0
          for pos in picks :
            if pos == robot and n_rem < 1 :
              n_rem += 1
              continue 
            else :
              new_pos.append(pos)
          self.set_state((robot, tuple(new_pos), drops, True))
          return self.rewards[1]

        elif action == 'drop': 
          if len(picks) == 0 :
            self.set_state(None)
            print("Done!!")
          else : 
            new_pos = []
            n_rem = 0
            for pos in drops :
              if pos == robot and n_rem < 1 :
                n_rem += 1
                continue 
              else :
                new_pos.append(pos)
            self.set_state((robot, picks, tuple(new_pos), False))
          return self.rewards[2] if len(picks) == 0 else self.rewards[3]

        elif action == 'out' : 
          self.set_state(None)
          return self.rewards[4]
