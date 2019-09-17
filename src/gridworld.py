import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from collections import namedtuple
from matplotlib.patches import Polygon
import math
import random
import time
import gym
np.set_printoptions(precision=2, suppress=1)
plt.style.use('seaborn-notebook')

def flatten_tuple(input_tuple):
  """ Helper function: flattens tuple and its coordinates """
  x = [e for tupl in input_tuple for e in tupl]
  x_x = []
  x_y = []
  for i in range(len(x)):
      if i%2==0:
        x_x.append(x[i])
      else:
        x_y.append(x[i])
  x_min = min(x)
  x_x_max = max(x_x)
  x_y_max = max(x_y)
  return x, x_x, x_y, x_min, x_x_max, x_y_max


class GridWorld(object):
  """ Creates a GridWorld environment """
  def __init__(self, height, width, # spatial dimensions of the layout 
               auto_grid = False,    # places start at [0,0] and goal at [h,while]
               cliff_default = False, # places a cliff at the bottom of layout 
               start = None, #  custom start cell (tuple of 2d arrays)
               goals = None, # custom goal cells (tuple of 2d arrays)
               walls = None, # custom wall cells (tuple of 2d arrays)
               cliffs = None, # custom cliff cells (tuple of 2d arrays)
               render_scale = 1, # scale for rendering
               cont_reward = -1.,  #reward when continuation
               wall_reward = -1.,  #reward when hitting the wall
               cliff_reward = -100., #reward at goal
               discount = 1.0,    #payoff discount
               noisy = False    #randomness in steps -->TODO
               ): 
         
    self.height = height
    self.width = width
    self.auto_grid = auto_grid
    self.start = start
    self.goals = goals
    self.walls = walls
    self.cliffs = cliffs
    self.cliff_default = cliff_default
    self.cont_reward = cont_reward
    self.cliff_reward = cliff_reward
    self.wall_reward = wall_reward
    self.discount = discount
    self.render_scale = render_scale
    self.noisy = noisy
    self.done = False
    self.action_space = [0, 1, 2, 3]
    self.observation_space_n = self.height*self.width
    self.action_space_n = len(self.action_space)
    self.obs_grid = np.arange(self.width*self.height).reshape(self.height,self.width)
    self.check_inputs(self.height,
                      self.width,
                      self.start,
                      self.goals,
                      self.walls,
                      self.cliffs)
    
    #make layout
    self.layout = np.zeros((self.height + 2, self.width + 2))
    self.layout[0,:] = -1
    self.layout[-1,:]= -1
    self.layout[:,0] = -1
    self.layout[:,-1]= -1
    
    # Auto Grid
    if self.auto_grid is True:
      if self.cliff_default is True:
        #cliff
        self.layout[self.height, 2:self.width] = 7
      #goal
      self.layout[self.height, self.width] = 100 
      #initial state
      self.initial_state = (1, self.height) 
      self.state = self.initial_state

    # Custom Grid
    else:
      #start
      if self.start is not None:
        self.initial_state = (1 + self.start[1], self.height - self.start[0])
        self.layout[self.height - self.start[0], 1 + self.start[1]] = 50
        self.state = self.initial_state
      else:
        self.initial_state = (1, self.height) 
        self.layout[1, self.height] = 50
        self.state = self.initial_state

      #goals
      if self.goals is not None:
        if type(self.goals) is tuple:
          for goal in self.goals:
            self.layout[self.height - goal[0], 1 + goal[1]] = 100
        else:
          self.layout[self.height - self.goals[0], 1 + self.goals[1]] = 100
      else:
        self.layout[self.height, self.width] = 100
      
      #walls
      if self.walls is not None:
        if type(self.walls) is tuple:
          for wall in self.walls:
            self.layout[self.height - wall[0], 1 + wall[1]] = -1
        else:
          self.layout[self.height - self.walls[0], 1 + self.walls[1]] = -1

      #cliffs
      if self.cliffs is not None:
        if type(self.cliffs) is tuple:
          for cliff in self.cliffs:
            self.layout[self.height - cliff[0], 1 + cliff[1]] = 7
        else:
          self.layout[self.height - self.cliffs[0], 1 + self.cliffs[1]] = 7
  
  def check_inputs(self,
                   height,
                   width,
                   start,
                   goals,
                   walls,
                   cliffs):
    
    assert height>1, "Error: height should be >1"
    assert width>1, "Error: width should be >1"
    
    if self.cliffs is not None:
      if type(self.cliffs) is tuple:
        assert self.cliffs.count(self.start)==0, "Error: start shouldn't be located on cliffs"
        _, _, _, c_min, c_x_max, c_y_max = flatten_tuple(self.cliffs)
        assert c_min>=0, "Error: one or more cliffs are located outside the grid"
        assert c_x_max<=self.height, "Error: one or more cliffs are located outside the grid (check heights)"
        assert c_y_max<=self.width, "Error: one or more cliffs are located outside the grid (check widths)"
      else:
        assert self.cliffs != self.start, "Error: start shouldn't be located on cliffs"
        c_min = min(self.cliffs)
        assert c_min >=0,  "Error: one or more cliffs are located outside the grid"
        assert self.cliffs[0]<=self.height, "Error: one or more cliffs are located outside the grid (check heights)"
        assert self.cliffs[1]<=self.width, "Error: one or more cliffs are located outside the grid (check widths)"

    if self.walls is not None:
      if type(self.walls) is tuple:
        assert self.walls.count(self.start)==0, "Error: start shouldn't be located on walls"
        _, _, _, w_min, w_x_max, w_y_max = flatten_tuple(self.walls)
        assert w_min>=0, "Error: one or more walls are located outside the grid"
        assert w_x_max<=self.height, "Error: one or more walls are located outside the grid (check heights)"
        assert w_y_max<=self.width, "Error: one or more walls are located outside the grid (check widths)"
      else:
        assert self.walls != self.start, "Error: start shouldn't be located on walls"
        w_min = min(self.walls)
        assert w_min >=0,  "Error: one or more walls are located outside the grid"
        assert self.walls[0]<=self.height, "Error: one or more walls are located outside the grid (check heights)"
        assert self.walls[1]<=self.width, "Error: one or more walls are located outside the grid (check widths)"

    if goals is not None and start is not None:
      if type(self.goals) is tuple:
        assert self.goals.count(self.start)==0, "Error: start shouldn't be located on goals"
        _, _, _, g_min, g_x_max, g_y_max = flatten_tuple(self.goals)
        assert g_min>=0, "Error: one or more goals are located outside the grid"
        assert g_x_max<=self.height, "Error: one or more goals are located outside the grid (check heights)"
        assert g_y_max<=self.width, "Error: one or more goals are located outside the grid (check widths)"
      else:
        assert self.goals != self.start, "Error: start and goal should be in different cells"
        g_min = min(self.goals)
        assert g_min >=0,  "Error: one or more goals are located outside the grid"
        assert self.goals[0]<=self.height, "Error: one or more goals are located outside the grid (check heights)"
        assert self.goals[1]<=self.width, "Error: one or more goals are located outside the grid (check widths)"

  def reset(self):
    self.state = self.initial_state
    self.done = False
    return self.state
     
  def step(self, action):
    x_old, y_old = self.state 
     
    if action == 0: #up
      x, y = x_old, y_old-1
      self.new_state = (x, y)
    elif action == 1: #right
      x, y = x_old+1, y_old
      self.new_state = (x, y)         
    elif action == 2: #down
      x, y = x_old, y_old+1
      self.new_state = (x, y)
    elif action == 3: #left
      x, y = x_old-1, y_old
      self.new_state = (x, y)
    else:
      raise ValueError("Invalid action: ", action, "is not 0, 1, 2, 3")
      
    #hit the wall
    if self.layout[y, x] == -1:
      self.reward = self.wall_reward
      self.new_state = (x_old, y_old)
      self.done = False
    #hit the cliff
    elif self.layout[y, x] == 7:
      self.reward = self.cliff_reward
      self.new_state = self.initial_state
      self.done = False
    #reach the goal
    elif self.layout[y, x] == 100:
      self.reward = -1
      self.new_state = (x, y)
      self.done = True
    #any other transition
    elif self.layout[y, x] == 0:
      self.reward = self.cont_reward
      self.done = False
      
    self.state = self.new_state
    
    return self.state, self.reward, self.done
  
  def get_obs(self):
    x, y = self.state
    obs = self.obs_grid[y-1,x-1]
    return obs
  
  def get_state(self, obs):
    idx = np.where(self.obs_grid == obs)
    x, y = idx[(1)], idx[(0)]
    x, y = x[0], y[0]
    return x, y

  def action_space_sample(self):
    return random.choice(self.action_space)
  
  def render(self, mode = None):
    scale = self.render_scale
    sx, sy = self.state
    plt.figure(figsize=(scale*self.height, scale*self.width))
    plt.imshow(self.layout > -1, interpolation="nearest", cmap='pink')
    ax = plt.gca()
    ax.grid(0)
    plt.xticks([])
    plt.yticks([])
    plt.title("Grid World")

    h, w = self.layout.shape

    # *** Plot ***

    #state
    verts = [(sx-0.5, sy-0.5), (sx-0.5, sy+0.5), (sx+0.5, sy+0.5), (sx+0.5, sy-0.5) ]
    poly = Polygon(verts, facecolor='g', edgecolor='b')
    ax.add_patch(poly)
    #grids
    for y in range(h-1):
      plt.plot([-0.5, w-0.5], [y+0.5, y+0.5], '-k', lw=2)
    for x in range(w-1):
      plt.plot([x+0.5, x+0.5], [-0.5, h-0.5], '-k', lw=2)
      
    #auto grid
    if self.auto_grid is True:
      #start
      plt.text(1, h-2, r"$\mathbf{S}$", ha='center', va='center')
      #goal
      plt.text(w-2, h-2, r"$\mathbf{G}$", ha='center', va='center')
    
    #custom grid 
    else:
      #start
      if self.start is not None:
        plt.text(1 + self.start[1], h-self.start[0]-2 , r"$\mathbf{S}$", ha='center', va='center')
      else:
        plt.text(1, h-2, r"$\mathbf{S}$", ha='center', va='center')
      #goals
      if self.goals is not None:
        if type(self.goals) is tuple:
          for goal in self.goals:
            plt.text(1 + goal[1], h-goal[0]-2, r"$\mathbf{G}$", ha='center', va='center')
        else:
          plt.text(1 + self.goals[1], h-self.goals[0]-2, r"$\mathbf{G}$", ha='center', va='center')
      else:
        plt.text(w-2, h-2, r"$\mathbf{G}$", ha='center', va='center')
      #walls
      if self.walls is not None:
        if type(self.walls) is tuple:
          for wall in self.walls:
            plt.text(1 + wall[1], h-wall[0]-2, r"", ha='center', va='center')
        else:
          plt.text(1 + self.walls[1], h-self.walls[0]-2, r"", ha='center', va='center')
    
    #cliffs
    if self.cliff_default is True:
      #default cliff
      verts = [(1.5, h-1.5), (1.5, h-2.5), (w-2.5, h-2.5), (w-2.5, h-1.5)]
      poly = Polygon(verts, facecolor='r', edgecolor='0.5')
      ax.add_patch(poly)
      for i in range(self.width-2):
        plt.text(2+i, h-2, r"$\mathbf{C}$", ha='center', va='center') 
    else:
      if self.cliffs is not None:
        if type(self.cliffs) is tuple:
            for cliff in self.cliffs:
              plt.text(1 + cliff[1], h-cliff[0]-2, r"$\mathbf{C}$", ha='center', va='center')
              verts = [(1 + cliff[1]-0.5, h-cliff[0]-2+0.5), 
                      (1 + cliff[1]+0.5, h-cliff[0]-2+0.5),
                      (1 + cliff[1]+0.5, h-cliff[0]-2-0.5), 
                      (1 + cliff[1]-0.5, h-cliff[0]-2-0.5)]
              poly = Polygon(verts, facecolor='r', edgecolor='0.5')
              ax.add_patch(poly)
        else:
          plt.text(1 + self.cliffs[1], h-self.cliffs[0]-2, r"$\mathbf{C}$", ha='center', va='center')
          verts = [(1 + self.cliffs[1]-0.5, h-self.cliffs[0]-2+0.5), 
                  (1 + self.cliffs[1]+0.5, h-self.cliffs[0]-2+0.5),
                  (1 + self.cliffs[1]+0.5, h-self.cliffs[0]-2-0.5), 
                  (1 + self.cliffs[1]-0.5, h-self.cliffs[0]-2-0.5)]
          poly = Polygon(verts, facecolor='r', edgecolor='0.5')
          ax.add_patch(poly)

    if mode == 'simple_render':
        plt.show()
        
    if mode == 'episode':    
        plt.show(block=False)
        plt.pause(0.3)
        plt.close()
