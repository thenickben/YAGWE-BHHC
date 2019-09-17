from gridworld import GridWorld

def example_1():
    #example 1
    height = 6
    width = 2
    start = [5,0]
    goals = ([5,0])
    walls = ([2,1],[2,2],[2,3],
             [3,1],[3,2],[3,3])
    cliffs = ([1,1],[1,2],[1,3])
    env = GridWorld(height, width, False, False, start, goals, walls, cliffs)
    env.render(mode = 'simple_render')

def example_2():
    #example 2
    height = 6
    width = 6
    env = GridWorld(height, width, True, True)
    env.render(mode = 'simple_render')

def example_3():
    #example 3
    height = 3
    width = 3
    start = [0,0]
    goals = ([2,2])
    walls = None
    cliffs = None
    env = GridWorld(height, width, False, False, start, goals, walls, cliffs)
    env.render(mode = 'simple_render')

#Example 1-step random agent
def random_play(n_steps):
    #env from example_3
    height = 3
    width = 3
    start = [0,0]
    goals = ([2,2])
    walls = None
    cliffs = None
    env = GridWorld(height, width, False, False, start, goals, walls, cliffs)
    
    #random actions over n_steps:
    env.reset()
    for step in range(n_steps):
        action = env.action_space_sample()
        new_state, reward, done = env.step(action)
        print("Step:", step, ", Action:",action, ", New state:", env.get_obs(), ", Done:",done, ", Reward:", reward)
        env.render(mode = 'episode')

if __name__ == '__main__':
    example_1()
    #example_2()
    #example_3()
    #random_play(10)