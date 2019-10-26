[//]: # (Image References)

[logo]: https://github.com/pytrainai/gridworld/blob/master/assets/logo.png
[example1]: https://github.com/pytrainai/gridworld/blob/master/figures/example_1.png "Grid for example 1"
[example2]: https://github.com/pytrainai/gridworld/blob/master/figures/example_2.png "Grid for example 2"
[example3]: https://github.com/pytrainai/gridworld/blob/master/figures/example_3.png "Grid for example 3"

# YAGWE(BHHC)

## (Yet Another Grid World Environment (but, hey, highly customizable!))

by Nick Ben (PytrainAI founder, assistant VP and HR manager (assistant)/facility sub-manager. Also a vice hat-trick skills "mentor, speaker and influencer" :-)

### Introduction

This is a simple but highly customizable implementation of a Grid World in Python using Matplotlib and following the spirit of OpenAI Gym environments.


The grid can be generated by specifying:
- **`height`** - height of the gridworld
- **`width`** - width of the gridworld
- **`start`** - cell where agent will start after resetting or reaching goals
- **`goals`** - goal cells, once reached the episode will finish and agent return to start
- **`walls`** - wall cells, will stop the agent from moving and penalize with `wall_reward`
- **`cliffs`** - cliff cells, will make the agent go back to start and penalize with `cliff_reward`
- **`continuation_reward`** - reward when simply moving (default = -1)
- **`cliff_reward`** - reward when falling into a cliff cell (default = -100)
- **`wall_reward`** - reward when bumping into a wall cell (default = -1)

It is also possible to construct a simple cliff gridworld (same as in OpenAI Gym) by setting `auto_grid=True`.

The environment, once instantiated, allow for the agent to navigate towards the grid by calling to the usual environment methods:

- `reset` - environment will be reset and agent will go to start position
- `step` - environment will take one single step acording to the action passed, and will return `next_state`, `reward` and `done`.

### Getting Started

1. Clone this repo and install all the requirements 

2. Run the test script or the sample notebook to see a demonstration of different grids:

![Example 1][example1]

![Example 2][example2]

![Example 3][example3]


### Roadmap

Next versions are planning to include:

 - Plotting a learned value function over the grid
 - Noisy steps, controlled by a noise factor
 - Jump cells, both deterministic and random jumps
 - Doors to be open after collecting keys
 - Connected grids (in order to allow Hierarchical learning)


### Collaboration and cites

You are more than welcome to colaborate! Please feel free to reach me out at pytrainteam@gmail.com. If you modify this code or use it for personal purposes don't forget to cite ;)
