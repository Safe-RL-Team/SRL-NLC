## HazardWorld

HazardWorld is a 2D Gridworld reinforcement learning environment designed to
test the language capabilities of safe reinforcement learning agents.
In HazardWorld, agents must collect a series of rewards, while avoiding
unsafe states specified in natural language.

## Installation 

In the top-level directory, run:

```
pip3 install -e .
```

## Basic Usage 

There is a UI application which allows you to manually control the agent with the arrow keys:

```
./manual_control.py
```

The environment being run can be selected with the `--env` option, eg:

```
./manual_control.py --env MiniGrid-HazardWorld-R-v0
```
## Training with Reinforcement Learning Agents

After installation, HazardWorld is registered as an environment with OpenAI gym. 
Simply import `gym_minigrid` and call a HazardWorld environment as you would 
any other gym environment. 

```
env = gym.make('MiniGrid-HazardWorld-BA-v0')
```