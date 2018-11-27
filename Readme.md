## Installation
```bash
$ git clone 
```

```bash
pip install -r requirements.txt
```

## Usage
1. To train an agent from scratch
```bash
$ python ddpg.py --run=train --env-path=/path/to/reacher 
```

2. To train an agent by reloading the weights
```bash
$ python ddpg.py 
--run=train 
--env-path=/path/to/reacher 
--actor-model-path=/path/to/checkpoint_actor.pth 
--critic-model-path=/path/to/checkpoint_critic.pth
```

3. To run a trained model
```bash
$ python ddpg.py 
--run=test 
--env-path=/path/to/reacher 
--actor-model-path=/path/to/checkpoint_actor.pth 
--critic-model-path=/path/to/checkpoint_critic.pth
```

## Introduction
In this environment, two agents control rackets to bounce a ball over a net. 
If an agent hits the ball over the net, it receives a reward of +0.1. 
If an agent lets a ball hit the ground or hits the ball out of bounds, it receives a reward of -0.01. 
Thus, the goal of each agent is to keep the ball in play.

#### State space
The observation space consists of 8 variables corresponding to the position and velocity of the ball and racket. 
Each agent receives its own, local observation. 


#### Action space
Two continuous actions are available, corresponding to movement toward (or away from) the net, and jumping.


## Solution
The task is episodic, and in order to solve the environment, 
the agents must get an average score of +0.5 
(over 100 consecutive episodes, after taking the maximum over both agents). 

Specifically, after each episode, add up the rewards that each agent received (without discounting), 
to get a score for each agent. This yields 2 (potentially different) scores. 
Then take the maximum of these 2 scores.
This yields a single score for each episode.

The environment is considered solved, when the average (over 100 episodes) of those scores is at least +0.5.


# Algorithm: [DDPG](https://arxiv.org/abs/1509.02971)
1. Randomly initialize critic network Q(s, a|θQ) and actor µ(s|θµ) 
with weights θQ and θµ.

2. Initialize target network Q' and µ' with weights: 
θQ' ← θQ, 
θµ' ← θµ

3. Initialize replay buffer R
4. for episode = 1, M do
    1. Initialize a random process N for action exploration
    2. Receive initial observation state s1
        1. for t = 1, T do
            1. Select action a(t) = µ(st|θµ) + Nt according to the current policy and 
                exploration noise
            2. Execute action at and observe reward rt and observe new state st+1
            3. Store transition (st, at, rt, st+1) in R
            4. Sample a random minibatch of N transitions (si, ai, ri, si+1) from R
            5. Update critic by minimizing the loss
            6. Update the actor policy using the sampled policy gradient
            7. Update the target networks
        2. end for
    3. end for


# Network architecture

## 1. Models

### 1.1 Actor Model (actor-target and actor-local have identical architecture)
```
Actor(
  (fc1): Linear(in_features=33, out_features=256, bias=True)
  (bn1): BatchNorm1d(256, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
  (d1): Dropout(p=0.1)
  (fc2): Linear(in_features=256, out_features=256, bias=True)
  (d2): Dropout(p=0.1)
  (fc3): Linear(in_features=256, out_features=4, bias=True)
)
```

### 1.2 Critic Model (critic-target and critic-local have identical architecture)
```
Critic(
  (fcs1): Linear(in_features=33, out_features=256, bias=True)
  (bn1): BatchNorm1d(256, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
  (d1): Dropout(p=0.1)
  (fc2): Linear(in_features=260, out_features=256, bias=True)
  (d2): Dropout(p=0.1)
  (fc3): Linear(in_features=256, out_features=1, bias=True)
)
```

## Hyper-parameters
```
1. BUFFER_SIZE      = int(1e5)  # replay buffer size
2. BATCH_SIZE       = 128       # minibatch size
3. GAMMA            = 0.99      # discount factor
4. TAU              = 1e-3      # for soft update of target parameters
5. ACTOR_LR         = 1e-4      # learning rate of the actor
6. CRITIC_LR        = 1e-4      # learning rate of the critic
8. FC1_UNITS        = 256       # Number of Neurons in the first hidden layer of both the actor and critic networks
9. FC2_UNITS        = 256       # Number of Neurons in the second hidden layer of both the actor and critic networks 
10. TIME_STEPS      = 5000      # Number of time-steps in an episode
```

### Experience
DDPG Algorithms need a lot of episodes to solve environments
which means slow convergence of neural networks is a huge bottleneck.
The model trained in this example took ~3 hours to hit 30+ points 
over 300 episodes. 

Previous attempts without batch-normalization had even poor 
convergence results with over 100s of episodes fed, a reward of 1 was 
hard to maintain for the network.

Gradient clipping also helped to keep the gradients from exploding as a
ReLU activation function was chosen.

![scores](https://github.com/AmreshVenugopal/drlnd_continous_control/blob/master/scores_over_100.png)