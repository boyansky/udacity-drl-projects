

# Summary
The project involved training an agent to navigate in a 3D environment - collecting specific items while avoiding others. The objective was to use value-based and model-free reinforcement learning models and more specifically Deep Q-learning.

I have successfully implemented Deep Q-learning and have reached the project goal. As a second step I have implemented Double Deep Q-learning to achieve improved performance. The advantage of DDQN over DQN is the implementation of two separate networks in choosing and evaluating the action. in this way, the model mitigates the overestimation of the value function.

|Random agent|Trained DDQN agent|
|------------|-------------|
|![Random Agent](gifs/random_agent.gif)|![Trained Agent](gifs/trained_agent.gif)|

# Model architecture
The objective of Q-learning is to learn an action-value function `Q(s,a)` in order to pick the best action among all possible actions `a` in any given state `s`.

Given the space in the given environment is continious, we cannot use tabular representation to calcualte the Q values. We use a function approximation instead by introducing a new parameter $\theta$ - $\hat{Q} (s, a; \theta)$.

In the current project we use neural network to calculate the function. The chosen architecture is 2 fully-connecterd hidden layers with 64 units and `relu` activation function. Mean-square error is used as a loss function and `Adam` - as a optimisation for finding the weights.

For improving the training, two techniques are incorporated -
fixed Q-targets and experience replay.

The results are further imrpoved by introducing a variout of the Deep Q-learning called Double DQN. The idea behind the model is to separate the calcualtion of the Q-target - one neural network is used for choosing the best action and another one for evaluating the action. The intuition behind is to correct over-optimistic Q-values.

Implementation of the model is done in `PyTorch`.

# Hyperparameters

| Hyperparameter                      | Value |
| ----------------------------------- | ----- |
| Replay buffer size                  | 1e5   |
| Batch size                          | 64    |
| $\gamma$ (discount factor)          | 0.99  |
| $\tau$                              | 1e-3  |
| Learning rate                       | 5e-4  |
| update interval                     | 4     |
| Number of episodes                  | 5000  |
| Max number of timesteps per episode | 1000  |
| Epsilon start                       | 1.0   |
| Epsilon minimum                     | 0.01  |
| Epsilon decay                       | 0.997 |

# Results

## DQN
Trained in 1200 episodes and an average training score of 15.34. Average score of playing with the trained agent is 13.05.

![DQN Results](dqn_training_results.jpg)

## DDQN
Trained in 914 episodes and an average training score of 15.04. Average score of playing with the trained agent is 14.60.

![DDQN Results](ddqn_training_results.jpg)

## Takeaway
The agent trained with Double DQN took less time to reach the target score and with lower variance in the training average scores across the episodes.

Also the trained DDQN agent performed better on average when playing.

## Ideas for improving the performance
- **Hyperparater adjustment** of the current implementation can lead to better results
- Introducing **Dueling DQN** can also lead to significant imrpovements. The model splits the Q-value calculation by having two output streams for calculating separately the state values `V(s)` and the advantage values of each action `A(s,a)`. The intuition here is that you can have a pretty bad state where it doesn't matter that much which action you take.
- Using **Prioritized Experience Replay** has previously shown a big improvement over Duble DQN. The idea behind is that there are some experiences that are more important than others. Sampling uniformally and with limited buffer capacity can lead to lower chances of getting these experiences and loosing older important experiences. This can be overcome by assinging priority values to each experience tuple.
