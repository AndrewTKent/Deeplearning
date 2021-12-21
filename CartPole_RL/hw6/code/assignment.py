from reinforce_with_baseline import ReinforceWithBaseline
from reinforce import Reinforce

import os
import sys
import gym

import matplotlib as plt
import tensorflow as tf
import numpy as np

from pylab import *

def visualize_episode(env, model):
    """
    HELPER - do not edit.
    Takes in an enviornment and a model and visualizes the model's actions for one episode.
    We recomend calling this function every 20 training episodes. Please remove all calls of 
    this function before handing in.

    :param env: The cart pole enviornment object
    :param model: The model that will decide the actions to take
    """

    done = False
    state = env.reset()
    env.render()

    while not done:
        newState = np.reshape(state, [1, state.shape[0]])
        prob = model.call(newState)
        newProb = np.reshape(prob, prob.shape[1])
        action = np.random.choice(np.arange(newProb.shape[0]), p = newProb)

        state, _, done, _ = env.step(action)
        env.render()

def visualize_data(total_rewards):
    """
    HELPER - do not edit.
    Takes in array of rewards from each episode, visualizes reward over episodes
    
    :param total_rewards: List of rewards from all episodes
    """

    x_values = arange(0, len(total_rewards), 1)
    y_values = total_rewards
    plot(x_values, y_values)
    xlabel('episodes')
    ylabel('cumulative rewards')
    title('Reward by Episode')
    grid(True)
    show()


def discount(rewards, discount_factor=.99):
    """
    Takes in a list of rewards for each timestep in an episode, and
    returns a list of the discounted rewards for each timestep, which
    are calculated by summing the rewards for each future timestep, discounted
    by how far in the future it is.
    For example, in the simple case where the episode rewards are [1, 3, 5] 
    and discount_factor = .99 we would calculate:
    dr_1 = 1 + 0.99 * 3 + 0.99^2 * 5 = 8.8705
    dr_2 = 3 + 0.99 * 5 = 7.95
    dr_3 = 5
    and thus return [8.8705, 7.95 , 5].
    Refer to the slides for more details about how/why this is done.

    :param rewards: List of rewards from an episode [r_{t1},r_{t2},...]
    :param discount_factor: Gamma discounting factor to use, defaults to .99
    :returns: discounted_rewards: list containing the discounted rewards for each timestep in the original rewards list
    """

    discounted = [rewards[-1]]
    
    for i in range(len(rewards) - 2, -1, -1):
        
        temp_reward = rewards[i] + discount_factor * discounted[0]
        discounted.insert(0, temp_reward)

    return discounted


def generate_trajectory(env, model):
    """
    Generates lists of states, actions, and rewards for one complete episode.
    :param  env: The openai gym environment
    :param  model: The model used to generate the actions
    :return A tuple of lists (states, actions, rewards), where each list has length equal to
            the number of timesteps in the episode
    """
    states = []
    actions = []
    rewards = []
    state = env.reset()
    done = False

    while not done:
        # TODO:
        # 1) use model to generate probability distribution over next actions
        state = tf.reshape(state, (1, state.shape[0]))
        probabilities = model(state)
        probabilities = np.reshape(probabilities,(model.num_actions))
        
        # 2) sample from this distribution to pick the next action
        action = np.random.choice(np.arange(model.num_actions), p=probabilities)  

        states.append(state)
        actions.append(action)
        state, rwd, done, _ = env.step(action)
        rewards.append(rwd)
        # show env to see how it plays
        # env.render()

    return states, actions, rewards


def train(env, model):
    """
    This function should train your model for one episode.
    Each call to this function should generate a complete trajectory for one episode (lists of states, action_probs,
    and rewards seen/taken in the episode), and then train on that data to minimize your model loss.
    Make sure to return the total reward for the episode.
    :param  env: The openai gym environment
    :param  model: The model
    :return The total reward for the episode
    """
    # TODO:
    # 1) Use generate trajectory to run an episode and get states, actions, and rewards.
    # 2) Compute discounted rewards.
    # 3) Compute the loss from the model and run backpropagation on the model.
    
    with tf.GradientTape() as tape:
        
        states, actions, rewards = generate_trajectory(env, model)
        discounted_rewards = discount(rewards)
        current_loss = model.loss(tf.convert_to_tensor(states), tf.convert_to_tensor(actions), tf.convert_to_tensor(discounted_rewards))
    
    gradients = tape.gradient(current_loss, model.trainable_variables)
    model.optimizer.apply_gradients(zip(gradients, model.trainable_variables))

    return np.sum(rewards)


def main():
    
    #sys.argv = ["REINFORCE", "REINFORCE"]
    #sys.argv = ["REINFORCE_BASELINE", "REINFORCE_BASELINE"]
    if len(sys.argv) != 2 or sys.argv[1] not in {"REINFORCE", "REINFORCE_BASELINE"}:
        print("USAGE: python assignment.py <Model Type>")
        print("<Model Type>: [REINFORCE/REINFORCE_BASELINE]")
        exit()

    env = gym.make("CartPole-v1")  # environment
    state_size = env.observation_space.shape[0]
    num_actions = env.action_space.n

    # Initialize model
    if sys.argv[1] == "REINFORCE":
        model = Reinforce(state_size, num_actions) 
    elif sys.argv[1] == "REINFORCE_BASELINE":
        model = ReinforceWithBaseline(state_size, num_actions)

    # TODO: 
    # 1) Train your model for 650 episodes, passing in the environment and the agent.
    # 2) Append the total reward of the episode into a list keeping track of all of the rewards.
    # 3) After training, print the average of the last 50 rewards you've collected.
    epochs = 650
    total_rewards = []
    
    print('Training Starting:')
    
    for epoch in range(epochs):
        
        total_reward = train(env, model)
        total_rewards.append(total_reward)
        
        print('\r', 'Training Completion: {0:.2f} %'.format(epoch / epochs * 100), end='')
        
    print('\nThe Average of Last 50 Rewards: {}'.format(np.mean(total_rewards[-50:])))

    # TODO: Visualize your rewards.
    visualize_data(total_rewards)


if __name__ == '__main__':
    main()