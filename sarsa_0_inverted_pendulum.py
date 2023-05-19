#!/usr/bin/env python

import argparse
import os
import numpy as np
from inverted_pendulum import InvertedPendulum, OUTPUT_DIR, PARAMS, DEBUG
import matplotlib.pyplot as plt

def print_policy(policy_matrix):
    """Print the policy using specific symbol.

    O noop, < left, > right
    """
    counter = 0
    shape = policy_matrix.shape
    policy_string = ""
    for row in range(shape[0]):
        for col in range(shape[1]):           
            if(policy_matrix[row,col] == 0): policy_string += " <  "
            elif(policy_matrix[row,col] == 1): policy_string += " O  "
            elif(policy_matrix[row,col] == 2): policy_string += " >  "           
            counter += 1
        policy_string += '\n'
    print(policy_string)

def get_return(state_list, gamma):
    """Get the return for a list of action-state values.

    @return get the Return
    """
    counter = 0
    return_value = 0
    for visit in state_list:
        reward = visit[2]
        return_value += reward * np.power(gamma, counter)
        counter += 1
    return return_value

def update_policy(observation, policy_matrix, state_action_matrix, tot_bins):
    """Update a policy making it greedy in respect of the state-action matrix.

    @return the updated policy
    """
    col = observation[1] + (observation[0]*tot_bins)
    if(policy_matrix[observation[0], observation[1]] != -1):      
        policy_matrix[observation[0], observation[1]] = np.argmax(state_action_matrix[:,col])
    return policy_matrix

def return_decayed_value(starting_value, minimum_value, global_step, decay_step):
    """Returns the decayed value.

    decayed_value = starting_value * decay_rate ^ (global_step / decay_steps)
    @param starting_value the value before decaying
    @param global_step the global step to use for decay (positive integer)
    @param decay_step the step at which the value is decayed
    """
    decayed_value = starting_value * np.power(0.9, (global_step/decay_step))
    if decayed_value < minimum_value:
            return minimum_value
    else:
            return decayed_value

def return_epsilon_greedy_action(policy_matrix, observation, epsilon=0.1):
    """Return an action choosing it with epsilon-greedy

    @param policy_matrix the matrix before the update
    @param observation the state obsrved at t
    @param epsilon the value used for computing the probabilities
    @return the updated policy_matrix
    """
    tot_actions = int(np.nanmax(policy_matrix) + 1)
    action = int(policy_matrix[observation[0], observation[1]])
    non_greedy_prob = epsilon / tot_actions
    greedy_prob = 1 - epsilon + non_greedy_prob
    weight_array = np.full((tot_actions), non_greedy_prob)
    weight_array[action] = greedy_prob
    act = np.random.choice(tot_actions, 1, p=weight_array)
    return act[0]

def update_state_action(state_action_matrix, observation, new_observation,
    action, new_action, reward, alpha, gamma, tot_bins):
    col = observation[1] + (observation[0]*tot_bins)
    qt = state_action_matrix[int(action), col]
    coltp1 = new_observation[1] + (new_observation[0]*tot_bins)
    qtp1 = state_action_matrix[int(new_action), coltp1]
    delta = reward + gamma * qtp1 - qt
    state_action_matrix[int(action), col] += alpha * delta
    return state_action_matrix

def plot_curve(data_list, filepath="./my_plot.png", 
               x_label="X", y_label="Y", 
               x_range=(0, 1), y_range=(0,1), color="-r", kernel_size=50, alpha=0.4, grid=True, first_hundred=None):
        """Plot a graph using matplotlib

        """
        if(len(data_list) <=1):
            print("[WARNING] the data list is empty, no plot will be saved.")
            return
        fig = plt.figure()
        ax = fig.add_subplot(111, autoscale_on=False, xlim=x_range, ylim=y_range)
        ax.grid(grid)
        ax.set_xlabel(x_label)
        ax.set_ylabel(y_label)
        ax.plot(data_list, color, alpha=alpha)  # The original data is showed in background
        
        kernel = np.ones(int(kernel_size))/float(kernel_size)  # Smooth the graph using a convolution
        tot_data = len(data_list)
        lower_boundary = int(kernel_size/2.0)
        upper_boundary = int(tot_data-(kernel_size/2.0))
        data_convolved_array = np.convolve(data_list, kernel, 'same')[lower_boundary:upper_boundary]
        #print("arange: " + str(np.arange(tot_data)[lower_boundary:upper_boundary]))
        #print("Convolved: " + str(np.arange(tot_data).shape))
        ax.plot(np.arange(tot_data)[lower_boundary:upper_boundary], data_convolved_array, color, alpha=1.0)  # Convolved plot
        if first_hundred:
           var = (first_hundred / tot_data) * (upper_boundary - lower_boundary) + lower_boundary
           ax.axvline(x=var, ymin=0, ymax=data_convolved_array[var], color='g', linestyle='--')
        fig.savefig(filepath)
        fig.clear()
        plt.close(fig)
        # print(plt.get_fignums())  # print the number of figures opened in background

def parse_opt():
    parser = argparse.ArgumentParser()
    parser.add_argument('--pole_mass', type=float, default=PARAMS['pole_mass'], help='the mass of the pole in kilograms')
    parser.add_argument('--cart_mass', type=float, default=PARAMS['cart_mass'], help='the mass of the cart in kilograms')
    parser.add_argument('--pole_lenght', type=float, default=PARAMS['pole_lenght'], help='length of the pole in meters')
    parser.add_argument('--delta_t', type=float, default=PARAMS['delta_t'], help='time steps')
    parser.add_argument('--alpha', type=float, default=PARAMS['alpha'], help='learning rate alpha')
    parser.add_argument('--tot_episode', type=int, default=PARAMS['tot_episode'], help='number of episodes to run the simulation')
    opt = parser.parse_args()
    return opt

def create_output_dir(alpha, epoch):
    OUTPUT_DIR = f"./output_group4_SARSA0_alpha_{alpha}_epoch_{epoch}"
    try:
        os.makedirs(OUTPUT_DIR)
    except:
        pass
    return OUTPUT_DIR

def main(opt):
    pole_mass, cart_mass, pole_lenght, delta_t, alpha, tot_episode = vars(opt).values()
    print(opt)
    OUTPUT_DIR = create_output_dir(alpha=alpha, epoch=tot_episode)

    env = InvertedPendulum(
        pole_mass=pole_mass,
        cart_mass=cart_mass,
        pole_lenght=pole_lenght,
        delta_t=delta_t)

    # Define the state arrays for velocity and position
    tot_action = 3  # Three possible actions
    tot_bins = 12  # the value used to discretize the space
    velocity_state_array = np.linspace(-np.pi, np.pi, num=tot_bins-1, endpoint=False)
    position_state_array = np.linspace(-np.pi/2.0, np.pi/2.0, num=tot_bins-1, endpoint=False)

    #Random policy
    policy_matrix = np.random.randint(low=0, high=tot_action, size=(tot_bins,tot_bins))
    if DEBUG:
        print("Policy Matrix:")
        print_policy(policy_matrix)

    state_action_matrix = np.zeros((tot_action, tot_bins*tot_bins))
    gamma = 0.999
    epsilon_start = 0.99  # those are the values for epsilon decay
    epsilon_stop = 0.1
    epsilon_decay_step = 10000
    print_episode = 500  # print every...
    movie_episode = 20000  # movie saved every...
    if not DEBUG:
        print_episode = tot_episode
        movie_episode = tot_episode / 4
    reward_list = list()
    step_list = list()
    first_hundred = None

    for episode in range(tot_episode):
        epsilon = return_decayed_value(epsilon_start, epsilon_stop, episode, decay_step=epsilon_decay_step)
        #Reset and return the first observation and reward
        observation = env.reset(exploring_starts=True)
        observation = (np.digitize(observation[1], velocity_state_array), 
                       np.digitize(observation[0], position_state_array))
        #Starting a new episode
        is_starting = True
        cumulated_reward = 0
        for step in range(100):
            #Take the action from the action matrix
            action = return_epsilon_greedy_action(policy_matrix, observation, epsilon=epsilon)
            #If the episode just started then it is
                #necessary to choose a random action (exploring starts)
            if(is_starting): 
                action = np.random.randint(0, tot_action)
                is_starting = False   
            #Move one step in the environment and get obs and reward
            new_observation, reward, done = env.step(action)
            new_observation = (np.digitize(new_observation[1], velocity_state_array), 
                               np.digitize(new_observation[0], position_state_array))  
            #Append the visit in the episode list
            new_action = int(policy_matrix[new_observation[0], new_observation[1]])
            state_action_matrix = update_state_action(state_action_matrix, observation, new_observation,
                                                      action, new_action, reward, alpha, gamma, tot_bins)
            policy_matrix = update_policy(observation, policy_matrix, state_action_matrix, tot_bins)
            observation = new_observation
            cumulated_reward += reward
            if done: break
        if not first_hundred and not done:
            first_hundred = episode
        reward_list.append(cumulated_reward)
        step_list.append(step)
        # Printing utilities
        if(episode % print_episode == 0):
            print("")
            print("Episode: " + str(episode+1))
            print("Epsilon: " + str(epsilon))
            print("Episode steps: " + str(step+1))
            print("Cumulated Reward: " + str(cumulated_reward))
            print("Policy matrix: ") 
            print_policy(policy_matrix)
        if(episode % movie_episode == 0):
            print(f"Saving the reward plot in: {OUTPUT_DIR}/reward_SARSA0.png")
            plot_curve(reward_list, filepath=f"{OUTPUT_DIR}/reward_SARSA0.png", 
                       x_label="Episode", y_label="Reward",
                       x_range=(0, len(reward_list)), y_range=(-0.1,100),
                       color="red", kernel_size=500, 
                       alpha=0.4, grid=True, first_hundred=first_hundred)
            print(f"Saving the step plot in: {OUTPUT_DIR}/step_SARSA0.png")
            plot_curve(step_list, filepath=f"{OUTPUT_DIR}/step_SARSA0.png", 
                       x_label="Episode", y_label="Steps", 
                       x_range=(0, len(step_list)), y_range=(-0.1,100),
                       color="blue", kernel_size=500, 
                       alpha=0.4, grid=True, first_hundred=first_hundred)
            print(f"Saving the gif in: {OUTPUT_DIR}/inverted_pendulum_SARSA0.gif")
            env.render(file_path=f'{OUTPUT_DIR}/inverted_pendulum_SARSA0.gif', mode='gif')
            print("Complete!")

    print("Policy matrix after " + str(tot_episode) + " episodes:")
    print_policy(policy_matrix)

if __name__ == "__main__":
    opt = parse_opt()
    main(opt)
