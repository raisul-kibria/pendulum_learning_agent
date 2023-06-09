#!/usr/bin/env python

import argparse
import os
import numpy as np
from utils.inverted_pendulum import InvertedPendulum, PARAMS, DEBUG
from utils.utils import print_policy, plot_curve, calculate_longest_streak, create_output_dir

def update_policy(observation, policy_matrix, state_action_matrix, tot_bins):
    """Update a policy making it greedy in respect of the state-action matrix.

    @return the updated policy
    """
    col = observation[1] + (observation[0]*tot_bins)
    if(policy_matrix[observation[0], observation[1]] != -1):      
        policy_matrix[observation[0], observation[1]] = np.argmax(state_action_matrix[:,col])
    return policy_matrix

def return_decayed_value(starting_value, minimum_value, global_step, decay_step):
    """Returns the exponentially decayed value.

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
    """updates the state_action matrix using the SARSA(0)
    Q-matrix equation.

    @return updated state_action matrix
    """
    col = observation[1] + (observation[0]*tot_bins)
    qt = state_action_matrix[int(action), col]
    coltp1 = new_observation[1] + (new_observation[0]*tot_bins)
    qtp1 = state_action_matrix[int(new_action), coltp1]
    delta = reward + gamma * qtp1 - qt
    state_action_matrix[int(action), col] += alpha * delta
    return state_action_matrix

def parse_opt():
    """function to add support for command line parameters.
    `python sarsa_0_inverted_pendulum.py --help` for details.
    """
    parser = argparse.ArgumentParser()
    parser.add_argument('--pole_mass', type=float, default=PARAMS['pole_mass'], help='the mass of the pole in kilograms')
    parser.add_argument('--cart_mass', type=float, default=PARAMS['cart_mass'], help='the mass of the cart in kilograms')
    parser.add_argument('--pole_lenght', type=float, default=PARAMS['pole_lenght'], help='length of the pole in meters')
    parser.add_argument('--delta_t', type=float, default=PARAMS['delta_t'], help='time steps')
    parser.add_argument('--alpha', type=float, default=PARAMS['alpha'], help='learning rate alpha')
    parser.add_argument('--tot_episode', type=int, default=PARAMS['tot_episode'], help='number of episodes to run the simulation')
    opt = parser.parse_args()
    return opt

def main(opt):

    # initializing parameters from passed arguments
    pole_mass, cart_mass, pole_lenght, delta_t, alpha, tot_episode = vars(opt).values()
    print("Starting simulation ...")
    print(vars(opt))

    # directory to store experiment-specific results
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

    # set parameters
    gamma = 0.999
    epsilon_start = 0.99  # those are the values for epsilon decay
    epsilon_stop = 0.1
    epsilon_decay_step = 10000
    print_episode = 500  # print every...
    movie_episode = 20000  # movie saved every...
    if not DEBUG:
        print_episode = tot_episode
        movie_episode = tot_episode // 5
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
            #Update policy with greedy strategy on the state-action matrix
            policy_matrix = update_policy(observation, policy_matrix, state_action_matrix, tot_bins)
            observation = new_observation
            cumulated_reward += reward
            if done: break
        if not first_hundred and not done:
            # first sucess during the experiment
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
            _ = print_policy(policy_matrix)
        if(episode % movie_episode == 0) or (episode == tot_episode - 1):
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
    ps, fm = print_policy(policy_matrix)

    # Metrics for comparative analysis
    streak, start_i, end_i = calculate_longest_streak(step_list)
    with open(os.path.join(OUTPUT_DIR, 'metrics.txt'), 'w') as f:
        f.writelines(f'SARSA ZERO\nPARAMETERS: {vars(opt)}\n\
                        Mean Number of steps: {np.mean(step_list)}\n\
                        Median of steps: {np.median(step_list)}\n\
                        Longest Streak of success: {streak} [{start_i}:{end_i}]\n\
                        Success ratio: {np.sum([1 for i in step_list if i+1 == 100]) / tot_episode}\n\
                        First Success: {first_hundred}\n\
                        Policy:\n{ps}\n\n\
                        Matrix:\n{fm}')
    np.save(os.path.join(OUTPUT_DIR, 'Q_matrix.npy'), state_action_matrix)

if __name__ == "__main__":
    opt = parse_opt()
    main(opt)
