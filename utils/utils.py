import matplotlib.pyplot as plt
import os
import numpy as np

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
            var = int((first_hundred / tot_data) * (upper_boundary - lower_boundary) + lower_boundary)
            ax.axvline(x=var, ymin=y_range[0], ymax=data_convolved_array[var - lower_boundary] / y_range[-1], color='g', linestyle='--')
        fig.savefig(filepath)
        fig.clear()
        plt.close(fig)
        # print(plt.get_fignums())  # print the number of figures opened in background

def calculate_longest_streak(numbers):
    """calculate the streak of consecutive successes (step=100).
    @param numbers list of all the steps in total episodes

    @return the starting and last index of the streak
    and the length
    """
    longest_streak = 0
    current_streak = 0
    streak_start = None
    streak_end = None

    for i, num in enumerate(numbers):
        if num+1 == 100:
            if current_streak == 0:
                streak_start = i
            current_streak += 1
            if current_streak > longest_streak:
                longest_streak = current_streak
                streak_end = i
        else:
            current_streak = 0

    if longest_streak == 0:
        streak_start = -1
        streak_end = -1
    if streak_start > streak_end:
        streak_start = streak_end - longest_streak
    return longest_streak, streak_start, streak_end

def print_policy(policy_matrix):
    """Print the policy using specific symbol.

    O noop, < left, > right
    """
    counter = 0
    shape = policy_matrix.shape
    policy_string = ""
    formatted_matrix = ""
    for row in range(shape[0]):
        for col in range(shape[1]):           
            if(policy_matrix[row,col] == 0): policy_string += " <  "
            elif(policy_matrix[row,col] == 1): policy_string += " O  "
            elif(policy_matrix[row,col] == 2): policy_string += " >  "
            formatted_matrix +=  f"{policy_matrix[row,col]:.4f}  "    
            counter += 1
        policy_string += '\n'
        formatted_matrix += '\n'
    print(policy_string)
    return policy_string, formatted_matrix

def create_output_dir(alpha, epoch, epsilon_strategy=None):
    """creates a directory to save the resulting curves and metrics.
    @param alpha the learning rate for SARSA algorithms
    @param epoch total number of episodes
    @param epsilon strategy as string

    @return the relative path as string
    """
    if epsilon_strategy:
        OUTPUT_DIR = f"./outputs/group4_SARSA_L_alpha_{alpha}_epoch_{epoch}_{epsilon_strategy}"
    else:
        OUTPUT_DIR = f"./outputs/group4_SARSA_0_alpha_{alpha}_epoch_{epoch}"
    try:
        os.makedirs(OUTPUT_DIR)
    except:
        pass
    return OUTPUT_DIR
