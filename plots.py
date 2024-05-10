import matplotlib.pyplot as plt
import base64, io
import numpy as np
import pandas as pd
import seaborn as sns


# label_type = 'DQN hidden layers'

# # Prepare data for plotting
# def prepare_data(rewards_over_seeds, label_value):
#     rewards_to_plot = [[reward[0] for reward in rewards] for rewards in rewards_over_seeds]
#     df = pd.DataFrame(rewards_to_plot)
#     # Melt the dataframe and add method label
#     df_melted = df.melt(var_name='episodes', value_name='reward')
#     df_melted[label_type] = label_value
#     return df_melted

#     # Calculate mean across columns (i.e., average over all seeds for each episode)
#     # rewards_to_plot = [[reward[0] for reward in rewards] for rewards in rewards_over_seeds]
#     # df = pd.DataFrame(rewards_to_plot)
#     # df_mean = df.mean(axis=0).reset_index()  # Calculate mean and reset index for plotting
#     # df_mean.columns = ['episodes', 'reward']  # Rename columns to be suitable for seaborn
#     # df_mean['DQN hidden layers'] = label_value  # Add label for distinguishing in the plot
#     # return df_mean

def prepare_data(rewards_over_seeds = [], label_key = "label_key", label_value = "label_value", bin_size=1):
    # Calculate mean across columns (i.e., average over all seeds for each episode)
    rewards_to_plot = [[reward[0] for reward in rewards] for rewards in rewards_over_seeds]
    df = pd.DataFrame(rewards_to_plot)
    df_mean = df.mean(axis=0).reset_index()  # Calculate mean and reset index for plotting
    df_mean.columns = ['episodes', 'reward']  # Rename columns to be suitable for seaborn
    
    # Group episodes into bins and calculate mean for each bin
    df_mean['episodes'] = np.floor(df_mean['episodes'] / bin_size) * bin_size
    df_binned = df_mean.groupby('episodes').mean().reset_index()
    
    df_binned[label_key] = label_value  # Add label for distinguishing in the plot
    return df_binned[['episodes', 'reward', 'DQN hidden layers']]

def plot_reward_by_episode(data = [], label_key = "label_key", title = "Plot"):
    # Combine dataframes
    combined_df = pd.concat(data, ignore_index=True)
    
    # Set up the plotting environment
    plt.rcParams["figure.figsize"] = (10, 7)
    # Create a legend for the first line.
    plt.axhline(y=200, color='r', linestyle='--')
    
    sns.set(style="darkgrid", context="talk", palette="rainbow")
    
    # Create the plot
    ax = sns.lineplot(x="episodes", y="reward", hue=label_key, data=combined_df, legend="auto")
    ax.set(
        title=title,
        xlabel="Episodes",
        ylabel="Average Reward"
    )


    plt.gcf().set_facecolor('none') 
    plt.xlim(-10, 500)
    plt.ylim(-250, 250)
    plt.savefig('plot.pdf')
    plt.show()

