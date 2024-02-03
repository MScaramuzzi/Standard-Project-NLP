import pandas as pd 
import numpy as np

# Visualization libraries
import matplotlib.pyplot as plt
import seaborn as sns
from matplotlib.ticker import FuncFormatter

def plot_instances_distribution(num_instances_tr: list[int], df_max_tr: pd.DataFrame,
                                num_instances_val: list[int], df_max_val: pd.DataFrame,
                                color_train: str, color_val: str, bar_width: int):
    
    df_max_tr = df_max_tr.copy()
    df_max_tr['perc_instances'] = num_instances_tr # add column with instance for each dialogue
    # compute how many times a dialogue is repeated in the dataset as a percentage
    df_grp_tr = df_max_tr['perc_instances'].value_counts(normalize=True).mul(100).to_frame().reset_index(names=['instances'])

    df_max_val = df_max_val.copy()
    df_max_val['perc_instances'] = num_instances_val
    df_grp_val = df_max_val['perc_instances'].value_counts(normalize=True).mul(100).to_frame().reset_index(names=['instances'])

    fig, axes = plt.subplots(figsize=(15, 6),dpi=300, nrows=1, ncols=2)
    plt.tight_layout()

    # I: Barplot of distribution of repetitions for training and validation set
    sns.barplot(data=df_grp_tr, x='instances', y='perc_instances',color=color_train,
                width=bar_width, label='Train',ax=axes[0])
    sns.barplot(data=df_grp_val, x='instances', y='perc_instances', color=color_val,
                width=bar_width, label='Validation',ax=axes[0])

    # Manually adjust the position of bars of the train set barplot
    for bar in axes[0].patches[len(df_grp_tr):]:
        bar.set_x(bar.get_x() + 0.3)

    x_pos = np.arange(bar_width/2,17 + bar_width/2 ,1)
    axes[0].set_xticks(x_pos)
    axes[0].set_yticks(np.arange(0,19,2))
    formatter = FuncFormatter(lambda y, _: f'{int(y)}%') # add percentage sign next to y ticks
    axes[0].yaxis.set_major_formatter(formatter)

    axes[0].set_title('Histogram of instances of the same dialogue', fontsize=16)
    axes[0].set_xlabel('')
    axes[0].set_ylabel('Percentages', fontsize=13)

    # II: Cumulative plot of number of instances of each dialogue
    sns.histplot(data=num_instances_tr, binwidth=1, element='step',
                fill=False, cumulative=True, stat='density',
                label='Train', color=color_train, ax=axes[1], lw=1.75)

    sns.histplot(data=num_instances_val, binwidth=1, element='step',
                fill=False, cumulative=True, stat='density',
                label='Validation',color=color_val,ax=axes[1], lw=1.75)

    axes[1].set_xticks(np.arange(2,17,1))
    # axes[1].set_yticks(np.arange(0,1.1,0.1))

    formatter = FuncFormatter(lambda y, _: f'{int(y*100)}%') # add percentage sign next to y ticks
    axes[1].yaxis.set_major_formatter(formatter)
    axes[1].set_ylabel('')
    axes[1].set_title('Cumulative distribution of instances of the same dialogue',fontsize=16)
    axes[1].legend()

    fig.text(0.5, -0.04, 'Number of instances', ha='center', size=14) # set common x axis label for the two subplots
    plt.show()


# Plot utterance statistics
def plot_num_utterances_dialogue(df: pd.DataFrame):
    """This function makes two plots side by side:
    1) an histogram for the number of utterances in a dialogue
    2) the boxplot for the number of utterances in a dialogue
    """
    df['nb_utterences'] = df['utterances'].apply(len) # Calculate number of utterances in the dialogue and create a new column
    
    plt.figure(figsize=(16, 7),dpi=250)
    plt.subplot(1, 2, 1) # Plot the histogram of utterance lengths on the left
    sns.histplot(data=df['nb_utterences'],binwidth=1, stat='percent',edgecolor='white')
    plt.title('Number of utterances per dialogue',fontsize=14)
    plt.xlabel('Utterance Length (number of utterances)',fontsize=12)
    plt.ylabel('Percent',fontsize=12)

    plt.xticks(np.arange(0, 26, 1),fontsize=12)  
    plt.yticks(np.arange(0, 11, 1),fontsize=12)
    formatter = FuncFormatter(lambda y, _: f'{int(y)}%') # add percentage sign next to y ticks
    plt.gca().yaxis.set_major_formatter(formatter)

    # Create the boxplot on the right
    plt.subplot(1, 2, 2)
    ax = sns.boxplot(x='nb_utterences', data=df)

    # Compute quartiles
    q1 = df['nb_utterences'].quantile(0.25)
    median = df['nb_utterences'].median()
    q3 = df['nb_utterences'].quantile(0.75)

    # Highlight quartiles with vertical lines
    ax.axvline(x=q1, color='g', linestyle='--', label=f'Q1: {q1:.2f}')
    ax.axvline(x=median, color='b', linestyle='--', label=f'Median: {median:.2f}')
    ax.axvline(x=q3, color='r', linestyle='--', label=f'Q3: {q3:.2f}')

    plt.title('Boxplot for number of utterances per dialogue',fontsize=14)
    plt.xlabel('Number of utterances per dialogue',fontsize=12)

    # Show quartile values on the x-axis
    ax.set_xticks([q1, median, q3])
    ax.set_xticklabels([f'{q1:.2f}', f'{median:.2f}', f'{q3:.2f}'], fontsize=12)
    plt.tight_layout()
    plt.legend(fontsize=12)
    plt.show()

def plot_num_words_utterance(lengths_array: np.array):
    """This function makes two plots side by side:
    1) an histogram for the number of words in a utterance
    2) the cumulative plot for the number of words in a utterance
    """
    plt.figure(figsize=(15, 6), dpi=350)
    plt.subplot(1, 2, 1)  # Create subplot for histogram of word counts
    sns.histplot(lengths_array, binwidth=4,binrange=[0,lengths_array.max()],stat='percent',edgecolor='white')
    
    plt.xticks(np.arange(0,lengths_array.max()+1,4))
    plt.yticks(np.arange(0, 50, 5))
    formatter = FuncFormatter(lambda y, _: f'{int(y)}%') # add percentage sign next to y ticks
    plt.gca().yaxis.set_major_formatter(formatter)
    plt.title('Histogram of Word Counts per Utterance')
    plt.xlabel('Number of Words')
    
    plt.subplot(1, 2, 2) # Plot cumulative plot 
    sns.histplot(data=lengths_array, binrange=[0,48], binwidth=2, element='step', fill=False, cumulative=True, stat='density')
    plt.xticks(np.arange(0, 49, 4))
    plt.yticks(np.arange(0,1.1,0.1))
    formatter = FuncFormatter(lambda y, _: f'{int(y*100)}%') # add percentage sign next to y ticks
    plt.gca().yaxis.set_major_formatter(formatter)

    plt.title("Cumulative distribution of words per utterance")
    plt.xlabel('Number of words ')
    plt.ylabel('')
    plt.grid() 
    plt.show()

# compute num words per dialogues
def plot_num_words_dial(max_dialogues_tr: list[str], max_dialogues_val: list[str],
                        color_train: str, color_val: str):

    num_word_diag_tr  = [len(x.split()) for x in max_dialogues_tr] # generate list with word lengths for dialogues in train set
    num_word_diag_val = [len(x.split()) for x in max_dialogues_val] # generate list with word lengths for dialogues in val set
    fig, axes = plt.subplots(figsize=(24,8),dpi=300,nrows=1,ncols=3)

    # plotting variables
    STEP = 20
    max_tr = max(num_word_diag_tr)
    bins_tr = np.arange(5, max(num_word_diag_tr) + STEP, STEP)

    min_val = min(num_word_diag_val)
    max_val = max(num_word_diag_val)
    bins_val = np.arange(10, max_val+STEP, STEP)

    # I. Plot histogram train set
    sns.histplot(data=num_word_diag_tr, stat='percent',
                color=color_train, edgecolor='white', bins=bins_tr,
                ax=axes[0], alpha=0.85, label='Train')

    axes[0].set_xticks(bins_tr)
    axes[0].set_yticks(np.arange(0, 24, 2))
    formatter = FuncFormatter(lambda y, _: f'{int(y)}%') # Set percentage sign for yticks

    # II. Plot histogram validation set
    sns.histplot(data=num_word_diag_val, stat='percent',
                color=color_val, edgecolor='white', bins=bins_val,
                ax=axes[1], alpha=0.85, label='Val')

    axes[1].set_yticks(np.arange(0, 22, 2))
    axes[1].set_xticks(bins_val)
    axes[1].set_ylabel('')

    # III. Plot cumulative for train and val set
    sns.histplot(data=num_word_diag_tr, bins=bins_tr,
                lw=2.5, color=color_train, element='step', fill=False,
                ax=axes[2], cumulative=True, stat='density',label='Train')

    sns.histplot(data=num_word_diag_val, bins=bins_val,
                lw=2.5, color=color_val, element='step', fill=False,
                ax=axes[2], cumulative=True, stat='density',label='Val')


    bins = np.arange(5, max(num_word_diag_tr) + STEP, STEP)
    axes[2].set_xticks(bins)
    axes[2].set_ylabel('')
    axes[2].set_title("Cumulative distribution of words per dialogue")
    axes[2].set_yticks(np.arange(0,1.1,0.1))

    #### Common plotting sections

    # Calculate the position for the title for first two plots
    x_suptitle = 0.5 * (axes[0].get_position().x1 + axes[1].get_position().x0)
    y_suptitle = 1.015  # Adjust the y-coordinate to avoid overlapping text

    # Add a placeholder title between the first two histograms
    fig.text(x_suptitle, y_suptitle, 'Histogram of number of words per dialogue in train and val set',
            ha='center', size=22)

    # Add percentage signs to each of the subplots
    for ax in axes:
        ax.tick_params(axis='both', labelsize=16)
        ax.legend(fontsize=18)
        if ax == axes[2]:
            formatter = FuncFormatter(lambda y, _: f'{int(y*100)}%') # add percentage sign next to y ticks for third plot
            ax.yaxis.set_major_formatter(formatter)
        else:
            ax.yaxis.set_major_formatter(formatter)

    plt.tight_layout()
    fig.text(0.5, -0.04, 'Number of words per dialogue', ha='center', size=18) # add common label for x axis for the three plots

    plt.show()

def plot_num_triggers_diag(trigger_array: np.array):
    """
    This function plots the histogram of number of triggers per dialogue
    """
    # Count the number of ones in each subarray
    ones_count_per_subarray = trigger_array.apply(lambda x: sum(1 for i in x if i == 1.0))
    plt.figure(figsize=(8,6),dpi=105)
    # Create a histogram with relative percentages
    sns.histplot(data=ones_count_per_subarray, binwidth=1, stat='percent',binrange=(-0.5, max(ones_count_per_subarray) + 0.5))

    # Set axis label and ticks
    plt.xlabel('Number of Ones')
    plt.ylabel('Percentage')
    plt.xticks(np.arange(0,10,1))
    plt.yticks(np.arange(0,80,10))

    # Set labels and title
    plt.title('Histogram of number of triggers per dialogue')
    formatter = FuncFormatter(lambda y, _: f'{int(y)}%') # add percentage sign next to y ticks
    plt.gca().yaxis.set_major_formatter(formatter)
    plt.show()

def plot_trigger_distance(trigger_array: np.array):
    """
    This function plots the distribution of the distances between the trigger and the target utterance
    """
    # Use apply and lambda to find one positions (i.e. find where are the trigger utterances) in each array
    one_positions = trigger_array.apply(lambda x: np.where(x == 1)[0])

    # Define a lambda function to calculate the distance between each element and the last element
    calculate_distances = lambda indices, last_index: [last_index - idx for idx in indices]

    # Use list comprehension and lambda to create a list of sublists consisting of distances
    last_indices = [len(arr) - 1 for arr in trigger_array]
    distances_list = [calculate_distances(indices, last_index) for indices, last_index in zip(one_positions, last_indices)]
    flat_dist = np.concatenate(distances_list) # flatten distance list to have all the distances in a single array
    plt.figure(figsize=(16,6),dpi=350)
    plt.subplot(1, 2, 1)
    sns.histplot(data=flat_dist, binwidth=1, stat='percent')

    plt.xticks(np.arange(0,17,1))
    plt.yticks(np.arange(0,55,5))
    formatter = FuncFormatter(lambda y, _: f'{int(y)}%') # add percentage sign next to y ticks
    plt.gca().yaxis.set_major_formatter(formatter)

    plt.title('Histogram of distance between trigger and target utterance')
    plt.xlabel('Distance from target utterance')
    plt.ylabel('Percent')

    plt.subplot(1, 2, 2)
    sns.histplot(data=flat_dist, binwidth=1, element='step', fill=False, cumulative=True, stat='density')
    plt.ylabel('')
    plt.xlabel('Distance from target utterance')

    plt.yticks(np.arange(0,1.1,0.1))
    plt.xticks(np.arange(0,17,1))
    formatter = FuncFormatter(lambda y, _: f'{int(y*100)}%') # add percentage sign next to y ticks
    plt.gca().yaxis.set_major_formatter(formatter)

    plt.grid()
    plt.title('Cumulative distribution of distance between trigger and target utterance')
    plt.show()