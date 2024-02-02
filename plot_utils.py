import pandas as pd 
import numpy as np

# Visualization libraries
import matplotlib.pyplot as plt
import seaborn as sns
from matplotlib.ticker import FuncFormatter

def plot_dist_instances(count_instances_tr,count_instances_val,count_inst_list=None,splits=["Train","Validation","Test"],test=False):
    """
    This function plots side by side the histogram and cumulative distribution of the instances 
    of dialogues in the dataset
    """

    plt.figure(figsize=(15, 6),dpi=350)
    plt.subplot(1, 2, 1) # Subplot for histogram of the instances
    
    # for i in len(range(count_inst_list)):
    #     max_range = max(count_inst_list[i])
    #     sns.histplot(data=count_inst_list[i], binwidth=binwidth, stat='percent',edgecolor='white',
    #             binrange=(min(count_inst_list[i])-binwidth/2, max(count_inst_list[i])+binwidth/2),label=splits[i])
        
    #     if max(count_inst_list[i]) > max_range:
    #         max_range = count_inst_list[i]
    #     if test == False and i == 1:
    #         break

    sns.histplot(data=[count_instances_tr,count_instances_val], stat='percent',edgecolor='white',discrete=True,multiple="dodge")
    # sns.histplot(data=count_instances_val, stat='percent',edgecolor='white',discrete=True,label='Val',multiple="dodge")

    # sns.histplot(data=count_instances_val, binwidth=1, stat='percent',edgecolor='white', discrete=True,
                #  binrange=(min(count_instances_val)-binwidth/2, max(count_instances_val)+binwidth/2),label='Validation')


    plt.title('Histogram of instances of the same dialogue')
    plt.xlabel('Number of instances')
    # plt.xticks(np.arange(2,max(max(count_instances_tr),max(count_instances_val))+1,1))
    plt.xticks(np.arange(2,17+1,1))

    formatter = FuncFormatter(lambda y, _: f'{int(y)}%') # add percentage sign next to y ticks
    plt.gca().yaxis.set_major_formatter(formatter)
    plt.yticks(np.arange(0, 22, 2))
    plt.legend(['Train','Validation'])

    # Subplot for the cumulative distribution of the instances
    plt.subplot(1, 2, 2)
    sns.histplot(data=count_instances_tr, binwidth=1, element='step', fill=False, cumulative=True, stat='density',label='Train')
    sns.histplot(data=count_instances_val, binwidth=1, element='step', fill=False, cumulative=True, stat='density',label='Validation')

    plt.title('Cumulative distribution of instances of the same dialogue')
    plt.xlabel('Number of instances')
    plt.ylabel('')
    plt.xticks(np.arange(2,max(max(count_instances_tr),max(count_instances_val))+1,1))
    
    formatter = FuncFormatter(lambda y, _: f'{int(y*100)}%') # add percentage sign next to y ticks
    plt.gca().yaxis.set_major_formatter(formatter)
    plt.yticks(np.arange(0,1.1,0.1))
    plt.grid()
    plt.legend()
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

def plot_num_words_dial(max_dialogues: list[str]):
    """This function makes two plots side by side:
    1) an histogram for the number of words in a dialogue
    2) the cumulative plot for the number of words in a doaòpgie
    """
    # compute num words per dialogues
    num_word_diag = [len(x.split()) for x in max_dialogues]
    plt.figure(figsize=(17, 6),dpi=350)
    plt.subplot(1, 2, 1)

    # Plot histogram
    sns.histplot(data=num_word_diag, binrange=[10,270], binwidth=20, stat='percent',edgecolor='white')
    plt.xticks(np.arange(0, 261, 20))
    plt.yticks(np.arange(0, 24, 2))
    # Set percentage sign for yticks
    formatter = FuncFormatter(lambda y, _: f'{int(y)}%')
    plt.gca().yaxis.set_major_formatter(formatter)

    plt.title("Number of words per dialogue")
    plt.xlabel('Number of words in the dialogue')

    # Plot cumulative plot
    plt.subplot(1, 2, 2)
    sns.histplot(data=num_word_diag, binrange=[0,260], binwidth=15, element='step', fill=False, cumulative=True, stat='density')
    plt.xticks(np.arange(0, 261, 15))
    plt.yticks(np.arange(0,1.1,0.1))

    formatter = FuncFormatter(lambda y, _: f'{int(y*100)}%') # add percentage sign next to y ticks
    plt.gca().yaxis.set_major_formatter(formatter)
    plt.xlabel('Number of words in the dialogue')
    plt.ylabel('')
    plt.title("Cumulative distribution of words per dialogue")
    plt.grid() 
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