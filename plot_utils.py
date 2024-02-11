import pandas as pd 
import numpy as np

# Visualization libraries
import matplotlib.pyplot as plt
import seaborn as sns
from matplotlib.ticker import FuncFormatter

# 2.1 Dialogues with multiple instances
def plot_instances_distribution(num_instances_tr: list[int], df_max_tr: pd.DataFrame,
                                num_instances_val: list[int], df_max_val: pd.DataFrame,
                                color_train: str, color_val: str, bar_width: int):
    """This function plots the number of instances of each dialogue, it generates two plots side by side:
    1) Bar chart of the percentage of instances for each dialogue in train and validation set, plotted adjacent to each other.  
    2) Cumulative plot of the distribution of instances for each dialogue in train and validation set, plotted jointly.
    """
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
    axes[0].set_ylabel('Percent', fontsize=13)

    # II: Cumulative plot of number of instances of each dialogue
    sns.histplot(data=num_instances_tr, binwidth=1, element='step',
                fill=False, cumulative=True, stat='density',
                label='Train', color=color_train, ax=axes[1], lw=1.75)

    sns.histplot(data=num_instances_val, binwidth=1, element='step',
                fill=False, cumulative=True, stat='density',
                label='Validation',color=color_val,ax=axes[1], lw=1.75)

    axes[1].set_xticks(np.arange(2,17,1))
    axes[1].set_yticks(np.arange(0,1.1,0.1))

    formatter = FuncFormatter(lambda y, _: f'{int(y*100)}%') # add percentage sign next to y ticks
    axes[1].yaxis.set_major_formatter(formatter)
    axes[1].set_ylabel('')
    axes[1].set_title('Cumulative distribution of instances of the same dialogue',fontsize=16)
    axes[1].legend()

    fig.text(0.5, -0.04, 'Number of instances', ha='center', size=14) # set common x axis label for the two subplots
    plt.show()

####### 2.2 Dialogues composition #######

# 2.2.1 Distribution of words per dialogue
def plot_num_words_dial(max_dialogues_tr: list[str], max_dialogues_val: list[str],
                        color_train: str, color_val: str):
    # compute num words per dialogues

    num_word_diag_tr  = [len(x.split()) for x in max_dialogues_tr] # generate list with word lengths for dialogues in train set
    num_word_diag_val = [len(x.split()) for x in max_dialogues_val] # generate list with word lengths for dialogues in val set
    fig, axes = plt.subplots(figsize=(24,8),dpi=300,nrows=1,ncols=3)

    # plotting variables
    STEP = 20
    max_tr = max(num_word_diag_tr)
    bins_tr = np.arange(5, max_tr + STEP, STEP)

    max_val = max(num_word_diag_val)
    bins_val = np.arange(10, max_val + STEP, STEP)

    # I. Plot histogram train set
    sns.histplot(data=num_word_diag_tr, stat='percent',
                color=color_train, edgecolor='white', bins=bins_tr,
                ax=axes[0], alpha=0.85, label='Train')
    
    # Set histograms ticks (and grid) to match the histogram rectangles  
    axes[0].set_xticks(bins_tr)
    axes[0].set_yticks(np.arange(0, 24, 2))
    axes[0].set_ylabel('Percent', fontsize=16, labelpad=20)
    formatter = FuncFormatter(lambda y, _: f'{int(y)}%') # Set percentage sign for yticks

    # II. Plot histogram validation set
    sns.histplot(data=num_word_diag_val, stat='percent',
                color=color_val, edgecolor='white', bins=bins_val,
                ax=axes[1], alpha=0.85, label='Val')

    axes[1].set_yticks(np.arange(0, 24, 2))
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

    #### Common plotting section ####

    # Calculate the position for the title for first two plots
    x_suptitle = 0.5 * (axes[0].get_position().x1 + axes[1].get_position().x0)
    y_suptitle = 1.0005  # Adjust the y-coordinate to avoid overlapping text

    # Add a title between the first two histograms
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

# 2.2.2 Distribution of utterances in dialogues
def plot_num_utterances_dial(df: pd.DataFrame, median_color: str):
    """This function makes two plots side by side:
    1) an histogram for the number of utterances in a dialogue
    2) the boxplot for the number of utterances in a dialogue
    """
    plt.style.use('default')

    df['nb_utterences'] = df['utterances'].apply(len) # Calculate number of utterances in the dialogue and create a new column
    fig, axes = plt.subplots(figsize=(16, 8),dpi=300, nrows=1, ncols=2)
    
    # I. Plot the histogram of utterance lengths for the whole dataset on the first subplot
    sns.histplot(data=df['nb_utterences'],binwidth=1, stat='percent',
                 ax=axes[0], edgecolor='white')

    axes[0].set_title('Number of utterances per dialogue',fontsize=14)
    axes[0].set_xlabel('Utterance Length (number of utterances)',fontsize=12)
    axes[0].set_ylabel('Percent',fontsize=12)

    axes[0].set_xticks(np.arange(0, 26, 1))
    axes[0].set_yticks(np.arange(0, 11, 1))
    formatter = FuncFormatter(lambda y, _: f'{int(y)}%') # add percentage sign next to y ticks
    axes[0].yaxis.set_major_formatter(formatter)

    axes[0].grid(axis='y')

    # II. Plot boxplot on the second subplot
    sns.boxplot(y='nb_utterences', data=df, fill=False, 
                width=0.5, ax=axes[1], linewidth=2,
                whis=1.2, medianprops=dict(linewidth=3.5, color=median_color))

    # Compute quartiles
    q1 = df['nb_utterences'].quantile(0.25)
    median = df['nb_utterences'].median() 
    q3 = df['nb_utterences'].quantile(0.75)

    # Highlight quartiles with vertical lines
    axes[1].axhline(y=q1, color='g', linestyle='--', label=f'Q1: {q1:.2f}',xmin=0, xmax=0.25)
    axes[1].axhline(y=median, color='b', linestyle='--', label=f'Median: {median:.2f}',xmin=0, xmax=0.25)
    axes[1].axhline(y=q3, color='firebrick', linestyle='--', label=f'Q3: {q3:.2f}', xmin=0, xmax=0.25)


    axes[1].set_title('Boxplot for number of utterances per dialogue',fontsize=14)
    axes[1].set_ylabel('Number of utterances per dialogue',fontsize=12)

    # Show quartile values on the x-axis
    for ax in axes:
        ax.tick_params(axis='both', which='major') 

    axes[1].set_yticks([q1, median, q3])
    axes[1].set_yticklabels([f'{q1:.2f}', f'{median:.2f}', f'{q3:.2f}'])
    axes[1].legend(fontsize=12)

    plt.tight_layout()
    plt.show()


# 2.2.3 Distribution of words per utterance
def plot_num_words_utterance(lengths_array_tr: np.array, lengths_array_val: np.array,
                             color_train: str, color_val: str):
    """This function makes two plots side by side:
    1) an histogram for the number of words in a utterance
    2) the cumulative plot for the number of words in a utterance
    """
    fig, axes = plt.subplots(figsize=(24, 8), dpi=350, nrows=1,ncols=3)

    # plotting variables 
    STEP = 5
    max_tr = max(lengths_array_tr)
    bins_tr = np.arange(0, max_tr + STEP, STEP)

    max_val = max(lengths_array_val)
    bins_val = np.arange(0, max_val + STEP, STEP)
    formatter = FuncFormatter(lambda y, _: f'{int(y)}%') # add percentage sign next to y ticks

    # I. Plot histogram train set
    sns.histplot(lengths_array_tr, bins=bins_tr,
                stat='percent',edgecolor='white', label='Train',
                ax=axes[0], alpha=0.85, color=color_train)
    
    # Set histograms ticks (and grid) to match the histogram rectangles  
    axes[0].set_xticks(bins_tr)
    axes[0].set_ylabel('Percent', fontsize=16, labelpad=20)

    # II. Plot histogram validation set
    sns.histplot(lengths_array_val, bins=bins_val,
                stat='percent',edgecolor='white', label='Validation',
                ax=axes[1], alpha=0.85, color=color_val)
    
    axes[1].set_xticks(bins_val)
    axes[1].set_yticks(np.arange(0,40,5))
    axes[1].set_ylabel('')

    # Calculate the position for the title for first two plots
    x_suptitle = 0.5 * (axes[0].get_position().x1 + axes[1].get_position().x0)
    y_suptitle = 1.0005  # Adjust the y-coordinate to avoid overlapping text

    # Add a title between the first two histograms
    fig.text(x_suptitle, y_suptitle, 'Histogram of number of words per utterance in train and val set',
            ha='center', size=22)

    # III. Plot cumulative plot

    sns.histplot(data=lengths_array_tr, bins=bins_tr, lw=2.5,
                color=color_train, element='step', fill=False, 
                ax=axes[2], cumulative=True, stat='density', label='Train')

    sns.histplot(data=lengths_array_val, bins=bins_val, lw=2.5,
                color=color_val, element='step', fill=False, 
                ax=axes[2], cumulative=True, stat='density', label='Val')

    axes[2].set_xticks(bins_tr)
    axes[2].set_title("Cumulative distribution of words per utterance")

    # Add percentage signs to each of the subplots
    for ax in axes:
        ax.tick_params(axis='both', labelsize=16)
        ax.legend(fontsize=16)
        if ax == axes[2]:
            formatter = FuncFormatter(lambda y, _: f'{int(y*100)}%') # add percentage sign next to y ticks for third plot
            ax.yaxis.set_major_formatter(formatter)
            ax.set_ylabel('')
        else:
            ax.yaxis.set_major_formatter(formatter)

    fig.text(0.5, -0.04, 'Number of words', ha='center', size=18) # add common label for x axis for the three plots
    fig.tight_layout()
    plt.show()

#### 2.4 Distribution of triggers across the dataset ####

def plot_num_triggers_diag(trigger_array: np.array):
    """
    This function plots the histogram of number of triggers per dialogue
    """
    # Count the number of ones in each subarray
    ones_count_per_subarray = trigger_array.apply(lambda x: sum(1 for i in x if i == 1.0))
    plt.figure(figsize=(8,6),dpi=105)
    # Create a histogram with relative percentages
    sns.histplot(data=ones_count_per_subarray, binwidth=1, 
                 stat='percent',binrange=(-0.5, max(ones_count_per_subarray) + 0.5))

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


#### 2.3 Distribution of emotions across the dataset ####

# 2.3.1 TBC

# 2.3.2 TBC


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


from matplotlib.ticker import FuncFormatter


def plot_emotion_distr(df_tr,df_val,df_test, model):

    # sns.set_context('paper', font_scale=1.075)

    fig, axes = plt.subplots(figsize=(13.5, 6),dpi=350, nrows=1, ncols=3)

    emotions_tr = df_tr['emotions'] 
    emotions_arr_tr =  np.concatenate(np.array(emotions_tr))

    emotions_val = df_val['emotions'] 
    emotions_arr_val =  np.concatenate(np.array(emotions_val))

    emotions_test = df_test['emotions'] 
    emotions_arr_test =  np.concatenate(np.array(emotions_test))

    df_em_perc_tr = pd.DataFrame(pd.Series(emotions_arr_tr)
                                       .value_counts(normalize=True),columns=['Percent'])\
                                       .reset_index().rename(columns={"index":"Emotion"}) # get the count for each frequency
    df_em_perc_tr['Percent'] = df_em_perc_tr['Percent']*100

    df_em_perc_val =  pd.DataFrame(pd.Series(emotions_arr_val)
                                       .value_counts(normalize=True),columns=['Percent'])\
                                       .reset_index().rename(columns={"index":"Emotion"}) # get the count for each frequency
    df_em_perc_val['Percent'] = df_em_perc_val['Percent']*100


    df_em_perc_test =  pd.DataFrame(pd.Series(emotions_arr_test)
                                       .value_counts(normalize=True),columns=['Percent'])\
                                       .reset_index().rename(columns={"index":"Emotion"}) # get the count for each frequency
    df_em_perc_test['Percent'] = df_em_perc_test['Percent']*100


    # Extract unique emotion labels from the training DataFrame
    unique_emotions = df_em_perc_tr['Emotion'].unique()
    train_palette = sns.color_palette("Set2", len(unique_emotions))
    color_dict = dict(zip(unique_emotions, train_palette))

    for i, (df_em_perc, title) in enumerate(zip([df_em_perc_tr, df_em_perc_val, df_em_perc_test], ['Train', 'Val', 'Test'])):
        if i == 0:
            sns.barplot(data=df_em_perc, x='Emotion', y='Percent', palette=color_dict, legend = True,
                        edgecolor='white', hue='Emotion', ax=axes[i], order=unique_emotions)
        else:
            sns.barplot(data=df_em_perc, x='Emotion', y='Percent', palette=color_dict,
                        hue='Emotion', legend=False, edgecolor='white',ax=axes[i], order=unique_emotions)

        formatter = FuncFormatter(lambda y, _: f'{int(y)}%')  # add percentage sign next to y ticks
        axes[i].tick_params(axis='x', labelsize='small')
        axes[i].yaxis.set_major_formatter(formatter)
        axes[i].set_yticks(np.arange(0, 50, 5))
        axes[i].set_title(title)
        if i !=0:
            axes[i].set_ylabel('')

    axes[0].legend(loc=1,ncols=2,fontsize=10)
    fig.suptitle(f"Distribution of emotion labels for {model}", fontsize=14)
    plt.show()
