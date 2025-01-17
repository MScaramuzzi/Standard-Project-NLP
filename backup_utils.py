# Data libraries
import numpy as np
import torch
import random
import pandas as pd 
import json
import os

# Visualization libraries
import matplotlib.pyplot as plt
import seaborn as sns
from matplotlib.ticker import FuncFormatter


# setting the seed
def ensure_reproducibility(seed) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    os.environ["PYTHONHASHSEED"] = str(seed)
    torch.cuda.manual_seed(seed)


def add_emotion_idx(df:pd.DataFrame) -> pd.DataFrame:
    # Generate a list of all emotions
    df_exploded = df.explode('emotions')

    # Get the unique values for emotions
    unique_emotions = df_exploded['emotions'].unique()
    # Generate a mapping from emotion to numerical index
    emotions_mapping = {emotion: idx for idx,emotion in enumerate(unique_emotions)}
    
    # Choose the position where you want to insert the new column
    position = df.shape[1] -2 # put our emotions_num next to the emotion (i.e. position = 6 - 2 = 4)
    emotions_num = df['emotions'].apply(lambda x: [emotions_mapping[emotion] for emotion in x])

    # Insert the new column at the specified position
    df.insert(position, 'emotions_num', emotions_num)

    return df

# load the data
def load_data(json_path: str) -> pd.DataFrame:
    # load the json file as a list of dictionaries
    with open(json_path, 'r') as j:
        json_data = json.loads(j.read())

    # normalize the json data to structure it into table format
    df_json = pd.json_normalize(json_data) 
    df_json.rename(columns={'episode':'dialogue_id'},inplace=True) # rename episode to dialogue_id
    return df_json

def get_repeated_instances(df: pd.DataFrame)-> tuple[list[str], list[int], list[int]]:
    """
    This function counts how many instances of the same dialogue are present in the dataframe
    and also it retrieves the biggest dialogue among each set of the instances of the same dialogue.

    This function returns:
    - diags : the list that will contain the biggest dialogues among the set ofinstances
    - count_list: the number of instances of the same dialogue that are in the dataset
    - indices: indices in the original dataframe to find the max dialogues
    """

    dialogues = df['utterances'] # get list of dialogues
    count = 0 # initialize count of instances to select
    max_diag = '' # string that will store the biggest dialogue among the group of cumulative dialogues
    count_list, diags = [] , [] # lists to store the count of instances and the dialogues
    indices = [] # store indices for retrieving max dialogue

    # loop through all the dialogues
    for idx, utt in enumerate(dialogues):
        if count == 0: # first dialogue 
            first_diag = ''.join(str(ele) for ele in utt) # create a string with the first element in the group of instances of dialogues
        curr_diag = ''.join(str(ele) for ele in utt) # convert the current dialogue to a string containing all the utterances

        # check if the current dialogue we are considering contains the first dialogue of the group
        if first_diag in curr_diag:
            count += 1 # increase the count and continue searching for dialogues in our list of dialogues
            max_diag = curr_diag # set the current dialogue as max dialogue (max dialogue is always the last in the set of instances)
        # if conditions not met --> we are in another set of dialogues, append max_diag and idx for storing
        else:        
            diags.append(max_diag)
            indices.append(idx)

            if len(diags) == 1:  # fix the edge case of the first dialogue
                count_list.append(count) # store this count for plotting later
            else:
                count_list.append(count+1) # increase count+1 to avoid miscounts
            count = 0 # reset the count

        if idx == len(dialogues)-1: # solve the edge case of last max dialogue
            diags.append(''.join(str(ele) for ele in utt))
            count_list.append(count)
            indices.append(idx)
    np_indices = np.array(indices) - 1 # avoid mispositioning of indexes       
    
    return diags,count_list,np_indices

# Function to compute number of words of each utterance in a dialogue
def get_lengths(utterances):
    return np.array([len(utterance.split()) for utterance in utterances])

def get_utterance_lenghts(df: pd.DataFrame)-> np.array:
    """"
    This function computes the utterance length for all the entries in the dataframe
    and then it flattens the array that stores them
    """

    df_copy = df.copy() # make copy to avoid issues pertaining to aliasing
    df_copy.loc[:,'lengths_array'] = df['utterances'].apply(get_lengths) # Apply the function to the 'utterance' column
    lengths_array = np.concatenate(df_copy['lengths_array'].values) # get the flattened array with all the utterance lengths
    return lengths_array

def suggestiveText(df: pd.DataFrame, task: str = 'ERC'):
    if task not in ['ERC', 'EFR']:
        print('Task not accepted. Please choose between ERC and EFR.')
        return

    df = df.copy()
    speakers = [s for s in df['speakers']]
    utterances = [u for u in df['utterances']]
    suggestive_texts = []

    if task == 'ERC':
        for spkrs,utts in zip(speakers, utterances):
            queries_list = []
            for focus in range(len(utts)):
                sugg = f''
                for i,(s,u) in enumerate(zip(spkrs,utts)):
                    if i == focus:
                        sugg += f'<s>{s} <mask> says: {u}</s> '
                    else:
                        sugg += f'{s} says: {u} '

                queries_list.append(sugg)
            suggestive_texts.append(queries_list)
    else:
        emotions = [e for e in df['emotions']]

        emotions_adverbs = {
            "neutral": '',
            "anger": 'angrily ',
            "disgust": 'disgustingly ',
            "fear": 'fearfully ',
            "joy": 'joyfully ',
            "surprise": 'surprisingly ',
            "sadness": 'sadly '
        }

        for spkrs, emos, utts in zip(speakers, emotions, utterances):
            sugg = f''
            for i,(s,e,u) in enumerate(zip(spkrs, emos, utts)):
                if i == len(utts)-1:
                    sugg += f'<s>{s} {emotions_adverbs[e]}says: {u}</s> '
                else:
                    sugg += f'{s} {emotions_adverbs[e]}says: {u} '

            suggestive_texts.append(sugg)

    df['suggestive_texts'] = suggestive_texts

    if task == 'ERC':
        return df.drop(columns=['dialogue_id','speakers','emotions','utterances','triggers'])
    else:
        return df.drop(columns=['dialogue_id','emotions_num'])


def preprocess_SuggestiveText_ERC(examples: pd.DataFrame, tokenizer, tok_max_len: int = 350):
    input_ids = torch.empty((0,tok_max_len), dtype=torch.int)
    attention_mask = torch.empty((0, tok_max_len), dtype=torch.int)

    for suggestive_texts in examples['suggestive_texts']:
        for sugg in suggestive_texts:
            tokens = tokenizer.tokenize(sugg)
            ids = tokenizer.convert_tokens_to_ids(tokens)

            # Truncate to max length
            if len(ids) > tok_max_len:
                ids = ids[:tok_max_len]

            # Pad to max length
            if len(ids) < tok_max_len:
                ids = ids + [tokenizer.pad_token_id] * (tok_max_len - len(ids))

            # Create attention mask
            attention = [1 if token_id != tokenizer.pad_token_id else 0 for token_id in ids]

            input_ids = torch.cat((input_ids, torch.tensor(ids).unsqueeze(0)), dim = 0)
            attention_mask = torch.cat((attention_mask, torch.tensor(attention).unsqueeze(0)), dim=0)

        labels = torch.tensor(np.concatenate(examples['emotions_num']))

    return {
        "input_ids": input_ids,
        "attention_mask": attention_mask,
        "labels": labels
    }

# FIXME: THIS
# train_st_efr_set = train_st_efr_dataset.map(ut.preprocess_SuggestiveText_EFR, batched=True, fn_kwargs={'tokenizer': roberta_tokenizer,
    #'pos_trig' : pos_trig}, remove_columns=train_st_efr_dataset.column_names)

def preprocess_SuggestiveText_EFR(examples: pd.DataFrame, tokenizer, 
                                  pos_trig, tok_max_len: int = 250,
                                  sequence_len: int = 50, window_size: int = 7):
    speakers_utterances_input_ids = []
    speakers_utterances_attention_mask = []
    emotions_utterances_input_ids = []
    emotions_utterances_attention_mask = []
    st_input_ids = torch.empty((0,tok_max_len), dtype=torch.int)
    st_attention_mask = torch.empty((0,tok_max_len), dtype=torch.int)

    padding_tensor = torch.tensor([tokenizer.pad_token_id]*sequence_len).unsqueeze(0)
    attention_padding = torch.tensor([0]*sequence_len).unsqueeze(0)

    for (speakers, emotions, utterances) in zip(examples['speakers'], examples['emotions'], examples['utterances']):
        spk_utt_combined = [s + ": " + u for (s,u) in zip(speakers[-window_size:],utterances[-window_size:])]
        emo_utt_combined = [e + ": " + u for (e,u) in zip(emotions[-window_size:],utterances[-window_size:])]

        spk_utt_encodings = tokenizer(spk_utt_combined, padding="max_length", truncation=True, max_length=sequence_len, return_tensors='pt')
        emo_utt_encodings = tokenizer(emo_utt_combined, padding="max_length", truncation=True, max_length=sequence_len, return_tensors='pt')

        # Padding sequences shorter than window_size
        if len(spk_utt_encodings['input_ids']) < window_size:
          for i in range(window_size - len(spk_utt_encodings['input_ids'])):
            spk_utt_encodings['input_ids'] = torch.cat((spk_utt_encodings['input_ids'], padding_tensor), dim=0)
            spk_utt_encodings['attention_mask'] = torch.cat((spk_utt_encodings['attention_mask'], attention_padding), dim=0)

            emo_utt_encodings['input_ids'] = torch.cat((emo_utt_encodings['input_ids'], padding_tensor), dim=0)
            emo_utt_encodings['attention_mask'] = torch.cat((emo_utt_encodings['attention_mask'], attention_padding), dim=0)

        speakers_utterances_input_ids.append(spk_utt_encodings['input_ids'].tolist())
        speakers_utterances_attention_mask.append(spk_utt_encodings['attention_mask'].tolist())

        emotions_utterances_input_ids.append(emo_utt_encodings['input_ids'].tolist())
        emotions_utterances_attention_mask.append(emo_utt_encodings['attention_mask'].tolist())

    for suggestive_text in examples['suggestive_texts']:
        tokens = tokenizer.tokenize(suggestive_text)
        ids = tokenizer.convert_tokens_to_ids(tokens)

        # Truncate to max length
        if len(ids) > tok_max_len:
            ids = ids[-tok_max_len:]  # Keeps the last tok_max_len tokens

        # Pad to max length
        if len(ids) < tok_max_len:
            ids = ids + [tokenizer.pad_token_id] * (tok_max_len - len(ids))

        # Create attention mask
        attention = [1 if token_id != tokenizer.pad_token_id else 0 for token_id in ids]

        st_input_ids = torch.cat((st_input_ids, torch.tensor(ids).unsqueeze(0)), dim = 0)
        st_attention_mask = torch.cat((st_attention_mask, torch.tensor(attention).unsqueeze(0)), dim=0)

    labels_batch = {k: examples[k] for k in examples.keys() if k in pos_trig}

    # create numpy array of shape (batch_size, num_labels)
    labels_matrix = np.zeros((len(examples['suggestive_texts']), len(pos_trig)))

    # fill numpy array
    for idx, label in enumerate(pos_trig):
        labels_matrix[:, idx] = labels_batch[label]

    labels = torch.tensor(labels_matrix.tolist())

    return {
        "speakers_utterances_input_ids": speakers_utterances_input_ids,
        "speakers_utterances_attention_mask": speakers_utterances_attention_mask,
        "emotions_utterances_input_ids": emotions_utterances_input_ids,
        "emotions_utterances_attention_mask": emotions_utterances_attention_mask,
        "suggestive_text_ids": st_input_ids,
        "suggestive_text_mask": st_attention_mask,
        "labels": labels
    }

############################### VISUALIZATIONS ######################################################

def plot_dist_instances(count_instances_tr,count_instances_val,count_inst_list=None,splits=["Train","Validation","Test"],test=False):
    """
    This function plots side by side the histogram and cumulative distribution of the instances 
    of dialogues in the dataset
    """

    plt.figure(figsize=(15, 6),dpi=350)
    plt.subplot(1, 2, 1) # Subplot for histogram of the instances
    

    sns.histplot(data=[count_instances_tr,count_instances_val], stat='percent',edgecolor='white',discrete=True,multiple="dodge")
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

    df['nb_utterences'] = df['utterances'].apply(len) # Calculate number of utterances in the dialogue and create a new column
    plt.figure(figsize=(16, 7),dpi=250)

    # Plot the histogram of utterance lengths on the left
    plt.subplot(1, 2, 1)
    sns.histplot(data=df['nb_utterences'],binwidth=1, stat='percent',edgecolor='white')

    plt.title('Number of utterances per dialogue',fontsize=14)
    plt.xlabel('Utterance Length (number of utterances)',fontsize=12)
    plt.ylabel('Percent',fontsize=12)

    plt.xticks(np.arange(0, 26, 1),fontsize=12)  
    formatter = FuncFormatter(lambda y, _: f'{int(y)}%') # add percentage sign next to y ticks
    plt.gca().yaxis.set_major_formatter(formatter)
    plt.yticks(np.arange(0, 11, 1),fontsize=12)

    # Create the boxplot on the right
    plt.subplot(1, 2, 2)
    ax = sns.boxplot(x='nb_utterences', data=df,fill=False)
    

    # Calculate quartiles
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
    # Create a histogram of word counts
    plt.figure(figsize=(15, 6), dpi=350)
    plt.subplot(1, 2, 1)

    sns.histplot(lengths_array, binwidth=4,binrange=[0,lengths_array.max()],stat='percent',edgecolor='white')

    plt.xticks(np.arange(0,lengths_array.max()+1,4))
    plt.yticks(np.arange(0, 50, 5))
    plt.xlabel('Number of Words')
    plt.title('Histogram of Word Counts per Utterance')
    formatter = FuncFormatter(lambda y, _: f'{int(y)}%') # add percentage sign next to y ticks
    plt.gca().yaxis.set_major_formatter(formatter)

    # Plot cumulative plot
    plt.subplot(1, 2, 2)
    sns.histplot(data=lengths_array, binrange=[0,48], binwidth=2, element='step', fill=False, cumulative=True, stat='density')
    plt.xticks(np.arange(0, 49, 4))
    plt.yticks(np.arange(0,1.1,0.1))

    formatter = FuncFormatter(lambda y, _: f'{int(y*100)}%') # add percentage sign next to y ticks
    plt.gca().yaxis.set_major_formatter(formatter)

    plt.xlabel('Number of words ')
    plt.ylabel('')
    plt.title("Cumulative distribution of words per utterance")
    plt.grid() 
    plt.show()

    plt.show()


def plot_num_words_dial(max_dialogues: list[str]):
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
