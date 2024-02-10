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

## Hugging Face libraries
from transformers import EvalPrediction


#region #### *--------------  BEGIN GENERIC UTILS SECTION --------------*

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

def get_emotions_freq(df_unique: pd.DataFrame):
    emotions_max_diag = df_unique['emotions']
    emotions_arr =  np.concatenate(np.array(emotions_max_diag))

    emotion_freq_df =  pd.DataFrame(pd.Series(emotions_arr)
                                        .value_counts(),columns=['Count'])\
                                        .reset_index().rename(columns={"index":"Emotion"}) # get the count for each frequency
    emotion_freq_dict = dict(zip(emotion_freq_df['Emotion'], emotion_freq_df['Count']))
    return emotion_freq_dict


def get_triggers_frequencies(df:pd.DataFrame, NUM_TRIGGERS,pos_trig):
    # This function returns the dictionary with the frequency of each trigger label 
    trig_freq_dict = {}
    df_efr = get_trigger_labels(df,NUM_TRIGGERS,pos_trig)
    # Iterate over each column in the DataFrame
    for column in df_efr.filter(like='trigger_'):
        # Compute the value counts for the column
        value_counts = df_efr[column].value_counts()
        # Add the value counts to the dictionary
        trig_freq_dict[column] = value_counts.get(1)   # Get the count of 1, default to 0 if not present
    return trig_freq_dict

#endregion # *-------------- END GENERIC UTILS SECTION --------------*


# region 
#### *-------------- BEGIN METRICS HELPER SECTION --------------*

def linear_scale_array(values, min_new=0, max_new=1):
    """This function performs a liner rescaling EFR predictions to the interval [0,1],
    this is needed because to decode correctly the predictions in the appropriate interval.   
    """
    min_old = np.min(values)
    max_old = np.max(values)
    scaled_values = ((values - min_old) / (max_old - min_old)) * (max_new - min_new) + min_new
    return scaled_values

def decod_pred_efr(preds: EvalPrediction): 
    """
    This function decodes the predictions back into the range [0,1].
    """
    probs = linear_scale_array(np.concatenate(preds))
    # next, use threshold to turn them into integer predictions
    y_pred = np.zeros(probs.shape)
    y_pred[np.where(probs >= 0.5)] = 1
    return y_pred

def decod_pred_efr_multilabel(preds):
    probs = linear_scale_array(preds)
    # next, use the threshold to turn them into integer predictions
    y_pred = np.zeros(probs.shape)
    y_pred[np.where(probs >= 0.5)] = 1
    return y_pred



#### *-------------- END METRICS HELPER SECTION --------------*
# endregion


# region 
##### *-------------- Preprocessing/Data handling section  --------------*

def get_trigger_labels(df: pd.DataFrame, num_pos_trig: int, pos_trig: list[str]) -> pd.DataFrame:
    """ 
    This function adds columns which will store the label for each single trigger in any given dialogue
    

    Args:
        df (pd.DataFrame): original dataframe with no additional trigger columns
        num_pos_trig (int): number of trigger columns to add
        pos_trig (list[str]): list that contains the names of the trigger columns to be added

    Returns:
        pd.DataFrame: Dataframe consisting o
    """    
    df = df.copy()

    # reverse the the list of triggers inside the trigger column in the dataframe
    reversed_triggers = [entry[-num_pos_trig:] for entry in df['triggers']]

    padded_reversed_triggers = []
    for elem in reversed_triggers: # loop over each element in the reversed_trigger
        # pad_size is computed as the number of zeros to add in order to reach the prescribed length (that is num_pos_trig which in our case is equal to 7)
        pad_size = num_pos_trig - len(elem) # if len(elem) == num_pos_trig or len(elem) < num_pos_trig  then no padding is added
        padding = [0.0]
        padded_reversed_triggers.append(padding * pad_size + elem) # use + operation between lists to concatenate them to the requested length

    df_pos_triggers = pd.DataFrame(padded_reversed_triggers, columns=pos_trig) # DataFrame of trigger labels in descendent order
    df = df.join(df_pos_triggers) # check if 'emotions_num' will be necessary later on

    return df



def suggestiveText(df: pd.DataFrame, task: str = 'ERC'):
    """ This function augments the dataset by adding an adverb that indicates the emotion of the speaker.
    Then the adverb is masked and the model will need to predict the masked token. This is done to exploit Masked language modeling
    through which BERT-like are trained. Thus making more apt for our task. 
    """    
    if task not in ['ERC', 'EFR']:
        print('Task not accepted. Please choose between ERC and EFR.')
        return

    df = df.copy()
    speakers = [s for s in df['speakers']]              # retrieve list of speakers  
    utterances = [u for u in df['utterances']]          # retrieve list of utterances
    suggestive_texts = []

    if task == 'ERC':
        for spkrs,utts in zip(speakers, utterances):    # loop through speaker utterance pair
            queries_list = []
            for focus in range(len(utts)):
                sugg = f''
                for i,(s,u) in enumerate(zip(spkrs,utts)):
                    if i == focus:
                        sugg += f'<s>{s} <mask> says: {u}</s> ' # add speaker + masking + utterance if we are focusing on that utterance
                    else:
                        sugg += f'{s} says: {u} '

                queries_list.append(sugg)
            suggestive_texts.append(queries_list)
    else:
        emotions = [e for e in df['emotions']]

        # define dictionary containing adverbs associated with emotions. Note that neutral is equivalent with no adverb given that it works as a 'negative' label
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

    # keep relevant columns for each task
    if task == 'ERC':
        return df.drop(columns=['dialogue_id','speakers','emotions','utterances','triggers'])
    else:
        return df.drop(columns=['dialogue_id','emotions_num'])


def preprocess_SuggestiveText_ERC(examples: pd.DataFrame, tokenizer, tok_max_len: int = 350):
    # Define custom preprocess to pass to the map function 
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
##### *-------------- Preprocessing/Data handling section  --------------*
# endregion