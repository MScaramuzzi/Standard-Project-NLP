import pandas as pd
import numpy as np
from sklearn.dummy import DummyClassifier
from sklearn.metrics import classification_report
from utils import ensure_reproducibility
from transformers import TrainingArguments, Trainer
import os
from transformers import AutoModelForSequenceClassification, AutoTokenizer

def baselines_ERC(df: pd.DataFrame, seeds: list[int]):
    # This function defines the dummy baselines for the ERC task, and fits them to the dataset

    utterances = df['utterances'] # retrieve utterances
    emotions = df['emotions_num'] # retrieve emotions

    utterances = np.concatenate(utterances)
    emotions = np.concatenate(emotions)

    #### Define baselines
    majority_classifier = DummyClassifier(strategy="prior") # Majority baseline won't need to be fitted for multiple seeds given that it is not randomic
    majority_classifier.fit(utterances, emotions)

    # Uniform baseline need to be fitted for each seed
    uniform_dict = {}
    for seed in seeds:
        seed_id = f'uniform_ERC_{seed}'
        uniform_dict[seed_id] = DummyClassifier(strategy="uniform", random_state=seed)
        uniform_dict[seed_id].fit(utterances, emotions)

    baseline = {
        'majority_ERC': majority_classifier,
        **uniform_dict # unpacking dictionary containing the uniform models trained on each seed
    }
    return baseline


def baselines_EFR(df: pd.DataFrame, seeds: list[int]):
    # This function defines the dummy baselines for the EFR task, and fits them to the dataset

    utterances = df['utterances'] # retrieve utterances
    triggers = df['triggers']

    utterances = np.concatenate(utterances)
    triggers = np.concatenate(triggers)

    #### Define baselines
    majority_classifier = DummyClassifier(strategy="prior") # Majority baseline won't need to be fitted for multiple seeds given that it is not randomic
    majority_classifier.fit(utterances, triggers)

    # Uniform baseline need to be fitted for each seed
    uniform_dict = {}
    for seed in seeds:
        seed_id = f'uniform_EFR_{seed}'
        uniform_dict[seed_id] = DummyClassifier(strategy="uniform", random_state=seed)
        uniform_dict[seed_id].fit(utterances, triggers)

    baseline = {
        'majority_EFR': majority_classifier,
        **uniform_dict # unpacking dictionary containing the uniform models trained on each seed
    }

    return baseline


# This function uses the defined dummy baselines of both task (ERC and EFR) to predict the output on the test set and print the classification report for every model.
def train_baselines_dummy(df_train: pd.DataFrame, df_val:pd.DataFrame, seeds: list[int], 
                          id2label: dict, unique_emotions: np.array, task: str):
    tbl = '---'

    if task not in ['ERC', 'EFR']:
        print('The task is either ERC or EFR.')
        return None

    if task == 'ERC':
        results_ERC = {}
        # Defining and training erc baseline
        models_ERC_dict = baselines_ERC(df_train, seeds)  # The dict contains all the models: the first is the majority, then we have a uniform model for each seed.

        # Predicting using erc baseline. Note that training majority baseline is independent of the seed.
        for key, model in models_ERC_dict.items():
            # TBD: spiega markdown/commento che non serve predictare le baseline sul validation
            results_ERC[key] = model.predict(np.concatenate(df_val['utterances']))

        # Printing classification reports
        for key, model in models_ERC_dict.items():
            name = key.split('_') # Uniform -> e.g.: 'uniform_ERC_seed'. Majority -> e.g.: majority_ERC
            if name[0] == 'uniform': # select uniform baseline
                print(f'\t[TASK] {name[1]} \t|\t[MODEL] {name[0]} \t|\t [SEED] {name[2]}')
                print(f'{tbl*20}')
            else:  # majority baseline
                print(f'\t\t[TASK] {name[1]} \t\t\t|\t\t[MODEL] {name[0]}')
                print(f'{tbl*20}')
            rep = classification_report(np.concatenate(df_val['emotions_num']), results_ERC[key], labels=list(id2label.keys()), target_names=unique_emotions)
            print(rep)
            print()
            print('***'*20,'\n\n')

    if task == 'EFR':
        results_EFR = {}
        # Defining and training efr baseline
        models_EFR_dict = baselines_EFR(df_train, seeds)

        # Predicting using efr baseline. Note that training majority baseline is independent of the seed.
        for key, model in models_EFR_dict.items():
            results_EFR[key] = model.predict(np.concatenate(df_val['utterances']))

        # Printing classification reports
        for key, model in models_EFR_dict.items():
            name = key.split('_')
            if name[0] == 'uniform': # select uniform baseline
                print(f'\t[TASK] {name[1]} \t|\t[MODEL] {name[0]} \t|\t [SEED] {name[2]}')
                print(f'{tbl*20}')
            else:  # majority model
                print(f'\t\t[TASK] {name[1]} \t\t\t|\t\t[MODEL] {name[0]}')
                print(f'{tbl*20}')
            rep = classification_report(np.concatenate(df_val['triggers']), results_EFR[key], target_names=['No-trigger','Trigger'])
            print(rep)
            print()
            print('***'*20,'\n\n')


def train_baseline_bert(model_name: str, task: str, 
                        checkpoint: str, args: TrainingArguments,
                        train_set, val_set,
                        tokenizer, seed: int,
                        compute_metrics, num_labels,
                        id2label, label2id):
    
    # model_name = 'fine_tuned_bert | full_bert
    # task = 'ERC' or 'EFR'
    ensure_reproducibility(seed) # setting the seed
    TABLE = '-' # outputting constant

    # Setting output directories
    out_dir = f"./train/{model_name}_{task}_{seed}"
    os.makedirs(out_dir, exist_ok=True)
    args.output_dir = out_dir

    args.seed = seed # set seed for hugging face Training Arguments

    # Check the task given in the input parameters is correct
    if task not in ['ERC', 'EFR']:
            print('Task not accepted. Please choose between ERC and EFR.')
            return

    if task == 'ERC':     # instantiate model for ERC task
        model = AutoModelForSequenceClassification.from_pretrained(checkpoint,
                                                            num_labels=num_labels,
                                                            id2label=id2label,
                                                            label2id=label2id)
        
    else: # task is EFR
        model = AutoModelForSequenceClassification.from_pretrained(checkpoint,num_labels=num_labels,
                                                                   id2label=id2label,
                                                                   label2id=label2id
                                                                   ) # EFR is binary classification on triggers i.e. num_labels = 1

    if model_name == 'full_bert':
        for param in model.parameters(): # unfreeze all bert weights to perform the fine-tuning on the whole architecture
                param.requires_grad = True

    trainer = Trainer(
        model,
        args,
        train_dataset = train_set,
        eval_dataset = val_set,
        tokenizer = tokenizer,
        compute_metrics = compute_metrics
    )

    # outputting utilities
    print()
    print()
    print(f'{TABLE*20} MODEL: {model_name} | TASK: {task} | SEED: {seed} {TABLE*20}')
    print()
    trainer.train()
    print()


    pass