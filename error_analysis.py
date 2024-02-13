import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
import pprint
import warnings
import utils as ut

from baselines import baselines_ERC, baselines_EFR
from utils import decod_pred_efr, restructuring_flat_preds

from sklearn.exceptions import UndefinedMetricWarning
from sklearn.metrics import classification_report, confusion_matrix, f1_score

from transformers import TrainingArguments, Trainer
from transformers import AutoModelForSequenceClassification



def get_errors_df(confusion_matrix, labels) -> pd.DataFrame:
    """
    This function returns a dataframe with three columns one for the ground truths, one for the predicted labels and the last 
    for the relative number of mismatched labels.
    """
    df_conf_matrix = pd.DataFrame(confusion_matrix, columns=labels, index=labels)
    results = []
    for row_idx, row in df_conf_matrix.iterrows(): # Iterate over the confusion matrix rows
        for col_idx, value in row.items(): # iterate over the 
            # this condition states that if i have a value in a field that is not the main diagonal of the confusion matrix and its value is greater than 0 than i have a mistake
            if (value > 0 and row_idx != col_idx): 
                results.append((row_idx, col_idx, value)) 
    return pd.DataFrame(results, columns=['true', 'predicted', 'errors'])


# region 
#### *---------- BEGIN INFER SECTION ----------*
# # ERC
# unroll_f1s_erc = {}
# f1s_erc = {}
# # EFR
# unroll_f1s_efr = {}
# f1s_efr = {}


def infer_baseline_dummy(task, df_train, df_test, structuring_df, 
                         target_labels, seed: int):
    # plotting utilities
    SECTION_SEPARATOR = '--'
    MODEL_SEPARATOR = '**'

    with warnings.catch_warnings():
        warnings.filterwarnings("ignore", category=UndefinedMetricWarning)

        f1s = {}
        unroll_f1s = {}
        sequence_f1s = {}

        # Dummy baselines
        if task not in ['ERC', 'EFR']:
            print('Task not accepted. Please choose between ERC and EFR.')
            return

        if task == 'ERC':
            dummies_model = baselines_ERC(df_train, seeds=[seed])
            y_true = np.concatenate(df_test['emotions_num'])
        else:
            dummies_model = baselines_EFR(df_train, seeds=[seed])
            y_true = np.concatenate(df_test['triggers'])

        uni_preds = dummies_model[f'uniform_{task}_{seed}'].predict(np.concatenate(df_test['utterances']))
        maj_preds = dummies_model[f'majority_{task}'].predict(np.concatenate(df_test['utterances']))

        uni_report = classification_report(y_true, uni_preds, target_names=target_labels)
        maj_report = classification_report(y_true, maj_preds, target_names=target_labels)

        print(f'{SECTION_SEPARATOR*8} BASELINES on seed {seed} {SECTION_SEPARATOR*8}')
        print()
        print(f'Uniform classifiers:') 
        print(uni_report)
        print()
        print(f'{MODEL_SEPARATOR*27}')
        print()
        print(f'Majority classifiers:') 
        print(maj_report)

        uni_report_dict = classification_report(y_true, uni_preds, target_names=target_labels, output_dict=True)
        maj_report_dict = classification_report(y_true, maj_preds, target_names=target_labels, output_dict=True)

        reports_dict = {'uni': uni_report_dict, 'maj': maj_report_dict}

        for key, report in reports_dict.items():

            if key == 'uni':
                model_name = f'UNIFORM {task}'
            else:
                model_name = f'MAJORITY {task}'
            
            f1s_targets = {}

            for name, measures_dict in report.items():

                if name in target_labels:
                    f1s_targets[name] = round(measures_dict['f1-score'], 3)
                if name == 'macro avg':
                    unroll_f1s[f'{model_name}'] = round(measures_dict['f1-score'], 3)
            f1s[f'{model_name}'] = f1s_targets
            
        if task.upper() == 'ERC':
            # uniform
            all_predictions_dialog, all_labels_dialog = ut.restructuring_flat_preds(uni_preds, structuring_df, 'emotions_num')
            sequence_f1s['UNIFORM ERC'] = round(ut.compute_sequence_f1_for_dialogues(all_predictions_dialog, all_labels_dialog), 3)
            # majority
            all_predictions_dialog, all_labels_dialog = ut.restructuring_flat_preds(maj_preds, structuring_df, 'emotions_num')
            sequence_f1s['MAJORITY ERC'] = round(ut.compute_sequence_f1_for_dialogues(all_predictions_dialog, all_labels_dialog), 3)
        
        else:
            # uniform
            all_predictions_dialog, all_labels_dialog = ut.restructuring_flat_preds(uni_preds, structuring_df, 'triggers')
            sequence_f1s['UNIFORM EFR'] = round(ut.compute_sequence_f1_for_dialogues(all_predictions_dialog, all_labels_dialog), 3)
            # majority
            all_predictions_dialog, all_labels_dialog = ut.restructuring_flat_preds(maj_preds, structuring_df, 'triggers')
            sequence_f1s['MAJORITY EFR'] = round(ut.compute_sequence_f1_for_dialogues(all_predictions_dialog, all_labels_dialog), 3)

    return f1s, unroll_f1s, sequence_f1s

def ordering_dict(ordered_dict, id2label):
    # This function is needed to correctly order the label in the confusion matrices
    reordered_dict = {}
    for k in ordered_dict.keys():
        for i in range(len(ordered_dict)):
            if id2label[i] == k:
                reordered_dict[i] = k
    return reordered_dict

def infer_bert_like(args: TrainingArguments, model_type:str, task: str,
                    test_set, structuring_df, tokenizer, id2label: dict,
                    compute_metrics, seed: int, ckpt):

    # model_type: fine_tuned_bert | full_bert

    with warnings.catch_warnings():
        warnings.filterwarnings("ignore", category=UndefinedMetricWarning)

        ####### CONFIG SECTION #######
        # plotting utilities
        SEED_SEPARATOR = '---'
        MODEL_SEPARATOR = '*'

        f1s = {} # for unroll only, this dictionary contains the f1 of the single classes
        unroll_f1s = {}
        sequence_f1s = {}

        model_type_str = ' '.join(model_type.split("_")).upper()  # for printing purpose

        ####### END CONFIG SECTION #######
        
        ####### MODEL SECTION #######

        # Loop over model types using the best checkpoint
        output_dir = f"./trained_models/{model_type}_{task}_{seed}/checkpoint-{ckpt}"
        args.output_dir = output_dir

        model = AutoModelForSequenceClassification.from_pretrained(output_dir) # retrieve the model checkpoint from the directory provided as input

        trainer = Trainer(
            model,
            args,
            tokenizer = tokenizer,
            compute_metrics = compute_metrics
        )

        predictions = trainer.predict(test_set)

        if task.upper() == 'ERC':
            preds = np.argmax(predictions.predictions, axis=1)
        else:
            preds = ut.decod_pred_efr(predictions.predictions)

        f1s[f'{model_type_str} {task.upper()}'] = predictions.metrics['test_f1s']
        id2label = ordering_dict(predictions.metrics['test_f1s'], id2label)

        unroll_f1s[f'{model_type_str} {task.upper()}'] = predictions.metrics['test_macro_f1']

        # Sequence f1
        if task.upper() == 'ERC':
            all_predictions_dialog, all_labels_dialog = ut.restructuring_flat_preds(preds, structuring_df, 'emotions_num')
            sequence_f1s[f'{model_type_str} {task.upper()}'] = ut.compute_sequence_f1_for_dialogues(all_predictions_dialog, all_labels_dialog)
        else:
            all_predictions_dialog, all_labels_dialog = ut.restructuring_flat_preds(preds, structuring_df, 'triggers')
            sequence_f1s[f'{model_type_str} {task.upper()}'] = ut.compute_sequence_f1_for_dialogues(all_predictions_dialog, all_labels_dialog)
        
        
        print()
        print(f'{MODEL_SEPARATOR*20} MODEL: {model_type_str} | TASK: {task.upper()} | SEED: {seed} {MODEL_SEPARATOR*20}')
        print()
        print('F1s:')
        pprint.pprint(f'{predictions.metrics["test_f1s"]}')
        print(f'Unrolled sequence f1: {round(predictions.metrics["test_macro_f1"], 3)}')
        print()
        print(f'Sequence f1: {round(sequence_f1s[f"{model_type_str} {task.upper()}"], 3)}')
        print()
        print(f'{SEED_SEPARATOR*30}')
        print()

        preds_label = [id2label[i] for i in preds]
        true_label = [id2label[i] for i in test_set['labels']]
        conf_matrix = confusion_matrix(true_label, preds_label)

        if task.upper() == 'ERC':
            _, ax = plt.subplots(figsize=(12, 8))
        else:
            _, ax = plt.subplots(figsize=(5, 4))

        sns.heatmap(conf_matrix, annot=True, fmt='d', cmap='Blues', xticklabels=id2label.values(), yticklabels=id2label.values(), ax=ax)

        ax.set_title(f'Confusion Matrix - {model_type_str} {task.upper()}')
        ax.set_xlabel('Predicted')
        ax.set_ylabel('True')

        plt.tight_layout()
        plt.show()

        # get worst/best predictions counting the number of wrongly predicted labels
        errors_df = get_errors_df(conf_matrix, labels=id2label.values())
        errors_df = errors_df.sort_values('errors', ascending=False, ignore_index=True)

        print()
        print(f'{SEED_SEPARATOR*30}')
        print()
        print('Worst predictions:')
        print()
        print(errors_df[:10])

        ###### END BERT BASELINE SECTION #######

        return f1s, unroll_f1s, sequence_f1s
#### *---------- END INFER SECTION ----------*



def plot_f1_w_distribution(df_train: pd.DataFrame, task: str, f1s: dict, unroll_f1s: dict, 
                           sequence_f1s: dict, id2label: dict) -> None:
    
    if task not in ['ERC', 'EFR']:
        print('Task not accepted. Please choose between "ERC" and "EFR".')
        return

    _, ax1 = plt.subplots(figsize=(14, 6), dpi=350)
    markers = ['o','X','d','*','p']
    sc_colors = ['limegreen','magenta', 'deepskyblue', 'firebrick', 'orange']

    if task == 'ERC':
        x = np.concatenate(df_train['emotions_num'])
    else:
        x = np.concatenate(df_train['triggers'])
    _, counts = np.unique(x, return_counts=True)

    for i, _ in id2label.items():
        emotion_norm = counts[i] / len(x)

        ax1.bar(i, emotion_norm, width=0.2, color='cornflowerblue', alpha=0.4, zorder=2)

        for j, (model_name, scores) in enumerate(f1s.items()):
            f1 = scores[id2label[i]]
            ax1.scatter(i, f1, color=sc_colors[j], marker=markers[j],
                        label=f'{model_name}' if i==0 else '', zorder=3)

    # plot line connecting per-class f1s for each model
    for j, (_, scores) in enumerate(f1s.items()):
        reordered_scores = [scores[v] for v in id2label.values()]
        plt.plot(range(len(id2label)), reordered_scores, color=sc_colors[j], linestyle='-', linewidth=1,
                     zorder=1)

    # # Plot unroll F1-scores
    for i, unroll_f1s in enumerate(unroll_f1s.values()):
        plt.scatter(len(id2label), unroll_f1s, color=sc_colors[i], marker=markers[i])

    # Plot sequence F1-scores
    for i, sequence_f1s in enumerate(sequence_f1s.values()):
        plt.scatter(len(id2label)+1, sequence_f1s, color=sc_colors[i], marker=markers[i])

    ax1.set_ylabel('F1-score')
    ax1.set_xticks(np.arange(len(id2label) + 2))
    ax1.set_xticklabels(list(id2label.values()) + ['Unroll seq f1', 'Seq f1'])
    ax1.set_yticks(np.arange(0, 1.1, 0.1))
    ax1.legend(loc='upper center')
    ax1.grid(linestyle='dashed')
    ax1.set_ylim([0, 1])
    ax2 = ax1.twinx()
    ax2.set_ylabel('Normalized distribution of values', color='royalblue')
    ax2.set_yticks(np.arange(0, 1.1, 0.1))
    ax2.tick_params(axis='y', labelcolor='royalblue')

    plt.show()
# endregion
