from typing import Optional, Sequence
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from torch import Tensor
from torch.utils.data import DataLoader 
import transformers
from transformers import AutoModel, AutoModelForSequenceClassification, AutoConfig
from utils import ensure_reproducibility
from transformers import TrainingArguments, Trainer
import os
from typing import Dict

def colgaCollator(batch):
    # Define custom collator to retrieve speakers, emotions, suggestive_text and labels
    speakers_utterances = {
        'input_ids': torch.stack([torch.tensor(item['speakers_utterances_input_ids']) for item in batch]),
        'attention_mask': torch.stack([torch.tensor(item['speakers_utterances_attention_mask']) for item in batch])
    }
    emotions_utterances = {
        'input_ids': torch.stack([torch.tensor(item['emotions_utterances_input_ids']) for item in batch]),
        'attention_mask': torch.stack([torch.tensor(item['emotions_utterances_attention_mask']) for item in batch])
    }
    suggestive_text = {
        'input_ids': torch.stack([torch.tensor(item['suggestive_text_ids']) for item in batch]),
        'attention_mask': torch.stack([torch.tensor(item['suggestive_text_mask']) for item in batch])
    }
    labels = torch.stack([torch.tensor(item['labels']) for item in batch])

    return {
        'speakers_utterances': speakers_utterances,
        'emotions_utterances': emotions_utterances,
        'suggestive_text': suggestive_text,
        'labels': labels
    }

# UTTERANCE_LEVEL SIZE = 128

class ConvExtractor(nn.Module):
    def __init__(self, checkpoint: str, device: torch.device):
        super(ConvExtractor, self).__init__()
        self.embedder = AutoModel.from_pretrained(checkpoint).to(device)
        self.embedder_config = AutoConfig.from_pretrained(checkpoint)
        self.input_conv = nn.Conv1d(in_channels = self.embedder_config.hidden_size, out_channels = 128, kernel_size = 3).to(device)
        self.bn1 = nn.BatchNorm1d(128)
        self.conv1 = nn.Conv1d(in_channels = 128, out_channels = 128, kernel_size = 4)
        self.conv2 = nn.Conv1d(in_channels = 128, out_channels = 128, kernel_size = 5)
        self.pool = nn.MaxPool1d(kernel_size = 2)
        self.conv3 = nn.Conv1d(in_channels = 128, out_channels = 256, kernel_size = 2)
        self.bn2 = nn.BatchNorm1d(256)
        self.fc = None # will be dynamically initialized in forward()

    def forward(self, x):
        x = self.embedder(**x)    # 100
        x = x.last_hidden_state.permute(0, 2, 1)
        x = self.input_conv(x)  # 98
        x = self.conv1(x)       # 95
        x = self.conv2(x)       # 91
        x = self.pool(x)        # 45
        x = self.bn1(x)
        x = F.relu(x)           
        x = self.conv3(x)       # 44
        x = self.pool(x)        # 22
        x = self.bn2(x)
        x = F.relu(x)
        x = x.reshape(x.shape[0], -1)   # flatten

        # Dynamically define the fully connected layer
        if self.fc is None:
            self.fc = nn.Linear(x.size(1), 128).to(x.device)
        
        x = self.fc(x)

        return x

class LocalNet(nn.Module):
    # Define local net to retrieve utterances (with suggestive text) and speaker as input  
    def __init__(self, checkpoint: str, device: torch.device):
        super(LocalNet, self).__init__()
        self.ext1 = ConvExtractor(checkpoint=checkpoint, device=device)
        self.ext2 = ConvExtractor(checkpoint=checkpoint, device=device)
        self.fc = nn.Linear(2*128, 128)
        self.dropout = nn.Dropout(p=0.4)

    def forward(self, x1, x2):
        x = torch.cat((F.relu(self.ext1(x1)),
                       F.relu(self.ext2(x2))), dim=1)
        x = self.fc(x)
        x = self.dropout(x)

        return x

class CoLGA(nn.Module):
    def __init__(self, checkpoint: str, device: torch.device, window_size: int = 7):
        super(CoLGA, self).__init__()
        self.window_size = window_size
        self.device = device
        self.globalNet = self.getGlobalNet(checkpoint).to(device)
        self.dropout_global = nn.Dropout(p=0.1, inplace=False)
        self.localNet = LocalNet(checkpoint=checkpoint, device=device)
        self.fc = nn.Linear(self.config.hidden_size+(self.window_size*128), self.window_size)
        self.dropout = nn.Dropout(p=0.4)
      

    def getGlobalNet(self, checkpoint: str):
        model = AutoModel.from_pretrained(checkpoint)
        self.config = AutoConfig.from_pretrained(checkpoint)
        
        return model
    
    def forward(self, x):
        x_global = F.relu(self.dropout_global(self.globalNet(**x['suggestive_text']).pooler_output))
        x_local = torch.empty((0)).to(self.device)
        for i in range(self.window_size):
            x_emo = {
                'input_ids': x['emotions_utterances']['input_ids'][:,i,:],
                'attention_mask': x['emotions_utterances']['attention_mask'][:,i,:]
            }
            x_spe = {
                'input_ids': x['speakers_utterances']['input_ids'][:,i,:],
                'attention_mask': x['speakers_utterances']['attention_mask'][:,i,:]
            }
            local_out = F.relu(self.localNet(x_emo, x_spe))
            x_local = torch.cat((x_local, local_out), dim=1)

        x = torch.cat((x_global, x_local), dim=1)
        x = self.fc(x)
        x = self.dropout(x)

        return x



class FocalLoss(nn.Module):
    """ Focal Loss, as described in https://arxiv.org/abs/1708.02002.

    Shape:
        - x: (batch_size, C) or (batch_size, C, d1, d2, ..., dK), K > 0.
        - y: (batch_size,) or (batch_size, d1, d2, ..., dK), K > 0.
    """

    def __init__(self,
                 alpha: Optional[Tensor] = None,
                 gamma: float = 0.,
                 reduction: str = 'mean'):
        """
        Args:
            alpha (Tensor, optional): Weights for each class.
            gamma (float, optional): A constant, as described in the paper.
            reduction (str, optional): 'mean', 'sum' or 'none'.
        """
        if reduction not in ('mean', 'sum', 'none'):
            raise ValueError('Reduction must be one of: "mean", "sum", "none".')

        super().__init__()
        self._alpha = alpha
        self._gamma = gamma
        self._reduction = reduction

        self._nll_loss = nn.NLLLoss(weight = self._alpha, reduction = 'none')
    
    def __repr__(self):
        arg_keys = ['alpha', 'gamma', 'reduction']
        arg_vals = [self.__dict__[k] for k in arg_keys]
        arg_strs = [f'{k}={v!r}' for k, v in zip(arg_keys, arg_vals)]
        arg_str = ', '.join(arg_strs)
        return f'{type(self).__name__}({arg_str})'

    def forward(self, input: Tensor, target: Tensor) -> Tensor:
        if input.ndim > 2:
            # (N, C, d1, d2, ..., dK) --> (N * d1 * ... * dK, C)
            c = input.shape[1]
            input = input.permute(0, *range(2, input.ndim), 1).reshape(-1, c)
            # (N, d1, d2, ..., dK) --> (N * d1 * ... * dK,)

        target = target.view(-1)

        # compute weighted cross entropy term: -alpha * log(pt)
        log_p = F.log_softmax(input, dim = -1)
        ce = self._nll_loss(log_p, target)

        # get true class column from each row
        all_rows = torch.arange(len(input))
        log_pt = log_p[all_rows, target]

        # compute focal term: (1 - pt)^gamma
        pt = log_pt.exp()
        focal_term = (1 - pt)**self._gamma

        # the full loss: -alpha * ((1 - pt)^gamma) * log(pt)
        loss = focal_term * ce

        if self._reduction == 'mean':
            loss = loss.mean()
        elif self._reduction == 'sum':
            loss = loss.sum()

        return loss

def focal_loss(alpha: Optional[Sequence] = None,
               gamma: float = 0.,
               reduction: str = 'mean',
               device: str = 'cuda',
               dtype = torch.float32) -> FocalLoss:
    """Factory function for FocalLoss.
    Args:
        alpha (Sequence, optional): Weights for each class. Will be converted to a Tensor if not None.
        gamma (float, optional): A constant, as described in the paper.
        device (str, optional): device to deploy
        reduction (str, optional): 'mean', 'sum' or 'none'.
        dtype (torch.dtype, optional): dtype to cast alpha to.

    Returns:
        A FocalLoss object
    """
    if alpha is not None:
        if not isinstance(alpha, Tensor):
            alpha = torch.tensor(alpha)
        alpha = alpha.to(device = device, dtype = dtype)
                   
    fl = FocalLoss(alpha = alpha, gamma = gamma, reduction = reduction)
    
    return fl


class MultiLabelFocalLoss(nn.Module):
    def __init__(self, alpha: Optional[Tensor] = None, gamma: float = 0., reduction: str = 'mean', device: torch.device = 'cuda'):
        super().__init__()
        self._alpha = alpha.to(device) if alpha is not None else None
        self._gamma = gamma
        self._reduction = reduction

        if reduction not in ('mean', 'sum', 'none'):
            raise ValueError('Reduction must be one of: "mean", "sum", "none".')

    def forward(self, input: Tensor, target: Tensor) -> Tensor:
        bce_loss = F.binary_cross_entropy_with_logits(input, target, reduction='none')

        pt = torch.sigmoid(input).detach()
        pt = torch.where(target == 1, pt, 1 - pt)
        focal_loss = self._alpha * (1 - pt) ** self._gamma * bce_loss

        if self._reduction == 'mean':
            return focal_loss.mean()
        elif self._reduction == 'sum':
            return focal_loss.sum()
        else:
            return focal_loss

class CustomScheduler:
    def __init__(self, optimizer, num_warmup_steps, num_training_steps, min_lr):
        self.optimizer = optimizer
        self.num_warmup_steps = num_warmup_steps
        self.num_training_steps = num_training_steps
        self.min_lr = min_lr
        self.current_step = 0
        self.last_lr = 0

    def step(self):
        if self.current_step < self.num_warmup_steps:
            lr = (self.optimizer.defaults['lr'] / self.num_warmup_steps) * self.current_step
        elif self.current_step < self.num_training_steps:
            lr = ((self.optimizer.defaults['lr'] - self.min_lr) / (self.num_training_steps - self.num_warmup_steps)) * (self.num_training_steps - self.current_step) + self.min_lr
        else:
            lr = self.min_lr

        for param_group in self.optimizer.param_groups:
            param_group['lr'] = lr

        self.last_lr = lr
        self.current_step += 1

    def get_last_lr(self):
        return [self.last_lr]

    def state_dict(self):
        return {}

class FocalLossTrainer(Trainer):
    def __init__(self, *args, focal_loss, **kwargs):
        super().__init__(*args, **kwargs)
        self.focal_loss = focal_loss

    # might remove it
    def log(self, logs: Dict[str, float]) -> None:
        logs["learning_rate"] = self._get_learning_rate()*pow(10, 6)
        super().log(logs)

    def create_optimizer_and_scheduler(self, num_training_steps):
        self.optimizer = torch.optim.AdamW(self.model.parameters(),
                                           lr=self.args.learning_rate,
                                           weight_decay=self.args.weight_decay)

        self.lr_scheduler = CustomScheduler(self.optimizer,
                                            num_warmup_steps = len(self.get_train_dataloader()),
                                            num_training_steps=num_training_steps//2, # num_training_steps halved for steeper decaying
                                            min_lr = self.args.learning_rate/10)

    def compute_loss(self, model, inputs, return_outputs=False):
        labels = inputs.pop("labels")
        outputs = model(**inputs)
        logits = outputs.logits
        # Compute Focal Loss
        loss = self.focal_loss(logits, labels)
        return (loss, outputs) if return_outputs else loss


def train_roberta(checkpoint: str, args: TrainingArguments,
                train_set, val_set, class_weigths,
                tokenizer, seed: int,
                compute_metrics, num_labels,
                id2label, label2id):
    
    ##### *-------- BEGIN UTILITIES SECTION  --------* ######
    ensure_reproducibility(seed) # setting the seed
    TABLE = '-' # outputting constant

    # Setting output directories
    out_dir = f"./train/roberta_ERC_{seed}"
    os.makedirs(out_dir, exist_ok=True)
    args.output_dir = out_dir

    args.seed = seed # set seed for hugging face Training Arguments
    ##### *-------- END UTILITIES SECTION  --------* ######

    ##### *-------- BEGIN MODEL DEFINITION SECTION  --------* ######
    model = AutoModelForSequenceClassification.from_pretrained(checkpoint,
                                                        num_labels=num_labels,
                                                        id2label=id2label,
                                                        label2id=label2id)
    
    to_freeze = 16
    for layer in model.roberta.encoder.layer[to_freeze:]: # unfreeze layers after to_freeze
        for param in layer.parameters():
            param.requires_grad = True

    # Define special tokens for performing suggestive text
    special_tokens_dict = {'bos_token': '<s>', 'eos_token': '</s>', 'mask_token': '<mask>'}
    num_added_toks = tokenizer.add_special_tokens(special_tokens_dict)

    # Resize the token embeddings so as to include the special added tokens for suggestive text
    model.resize_token_embeddings(len(tokenizer))

    ##### *-------- BEGIN OUTPUTTING UTILITIES SECTION  --------* ######
    print()
    print()
    print(f'{TABLE*20} MODEL: ROBERTA | TASK: ERC | SEED: {seed} {TABLE*20}') # we will use this model only for ERC task
    print()
    ##### *-------- END UTILITIES SECTION  --------* ######

    ##### *-------- BEGIN TRAINING MODEL SECTION  --------* ######
    focal_trainer = FocalLossTrainer( # Instantiate our custom trainer which overloads the default Trainer of Hugging face, this was needed to add the focal loss
        model = model,
        args = args,
        train_dataset = train_set,
        eval_dataset = val_set,
        tokenizer = tokenizer,
        focal_loss = focal_loss(alpha=class_weigths, gamma=2, reduction = 'mean'), 
        compute_metrics = compute_metrics
    )

    # Train the model
    focal_trainer.train()
    ##### *-------- END TRAINING MODEL SECTION  --------* ######
    
    print()

    pass
