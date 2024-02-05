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

# UTTERANCE_LEVEL SIZE = 128

class ConvExtractor(nn.Module):
    def __init__(self, checkpoint: str):
        super(ConvExtractor, self).__init__()
        self.embedder = AutoModel.from_pretrained(checkpoint)
        self.embedder_config = AutoConfig.from_pretrained(checkpoint)
        self.input_conv = nn.Conv1d(in_channels = self.embedder_config.hidden_size, out_channels = 128, kernel_size = 3)
        self.conv1 = nn.Conv1d(in_channels = 128, out_channels = 128, kernel_size = 4)
        self.conv2 = nn.Conv1d(in_channels = 128, out_channels = 128, kernel_size = 5)
        self.pool = nn.MaxPool1d(kernel_size = 2)
        self.conv3 = nn.Conv1d(in_channels = 128, out_channels = 256, kernel_size = 2)
        self.fc = nn.Linear(256*22, 128)  # FIXME input_neurons = (input_length - kernel_size + 2 * padding) / stride + 1

    def forward(self, x):
        x = self.embedder(**x)    # 100
        x = x.last_hidden_state.permute(0, 2, 1)
        x = self.input_conv(x)  # 98
        x = self.conv1(x)       # 95
        x = self.conv2(x)       # 91
        x = self.pool(x)        # 45
        x = F.relu(x)           
        x = self.conv3(x)       # 44
        x = self.pool(x)        # 22
        x = F.relu(x)
        x = x.reshape(x.shape[0], -1)   # flatten
        x = self.fc(x)

        return x

class LocalNet(nn.Module):
    def __init__(self, checkpoint: str):
        super(LocalNet, self).__init__()
        self.ext1 = ConvExtractor(checkpoint=checkpoint)
        self.ext2 = ConvExtractor(checkpoint=checkpoint)
        self.fc = nn.Linear(2*128, 128)
        self.dropout = nn.Dropout(p=0.4)

    def forward(self, x1, x2):
        x = torch.cat((F.relu(self.ext1(x1)),
                       F.relu(self.ext2(x2))), dim=1)
        x = self.fc(x)
        x = self.dropout(x)

        return x

class CoLGA(nn.Module):
    def __init__(self, checkpoint: str, window_size: int = 7):
        super(CoLGA, self).__init__()
        self.window_size = window_size
        self.globalNet = self.getGlobalNet(checkpoint)
        self.dropout_global = nn.Dropout(p=0.1, inplace=False)
        self.localNet = LocalNet(checkpoint=checkpoint)
        self.fc = nn.Linear(self.config.hidden_size+(self.window_size*128), self.window_size)
        self.dropout = nn.Dropout(p=0.4)

    def getGlobalNet(self, checkpoint: str):
        model = AutoModelForSequenceClassification.from_pretrained(checkpoint)
        self.config = AutoConfig.from_pretrained(checkpoint)
        
        # customize classifier
        model.classifier = nn.Linear(self.config.hidden_size, self.config.hidden_size)
        
        return model
    
    def forward(self, x):
        x_global = F.relu(self.dropout_global(self.globalNet(**x['suggestive_text']).logits))
        x_local = torch.empty((0))
        for i in range(self.window_size):
            shape = x['emotions_utterances']['input_ids'].shape
            print(f'in loop: {shape}')
            x_emo = {
                'input_ids': x['emotions_utterances']['input_ids'][i],
                'attention_mask': x['emotions_utterances']['attention_mask'][i]
            }
            x_spe = {
                'input_ids': x['speakers_utterances']['input_ids'][i],
                'attention_mask': x['speakers_utterances']['attention_mask'][i]
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
