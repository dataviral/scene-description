import torch
import torch.nn as nn

def loss(classification_true, classification_pred, langgen_true, langgen_pred, pad_idx=100):
    classification_loss = nn.functional.cross_entropy(classification_pred, classification_true)
    langgen_loss = nn.functional.cross_entropy
    (   
        langgen_pred.view(-1, langgen_pred.size(-1)), 
        langgen_true.view(-1),
        ignore_index=pad_idx
    )
    return classification_loss, langgen_loss