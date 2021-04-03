"""
Scene Desription Model
    Language Understanding: LSTM / Transformer
    Vision Understanding: Resnet18
    Classification: Linear
    Langauge Generation: LSTM
"""

import torch
import torch.nn as nn
from transformers import BertTokenizerFast, BertModel
from torchvision.models import resnet18
from . import global_vars

class LanguageInput(nn.Module):

    def __init__(self):
        super(LanguageInput, self).__init__()

        self.berttokenizer = BertTokenizerFast.from_pretrained('bert-base-uncased')
        self.bertmodel = BertModel.from_pretrained('bert-base-uncased')

        # Do not Finetune
        for i, p in enumerate(self.bertmodel.parameters()):
            p.requires_grad = False
    
    def forward(self, inputs, device="cuda"):
        toks = self.berttokenizer(inputs, return_tensors="pt", padding=True).to(device)
        op = self.bertmodel(**toks)
        return op.last_hidden_state[:, 0, :]


class VisionInput(nn.Module):

    def __init__(self):
        super(VisionInput, self).__init__()

        self.resnetmodel = nn.Sequential(*list(resnet18(pretrained=True).children())[:-1])

        # Finetune only final layer
        for i, p in enumerate(self.resnetmodel.parameters()):
            if i == 7: break
            p.requires_grad = False
    
    def forward(self, inputs):
        op = self.resnetmodel(inputs)
        return op.view(op.size(0), -1)


class Classification(nn.Module):

    def __init__(self, num_answers):
        super(Classification, self).__init__()

        self.num_answers = num_answers
        self.tanh = nn.Tanh()
        self.linear1 = nn.Linear(512 + 768, 512)
        self.linear2 = nn.Linear(512, num_answers)
    
    def forward(self, inputs):
        reduced_input_embeddings = self.tanh(self.linear1(inputs))
        op = self.linear2(reduced_input_embeddings)
        return op, reduced_input_embeddings

class LanguageGeneration(nn.Module):

    def __init__(self, vocab_size):
        super(LanguageGeneration, self).__init__()

        self.vocab_size = vocab_size
        self.embedding_dim = 128

        self.embedding = nn.Embedding(vocab_size, self.embedding_dim)
        self.lstm1 = nn.LSTM(
            input_size=self.embedding_dim,
            hidden_size=512,
            batch_first=True
        )
        self.lstm2 = nn.LSTM(
            input_size=512,
            hidden_size=512,
            batch_first=True
        )
        self.projection = nn.Linear(512, vocab_size)
    
    def forward(self, hidden_state, inputs=None, max_len=30, sos_tok_id=global_vars.BOS_IDX, device="cuda"):
        # hidden_state: (batch_size, ldim + vdim)

        hidden_state = hidden_state.view(1, hidden_state.size(0), hidden_state.size(1))
        if inputs is not None:
            # Train phase
            embs = self.embedding(inputs)
            op, _ = self.lstm1(embs, (hidden_state, torch.zeros_like(hidden_state).to(device)))
            op, _ = self.lstm2(op)
            op = self.projection(op)
            return op
        else:
            # Eval phase

            batch_size = hidden_state.size(1)
            
            inputs = torch.ones((batch_size, 1)).long().to(device) * sos_tok_id

            hidden1 = (hidden_state, torch.zeros_like(hidden_state).to(device))
            hidden2 = None
            outputs = []
            for i in range(max_len):
                embs = self.embedding(inputs)
                op, hidden1 = self.lstm1(embs, hidden1)
                op, hidden2 = self.lstm2(op, hidden2)
                op = self.projection(op)
                inputs = nn.functional.softmax(op, dim=-1).argmax(dim=-1)
                outputs.append(op)
            return torch.cat(outputs, dim=1)

class SceneDescription(nn.Module):

    def __init__(self, num_answers, vocab_size):
        super(SceneDescription, self).__init__()

        self.num_answers = num_answers
        self.vocab_size = vocab_size

        self.language_understanding = LanguageInput()
        self.vision_understanding = VisionInput()
        self.classification = Classification(num_answers)
        self.language_generation = LanguageGeneration(vocab_size)
    
    def forward(self, language_input, vision_input, language_output=None, max_len=30, sos_tok_id=global_vars.BOS_IDX, device="cuda"):
        language_embeddings = self.language_understanding(language_input, device=device)
        vision_embeddings = self.vision_understanding(vision_input)

        vqa_embeddings = torch.cat([language_embeddings, vision_embeddings], dim=-1)
        classification_op, vqa_embeddings = self.classification(vqa_embeddings)

        generation_op = self.language_generation(
            vqa_embeddings,
            inputs=language_output,
            max_len=max_len,
            sos_tok_id=sos_tok_id,
            device=device
        )

        return classification_op, generation_op


