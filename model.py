import torch
import torch.nn as nn
from transformers import AutoModel


class IsoBN(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.cov = torch.zeros(config.hidden_size, config.hidden_size).cuda()
        self.std = torch.zeros(config.hidden_size).cuda()

    def forward(self, input, momentum=0.05, eps=1e-3, beta=0.5):
        if self.training:
            x = input.detach()
            n = x.size(0)
            mean = x.mean(dim=0)
            y = x - mean.unsqueeze(0)
            std = (y ** 2).mean(0) ** 0.5
            cov = (y.t() @ y) / n
            self.cov.data += momentum * (cov.data - self.cov.data)
            self.std.data += momentum * (std.data - self.std.data)
        corr = torch.clamp(self.cov / torch.ger(self.std, self.std), -1, 1)
        gamma = (corr ** 2).mean(1)
        denorm = (gamma * self.std)
        scale = 1 / (denorm + eps) ** beta
        E = torch.diag(self.cov).sum()
        new_E = (torch.diag(self.cov) * (scale ** 2)).sum()
        m = (E / (new_E + eps)) ** 0.5
        scale *= m
        return input * scale.unsqueeze(0).detach()


class ClsModel(nn.Module):
    def __init__(self, args, config):
        super().__init__()
        self.args = args
        self.config = config
        self.model = AutoModel.from_pretrained(args.model_name_or_path, config=config)
        self.classifier = nn.Linear(config.hidden_size, config.num_labels)
        self.loss_fnt = nn.CrossEntropyLoss()
        self.isobn = IsoBN(config)
        self.dropout = nn.Dropout(config.hidden_dropout_prob)

    def forward(self, input_ids, attention_mask=None, labels=None):
        pooled_output = self.model(
            input_ids=input_ids,
            attention_mask=attention_mask
        ).pooler_output
        if self.config.isobn:
            pooled_output = self.isobn(pooled_output)
        pooled_output = self.dropout(pooled_output)
        logits = self.classifier(pooled_output)
        outputs = (logits,)
        if labels is not None:
            loss = self.loss_fnt(logits, labels)
            outputs = (loss, ) + outputs
        return outputs
