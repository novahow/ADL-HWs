from typing import Dict

import torch
from torch.nn import Embedding
import torch.nn as nn

class SeqClassifier(torch.nn.Module):
    def __init__(
        self,
        embeddings: torch.tensor,
        hidden_size: int,
        num_layers: int,
        dropout: float,
        bidirectional: bool,
        num_class: int,
    ) -> None:
        super(SeqClassifier, self).__init__()
        self.h2 = 256
        self.embed = Embedding.from_pretrained(embeddings, freeze=False)
        # TODO: model architecture
        self.gru = nn.GRU(embeddings.shape[-1], hidden_size, num_layers, True, True, dropout, bidirectional)
        self.d1 = nn.Dropout()
        self.l1 = nn.Linear((1 + bidirectional) * hidden_size, self.h2)
        self.relu1 = nn.ReLU()
        self.bn1 = nn.BatchNorm1d(self.h2)
        self.l2 = nn.Linear(self.h2, num_class)
    @property
    def encoder_output_size(self) -> int:
        # TODO: calculate the output dimension of rnn
        raise NotImplementedError

    def forward(self, batch) -> Dict[str, torch.Tensor]:
        # TODO: implement model forward
        out = self.embed(batch)
        out, h = self.gru(out)
        # print(len(out), out[0].shape)
        # out = 
        out = self.l1(out)
        out = self.d1(out)
        # out = self.bn1(out)
        out = self.relu1(out)
        out = self.l2(out)
        return torch.mean(out, dim=1)




class Slottagger(nn.Module):
    def __init__(
        self,
        embeddings: Dict[str, torch.tensor],
        hidden_size: int,
        num_layers: int,
        dropout: float,
        bidirectional: bool,
        num_class: Dict[str, int],
        task=['slot', 'intent'],
        device='cuda'
    ) -> None:
        super(Slottagger, self).__init__()
        self.idim = embeddings[task[0]].shape[-1]
        self.hiddim = hidden_size
        self.qkv_proj = nn.Linear((1 + bidirectional) * self.hiddim, self.hiddim * 3)
        self.sfx = nn.Softmax(dim=2)
        self.gru = nn.GRU(self.idim, hidden_size, num_layers, True, True, dropout, bidirectional)
        self.dgru = nn.GRU(hidden_size, hidden_size, num_layers, True, True, dropout, False, 0)
        self.drop = nn.Dropout(dropout)
        self.embed = {t: Embedding.from_pretrained(embeddings[t], freeze=False) for t in task}
        self.o_proj = nn.Linear(self.hiddim, self.hiddim)
        self.intcls = nn.Linear(self.hiddim, num_class[task[-1]])
        self.tagl = nn.Sequential(
            nn.Linear(self.hiddim, self.hiddim // 2),
            nn.LayerNorm(self.hiddim // 2),
            nn.Dropout(dropout),
            nn.ReLU(),
            nn.Linear(self.hiddim // 2, num_class[task[0]])
        )
        # self.tagl = nn.Linear(self.hiddim, num_class[task[0]])
        self.bn1 = nn.BatchNorm1d(self.hiddim)
        self.relu1 = nn.ReLU()
    def scaled_dot_product(self, q, k, v):
        attention = torch.bmm(q, k.transpose(1, 2))
        attention = self.sfx(attention)
		# 添加dropout
        attention = self.drop(attention)
		# 和V做点积
        context = torch.bmm(attention, v)
        return context, attention

    def attention(self, x, mask=None, return_attention=False):
        batch_size, seq_length, embed_dim = x.shape
        qkv = self.qkv_proj(x)

        # Separate Q, K, V from linear output
        q, k, v = qkv.chunk(3, dim=-1)

        # Determine value outputs
        values, attention = self.scaled_dot_product(q, k, v)
        o = self.o_proj(values)
        return o, attention



    def forward(self, x, t):
        out = self.embed[t](x)
        out, h = self.gru(out)
        t = out.chunk(2, dim=-1)
        b = h.chunk(2, dim=0)
        # print(t[0].shape, ((b[0] + b[1]) / 2).shape)
        tago, hn = self.dgru((t[0] + t[1]) / 2, (b[0] + b[1]) / 2)
        # out, att = self.attention(out)
        h = self.bn1(torch.permute(h, (1, 2, 0)))
        intent = self.intcls(torch.mean(h, -1))
        # intent = torch.mean(intent, dim=1)
        # out = self.bn1(torch.permute(out, [0, 2, 1]))
        
        # out = self.relu1(torch.permute(out, [0, 2, 1]))
        out = self.tagl(tago)
        # print(out.shape)
        return intent, out

       