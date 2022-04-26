from typing import List, Dict

from torch.utils.data import Dataset
import torch
from utils import Vocab
from utils import pad_to_len

class SeqClsDataset(Dataset):
    def __init__(
        self,
        data: List[Dict],
        vocab: Vocab,
        label_mapping: Dict[str, int],
        max_len: int,
    ):
        self.data = data
        self.vocab = vocab
        self.label_mapping = label_mapping
        self._idx2label = {idx: intent for intent, idx in self.label_mapping.items()}
        self.max_len = max_len
        self.int = 'intent'

    def __len__(self) -> int:
        return len(self.data)

    def __getitem__(self, index) -> Dict:
        instance = self.data[index]
        return instance

    @property
    def num_classes(self) -> int:
        return len(self.label_mapping)

    def collate_fn(self, samples: List[Dict]) -> Dict:
        # TODO: implement collate_fn
        texts = [e['text'].split() for e in samples]
        ids = torch.tensor([int(e['id'].split('-')[-1]) for e in samples])
        batchText = torch.tensor(self.vocab.encode_batch(batch_tokens=texts, to_len=self.max_len))
        # print(batchText)
        batchIntent = torch.empty(0)
        if(self.int in samples[0].keys()):
            batchIntent = torch.tensor([self.label_mapping[e['intent']] for e in samples])

        # print(batchIntent.shape, batchText.shape)
        return {'text': batchText, 'id': ids, 'intent': batchIntent}

    def label2idx(self, label: str):
        return self.label_mapping[label]

    def idx2label(self, idx: int):
        return self._idx2label[idx]

class SlotDataset(Dataset):
    def __init__(
        self,
        data: List[Dict],
        vocab: Vocab,
        label_mapping: Dict[str, int],
        max_len: int,
    ):
        # print(label_mapping)
        self.data = data
        self.vocab = vocab
        self.label_mapping = label_mapping
        self._idx2label = {idx: intent for intent, idx in self.label_mapping.items()}
        self.max_len = max_len
        self.int = 'tags'

    def __len__(self) -> int:
        return len(self.data)

    def __getitem__(self, index) -> Dict:
        instance = self.data[index]
        return instance

    @property
    def num_classes(self) -> int:
        return len(self.label_mapping)

    def pad_to_len(seqs: List[List[int]], to_len: int, padding: int) -> List[List[int]]:
        paddeds = [seq[:to_len] + [padding] * max(0, to_len - len(seq)) for seq in seqs]
        return paddeds

    def collate_fn(self, samples: List[Dict]) -> Dict:
        # TODO: implement collate_fn
        texts = [e['tokens'] for e in samples]
        seqlen = torch.tensor([len(e) for e in texts])
        self.max_len = torch.max(seqlen)
        ids = torch.tensor([int(e['id'].split('-')[-1]) for e in samples])
        batchText = torch.tensor(self.vocab.encode_batch(batch_tokens=texts, to_len=self.max_len))
        # print(batchText)
        batchIntent = torch.empty(0)
        if(self.int in samples[0].keys()):
            batchIntent = ([[self.label_mapping[l] for l in e[self.int]] for e in samples])
            batchIntent = pad_to_len(batchIntent, self.max_len, 0)
            batchIntent = torch.tensor(batchIntent)
        # print(batchIntent.shape, batchText.shape)
        return {'text': batchText, 'id': ids, 'slot': batchIntent, 'len': seqlen}

    def label2idx(self, label: str):
        return self.label_mapping[label]

    def idx2label(self, idx: int):
        return self._idx2label[idx]
