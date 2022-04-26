from argparse import Namespace
from os import truncate
import pstats
# from pathy import dataclass
from transformers.tokenization_utils_base import PreTrainedTokenizerBase, PaddingStrategy
from datasets import load_dataset, load_metric
from typing import List, Dict, Optional, Union
from torch.utils.data import Dataset
import torch
from pathlib import Path
from os.path import isfile, join
import pandas as pd
import numpy as np
from itertools import chain
from multiprocessing import Pool
from multiprocessing.pool import ThreadPool
import random

class MultDs(Dataset):
    def __init__(
        self,
        tokenizer: PreTrainedTokenizerBase,
        data_dir,
        contexts: pd.DataFrame,        
        max_len: int,
        bsize: int
    ):
        self.data = pd.read_json(data_dir)
        self.questions = [e['question'] for i, e in self.data.iterrows()]
        self.contexts = contexts
        self.max_len = max_len
        self.tokenizer = tokenizer
        self.pool = ThreadPool(8)
        # self.concat = [self.getsingle(i) for i in range(len(self.data))]
        self.concat = self.pool.map(self.getsingle, range(len(self.data)))
        # self.concat = self.preprocess_function()
        self.bsize = bsize
        print('28', len(self.concat), type(self.concat[0]))
    def __len__(self) -> int:
        return len(self.data)

    def __getitem__(self, index) -> Dict:
        cands = self.data.loc[index]['paragraphs']
        label = -1 if 'relevant' not in self.data else cands.index(self.data.loc[index]['relevant'])
        instance = {'data': self.concat[index], 'label': label}
        # print('36', instance)
        return instance

    def start(self):
        self.concat = self.map(self.getsingle, range(len(self.data)))
        return self.concat
    def collate_fn(self, samples: List[Dict]) -> Dict:
        # print('37', type(samples[0]['data']))
        flattened_features = list(chain(*[e['data'] for e in samples]))
        batch = self.tokenizer.pad(
            flattened_features,
            padding=True,
            return_tensors="pt",
        )
        
        batch = {k: v.view(self.bsize, 4, -1) for k, v in batch.items()}
        # print('58', batch)
        batch['labels'] = torch.tensor([e['label'] for e in samples], dtype=torch.int64)

        return batch

    def getsingle(self, idx):
        cands = self.data.loc[idx]['paragraphs']
        conts = [self.contexts.loc[e][0] for e in cands]
        rets = [self.tokenizer(self.questions[idx], e, truncation=True) for e in conts]
        # print('62', type(rets[0]))
        return rets
    
    def label2idx(self, label: str):
        return self.label_mapping[label]

    def idx2label(self, idx: int):
        return self._idx2label[idx]

    def preprocess_function(self):
        first_sentences = [[context] * 4 for context in self.questions]
        # question_headers = examples[question_header_name]
        second_sentences = [
            [f"{self.contexts.loc[end][0]}" for end in self.data.loc[i]['paragraphs']] for i in range(len(self.data))
        ]

        # Flatten out
        first_sentences = list(chain(*first_sentences))
        second_sentences = list(chain(*second_sentences))
        # print('341', first_sentences, second_sentences)
        # Tokenize
        tokenized_examples = self.tokenizer(
            first_sentences,
            second_sentences,
            truncation=True,
        )
        # print('350', tokenized_examples)
        # Un-flatten
        return {k: [v[i : i + 4] for i in range(0, len(v), 4)] for k, v in tokenized_examples.items()}



class QAds(Dataset):
    def __init__(
        self,
        tokenizer: PreTrainedTokenizerBase,
        data_dir,
        contexts: pd.DataFrame,        
        max_len: int,
        bsize: int,
        split: str
    ):
        self.data = pd.read_json(data_dir)
        self.split = split
        self.questions = [e['question'] for i, e in self.data.iterrows()]
        # self.labels = [e['relevant'] for i, e in self.data.iterrows()]
        # self.answers = [e['answer'] for i, e in self.data.iterrows()]
        self.contexts = contexts
        self.tokenizer = tokenizer
        self.tkc = tokenizer(list(contexts[0].values), add_special_tokens=False)
        self.tkq = tokenizer(self.questions, add_special_tokens=False)
        self.bsize = bsize
        self.max_plen = 412
        self.max_qlen = 100
        self.max_len = self.max_plen + self.max_qlen
        # print('28', len(self.concat), type(self.concat[0]))
    def __len__(self) -> int:
        return len(self.data)

    def __getitem__(self, index) -> Dict:
        osp = self.data.loc[index]['answer']['start']
        oep = osp + len(self.data.loc[index]['answer']['text']) - 1
        cands = self.data.loc[index]['paragraphs']
        label = -1 if 'relevant' not in self.data else cands.index(self.data.loc[index]['relevant'])
        
        if self.split == "train":
            # Convert answer's start/end positions in paragraph_text to start/end positions in tokenized_paragraph  
            totalp = self.tkc[self.data.loc[index]['relevant']]
            totalq = self.tkq[index]
            # Paragraph is split into several windows, each with start positions separated by step "doc_stride"
            
            answer_start_token = -1 if label < 0 else totalp.char_to_token(osp) 
            answer_end_token = -1 if label < 0 else totalp.char_to_token(oep) 
            # A single window is obtained by slicing the portion of paragraph containing the answer
            mid = (answer_start_token + answer_end_token) // 2
            paragraph_start = max(0, min(answer_start_token - random.randint(1, self.max_plen - (answer_end_token - answer_start_token)),
                                        len(totalp.ids) - self.max_plen))
            paragraph_end = paragraph_start + self.max_plen
            input_ids_question = [self.tokenizer.cls_token_id] + totalq.ids[:self.max_qlen] + [self.tokenizer.sep_token_id]
            input_ids_paragraph = totalp.ids[paragraph_start: paragraph_end] + [self.tokenizer.sep_token_id]
            ttk = self.tokenizer('')
            ttk['input_ids'] = input_ids_question + input_ids_paragraph
            ttk['token_type_ids'] = [0] * len(input_ids_question) + [1] * len(input_ids_paragraph)
            ttk['attention_mask'] = [1] * len(ttk['input_ids'])
            instance = {'data': ttk, 'sp': answer_start_token + len(input_ids_question) - paragraph_start, 
                        'ep': answer_end_token + len(input_ids_question) - paragraph_start}
        
            # print('36', instance)
            return instance
        else:
            totalp = self.tkc[self.data.loc[index]['relevant']]
            totalq = self.tkq[index]
            # Paragraph is split into several windows, each with start positions separated by step "doc_stride"
            rets = []
            for i in range(0, len(totalp.ids), self.doc_stride):
                # Slice question/paragraph and add special tokens (101: CLS, 102: SEP)
                input_ids_question = [self.tokenizer.cls_token_id] + totalq.ids[:self.max_qlen] + [self.tokenizer.sep_token_id]
                input_ids_paragraph = totalp.ids[i : i + self.max_plen] + [self.tokenizer.sep_token_id]
                ttk = self.tokenizer('')
                ttk['input_ids'] = input_ids_question + input_ids_paragraph
                ttk['token_type_ids'] = [0] * len(input_ids_question) + [1] * len(input_ids_paragraph)
                ttk['attention_mask'] = [1] * len(ttk['input_ids'])
                rets.append(ttk)
            return rets

    def collate_fn(self, samples: List[Dict]) -> Dict:
        # print('184', type(samples[0]['data']))
        # flattened_features = list(chain(*[e['data'] for e in samples]))
        # print('186', samples, len(samples[0]['data']['input_ids']))
        batch = self.tokenizer.pad(
            [e['data'] for e in samples],
            padding=True,
            return_tensors="pt",
        )
        
        batch = {k: v.view(self.bsize, -1) for k, v in batch.items()}
        batch['sp'] = torch.tensor([e['sp'] for e in samples], dtype=torch.int64)
        batch['ep'] = torch.tensor([e['ep'] for e in samples], dtype=torch.int64)
        return batch

class TestDs(Dataset):
    def __init__(
        self,
        tokenizer: PreTrainedTokenizerBase,
        data_dir,
        contexts: pd.DataFrame,        
        args: Namespace,
        bsize: int,
    ):
        self.data = pd.read_json(data_dir)
        self.questions = [e['question'] for i, e in self.data.iterrows()]
        
        self.contexts = contexts
        self.tokenizer = tokenizer
        self.tkc = tokenizer(list(contexts[0].values), add_special_tokens=False)
        self.tkq = tokenizer(self.questions, add_special_tokens=False)
        self.bsize = bsize
        self.max_plen = args.max_plen
        self.max_qlen = args.max_qlen
        self.max_len = args.max_plen + args.max_qlen
        self.pool = ThreadPool(8)
        self.concat = self.pool.map(self.getsingle, range(len(self.data)))
        self.pool.close()
        print('28', len(self.concat), type(self.concat[0]))
    def __len__(self) -> int:
        return len(self.data)

    def __getitem__(self, index) -> Dict:
        cands = self.data.loc[index]['paragraphs']
        label = -1 if 'relevant' not in self.data else cands.index(self.data.loc[index]['relevant'])
        answer = None if 'answer' not in self.data else self.data.loc[index]['answer']['text']
        flattened_features = list(chain(self.concat[index]))
        batch = self.tokenizer.pad(
            flattened_features,
            padding=True,
            return_tensors="pt",
        )
        batch = {k: v.view(self.bsize, 4, -1) for k, v in batch.items()}
        instance = {'data': batch, 'label': label}
        # print('36', instance)
        # return instance
        totalp = [self.tkc[e] for e in self.data.loc[index]['paragraphs']]
        totalq = self.tkq[index]
        # Paragraph is split into several windows, each with start positions separated by step "doc_stride"
        
        rets = [self.contexts.loc[e][0] for e in cands]
        instance['pos'] = rets
        instance['question'] = self.questions[index]
        instance['id'] = self.data.loc[index]['id']
        instance['answer'] = answer
        return instance
    
    def getsingle(self, idx):
        cands = self.data.loc[idx]['paragraphs']
        # print('235', cands)
        conts = [self.contexts.loc[e][0] for e in cands]
        rets = [self.tokenizer(self.questions[idx], e, truncation=True, max_length=self.max_len) for e in conts]
        # print('62', type(rets[0]))
        return rets


class FinDs(Dataset):
    def __init__(
        self,
        tokenizer: PreTrainedTokenizerBase,
        data_dir,
        contexts: pd.DataFrame,        
        args: Namespace,
        bsize: int,
        split: str
    ):
        self.data = pd.read_json(data_dir)
        self.split = split
        self.questions = [e['question'] for i, e in self.data.iterrows()]
        self.contexts = contexts
        self.tokenizer = tokenizer
        self.tkc = tokenizer(list(contexts[0].values), add_special_tokens=False)
        self.tkq = tokenizer(self.questions, add_special_tokens=False)
        self.bsize = bsize
        self.max_plen = args.max_plen
        self.max_qlen = args.max_qlen
        self.max_len = args.max_len
        # print('28', len(self.concat), type(self.concat[0]))
    def __len__(self) -> int:
        return len(self.data)

    def __getitem__(self, index) -> Dict:
        osp = self.data.loc[index]['answer']['start']
        oep = osp + len(self.data.loc[index]['answer']['text']) - 1
        cands = self.data.loc[index]['paragraphs']
        label = -1 if 'relevant' not in self.data else cands.index(self.data.loc[index]['relevant'])
        candttk, sps, eps, rels = [], [], [], []
        for ctx in cands:
            totalp = self.tkc[ctx]
            totalq = self.tkq[index]
            # print('285', ctx, label)
            randoff = random.randint(len(totalp.ids) // 4, len(totalp.ids) // 2)
            answer_start_token = randoff if label < 0 or ctx != cands[label] \
                                else totalp.char_to_token(osp) 
            answer_end_token = min(len(totalp.ids) - 1, randoff + (oep - osp)) if label < 0 or ctx != cands[label] \
                                else totalp.char_to_token(oep) 
            mid = (answer_start_token + answer_end_token) // 2
            # print('292', answer_start_token, answer_end_token)
            paragraph_start = max(0, min(answer_start_token - random.randint(1, self.max_plen - (answer_end_token - answer_start_token)),
                                        len(totalp.ids) - self.max_plen))
            paragraph_end = paragraph_start + self.max_plen
            input_ids_question = [self.tokenizer.cls_token_id] + totalq.ids[:self.max_qlen] + [self.tokenizer.sep_token_id]
            input_ids_paragraph = totalp.ids[paragraph_start: paragraph_end] + [self.tokenizer.sep_token_id]
            ttk = self.tokenizer('')
            ttk['input_ids'] = input_ids_question + input_ids_paragraph
            ttk['token_type_ids'] = [0] * len(input_ids_question) + [1] * len(input_ids_paragraph)
            ttk['attention_mask'] = [1] * len(ttk['input_ids'])
            if len(ttk['input_ids']) > self.max_len:
                print(len(ttk['input_ids']))
                assert len(ttk['input_ids']) <= self.max_len
            candttk.append(ttk)
            sps.append(answer_start_token + len(input_ids_question) - paragraph_start)
            eps.append(answer_end_token + len(input_ids_question) - paragraph_start)
            rels.append(1 * (ctx == cands[label]))
            # print('36', instance)
            
        instance = {'data': candttk, 'sp': sps, 
                        'ep': eps, 'label': label, 'rel': rels}
            
        return instance

    def collate_fn(self, samples: List[Dict]) -> Dict:
        # print('311', type(samples[0]['data']))
        flattened_features = list(chain(*[e['data'] for e in samples]))
        # print('313', samples)
        batch = self.tokenizer.pad(
            flattened_features,
            padding=True,
            return_tensors="pt",
        )
        
        batch = {k: v.view(self.bsize * 4, -1) for k, v in batch.items()}
        batch['sp'] = torch.tensor([e['sp'] for e in samples], dtype=torch.int64).view(self.bsize * 4)
        batch['ep'] = torch.tensor([e['ep'] for e in samples], dtype=torch.int64).view(self.bsize * 4)
        batch['label'] = torch.tensor([e['label'] for e in samples], dtype=torch.int64)
        batch['rel'] = torch.tensor([e['rel'] for e in samples], dtype=torch.int64).view(self.bsize * 4)
        return batch

def unwrap_self_f(arg, **kwarg):
    return ConDs.getsingle(*arg, **kwarg)
 
     
    

class ConDs(Dataset):
    def __init__(
        self,
        tokenizer: PreTrainedTokenizerBase,
        data_dir,
        contexts: pd.DataFrame,        
        args: Namespace,
        bsize: int,
        split: str
    ):
        self.data = pd.read_json(data_dir)
        self.split = split
        self.questions = [e['question'] for i, e in self.data.iterrows()]
        self.contexts = contexts
        self.tokenizer = tokenizer
        self.tkc = tokenizer(list(contexts[0].values), add_special_tokens=False, max_length=2000, truncation=True, return_offsets_mapping=True)
        self.tkq = tokenizer(self.questions, add_special_tokens=False)
        self.bsize = bsize
        self.max_plen = args.max_plen
        self.max_qlen = args.max_qlen
        self.max_len = args.max_len
        self.args = args
        self.pool = ThreadPool(8)
        self.concat = self.pool.map(self.getsingle, range(len(self.data)))
        # self.concat = self.pool.map(self.getsingle, range(100))
        print('357', len(self.concat))
        self.pool.close()
        self.lens = [0] * (len(self.concat) + 1)
        for i in range(len(self.concat)):
            self.lens[i + 1] = self.lens[i] + self.concat[i]['len'].iloc[0]
        self.concat = pd.concat(self.concat)
        self.concat = self.concat.drop('len', axis=1)
        print('359', len(self.concat), len(self.data), self.concat.iloc[0]['data']['input_ids'])
        # self.concat = []
    def __len__(self) -> int:
        return len(self.concat) if self.split == 'train' else len(self.lens) - 1

    def getsingle(self, index):
        osp = self.data.loc[index]['answer']['start'] if 'answer' in self.data else None
        oep = osp + len(self.data.loc[index]['answer']['text']) - 1  if 'answer' in self.data else None
        cands = self.data.loc[index]['paragraphs']
        label = -1 if ('relevant' not in self.data) or self.split != 'train' else self.data.loc[index]['relevant']
        candttk, sps, eps, rels, pids, oms, pst = [], [], [], [], [], [], []
        df = pd.DataFrame()
        for ctx in cands:
            # print('380', ctx, label)
            totalp = self.tkc[ctx]
            totalq = self.tkq[index]
            if ctx == label:
                answer_start_token = totalp.char_to_token(osp) 
                answer_end_token = totalp.char_to_token(oep)
                
                for i in range(5): 
                    paragraph_start = max(0, min(answer_start_token - random.randint(1, self.max_plen - (answer_end_token - answer_start_token + 1)),
                                                    len(totalp.ids) - self.max_plen))
                    paragraph_end = paragraph_start + self.max_plen
                    input_ids_question = [self.tokenizer.cls_token_id] + totalq.ids[:self.max_qlen] + [self.tokenizer.sep_token_id]
                    input_ids_paragraph = totalp.ids[paragraph_start: paragraph_end] + [self.tokenizer.sep_token_id]
                    ttk = self.tokenizer('')
                    ttk['input_ids'], ttk['token_type_ids'], ttk['attention_mask'], _ = self.padding(input_ids_question, input_ids_paragraph)
                    if len(ttk['input_ids']) > self.max_len:
                        print(len(ttk['input_ids']))
                        assert len(ttk['input_ids']) <= self.max_len
                    candttk.append(ttk)
                    sps.append(answer_start_token + len(input_ids_question) - paragraph_start)
                    eps.append(answer_end_token + len(input_ids_question) - paragraph_start)
                    rels.append(1)
                
                for i in range(0, len(totalp), self.args.doc_stride):
                    sp, ep = i, i + self.max_plen
                    nei = 0
                    if sp <= answer_end_token < ep and sp <= answer_start_token < ep:
                        continue
                    
                    if ep >= answer_start_token and ep < answer_end_token:
                        ep = answer_start_token
                        nei = 1
                    if sp >= answer_start_token and sp <= answer_end_token:
                        sp = answer_end_token + 1
                        nei = 1
                    input_ids_question = [self.tokenizer.cls_token_id] + totalq.ids[:self.max_qlen] + [self.tokenizer.sep_token_id]
                    input_ids_paragraph = totalp.ids[sp: ep] + [self.tokenizer.sep_token_id]
                    ttk = self.tokenizer('')
                    ttk['input_ids'], ttk['token_type_ids'], ttk['attention_mask'], _ = self.padding(input_ids_question, input_ids_paragraph)
                    if len(ttk['input_ids']) > self.max_len:
                        print(len(ttk['input_ids']))
                        assert len(ttk['input_ids']) <= self.max_len
                    candttk.append(ttk)
                    sps.append(((len(input_ids_paragraph) - 1)) + len(input_ids_question))
                    eps.append(((len(input_ids_paragraph) - 1)) + len(input_ids_question))
                    rels.append(0)
                
            else:
                for i in range(0, len(totalp), self.args.doc_stride):
                    if i / self.args.doc_stride > 3 and self.split == 'train':
                        break
                    paragraph_start = i
                    paragraph_end = paragraph_start + self.max_plen
                    input_ids_question = [self.tokenizer.cls_token_id] + totalq.ids[:self.max_qlen] + [self.tokenizer.sep_token_id]
                    input_ids_paragraph = totalp.ids[paragraph_start: paragraph_end] + [self.tokenizer.sep_token_id]
                    ttk = self.tokenizer('')
                    ttk['input_ids'], ttk['token_type_ids'], ttk['attention_mask'], ttk['offset_mapping'] = \
                            self.padding(input_ids_question, input_ids_paragraph, totalp.offsets[paragraph_start: paragraph_end])
                    if len(ttk['input_ids']) > self.max_len:
                        print(len(ttk['input_ids']))
                        assert len(ttk['input_ids']) <= self.max_len
                    candttk.append(ttk)
                    sps.append(len(input_ids_paragraph) - 1 + len(input_ids_question))
                    eps.append(len(input_ids_paragraph) - 1 + len(input_ids_question))
                    rels.append(0)
                    pids.append(ctx)
        instance = {'data': candttk, 'sp': sps, 
                        'ep': eps, 'label': label, 'rel': rels,
                            'pid': pids}
        # print('431', instance)
        for k, v in instance.items():
            df[k] = pd.Series(v)
        
        df['len'] = len(instance['sp'])
        # print('432', df.iloc[0]['data']['input_ids'])
        return df

    
    def __getitem__(self, index) -> Dict:
        if self.split == 'train':
            return self.concat.iloc[index]
        else:
            s, e = self.lens[index], self.lens[index + 1]
            # print('460', type(qcs), type(qcs['data']), self.concat.iloc[0]['data']['input_ids'])
            batch = {}
            batch['input_ids'] = torch.tensor([self.concat.iloc[i]['data']['input_ids'] for i in range(s, e)])
            batch['token_type_ids'] = torch.tensor([self.concat.iloc[i]['data']['token_type_ids'] for i in range(s, e)])
            batch['attention_mask'] = torch.tensor([self.concat.iloc[i]['data']['attention_mask'] for i in range(s, e)])
            batch['sp'] = torch.tensor(self.concat['sp'].iloc[s: e], dtype=torch.int64)
            batch['ep'] = torch.tensor(self.concat['ep'].iloc[s: e], dtype=torch.int64)
            batch['rel'] = torch.tensor(self.concat['rel'].iloc[s: e], dtype=torch.int64)
            batch['offset'] = torch.tensor([self.concat.iloc[i]['data']['offset_mapping'] for i in range(s, e)])
            batch['pid'] = torch.tensor(self.concat['pid'].iloc[s: e], dtype=torch.int64)
            return batch

    def collate_fn(self, samples: List[Dict]) -> Dict:
        batch = {}
        batch['input_ids'] = torch.tensor([e['data']['input_ids'] for e in samples])
        batch['token_type_ids'] = torch.tensor([e['data']['token_type_ids'] for e in samples])
        batch['attention_mask'] = torch.tensor([e['data']['attention_mask'] for e in samples])
        batch['sp'] = torch.tensor([e['sp'] for e in samples], dtype=torch.int64)
        batch['ep'] = torch.tensor([e['ep'] for e in samples], dtype=torch.int64)
        # batch['label'] = torch.tensor([e['label'] for e in samples], dtype=torch.int64)
        batch['rel'] = torch.tensor([e['rel'] for e in samples], dtype=torch.int64)
        return batch
    def padding(self, input_ids_question, input_ids_paragraph, om=None):
        # Pad zeros if sequence length is shorter than max_seq_len
        padding_len = self.max_len - len(input_ids_question) - len(input_ids_paragraph)
        # Indices of input sequence tokens in the vocabulary
        input_ids = input_ids_question + input_ids_paragraph + [0] * padding_len
        # Segment token indices to indicate first and second portions of the inputs. Indices are selected in [0, 1]
        token_type_ids = [0] * len(input_ids_question) + [1] * len(input_ids_paragraph) + [0] * padding_len
        # Mask to avoid performing attention on padding token indices. Mask values selected in [0, 1]
        attention_mask = [1] * (len(input_ids_question) + len(input_ids_paragraph)) + [0] * padding_len
        om = (om + [(0, 0)] * padding_len) if om != None else om
        return input_ids, token_type_ids, attention_mask, om
    
    def run(self):
        pool = Pool(processes=8)
        pool.map(unwrap_self_f, zip([self]*len(self.data), self.data))