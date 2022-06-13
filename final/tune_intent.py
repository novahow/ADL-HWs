import json
from argparse import ArgumentParser
from operator import itemgetter
from typing import Dict, List
import random
import torch
import numpy as np
from tqdm import tqdm
from transformers import AutoModelForQuestionAnswering, AutoTokenizer, QuestionAnsweringPipeline
from os.path import join
from torch.utils.data import Dataset, DataLoader
from accelerate import Accelerator

intent_questions: Dict[str, List[str]] = {
    "Song": [
        "Is the intent asking about looking up songs ?",
        "Is the user asking about looking up songs ?",
        "Are there any websites which advertise or advertise songs for free?",
        "Is the users question about looking up songs?",
        "Is there a way to ask for help with the search of songs?",
        "How much time does someone waste searching for songs?",
        "Is the user asked about searching up song?",
        "Is a user ask about searching up songs?",
        "Does the user consider to look up songs?",
        "Is the intent asking about playing songs ?",
        "Is the user asking about playing songs ?",
        "Is the user asking about playing songs?",
        "Is your user asking about playing songs?",
        "Is the user asking about playing music?",
        "Why does the user ask about playing a song?",
        "Is a user asking about playing songs?",
        "Does my iPhone asks about playing songs?",
        "Does the user ask about playing songs?",
        "Is the user planning to playing songs ?",
        "Is the intent asking about looking up music ?",
        "Is the user asking about looking up music ?",
        "Are you asking people to look up music?",
        "Is the user asking about looking up music?",
        "Is the user asking about searching for music?",
        "Why does it seem that people are obsessed with looking up music?",
        "Is the user asking about searching music?",
        "How s/he asked about searching up music?",
        "Will the user ask about finding other music?",
        "Is it helpful when I ask for help about searching for music on a website?",
        "Is it the user asking about looking up songs (or saying songs)?",
        "Why is the user so interested in looking up music?",
        "Does the user want to look up music ?",
    ],
    "Movie": [
        "Is the intent asking about finding movies ?",
        "Is the user asking about finding movies ?",
        "Does someone want to find a movie?",
        "Does the user ask about finding movies?",
        "Why does user ask to find movies?",
        "Is the user asking about finding movies?",
        "Is the user about looking movies and trawl?",
        "Is the user asking about finding movies. Is it true that it is the same question of no different people?",
        "When did you start a game and you start asking about movies?",
        "What are the users complaints about getting movies?",
        "Does the user hope to find movies ?",
        "Is the intent asking about getting the time for movies ?",
        "Is the user asking about getting the time for movies ?",
        "What's your question about getting the time for movies?",
        "Is my mom asking about getting time for movies?",
        "How can I get the time for movies?",
        "Is the user asking about getting the time for movies?",
        "Can you fix my time problem for movies?",
        "What is the thing the user is asking about getting a time in movie or TV watching?",
        "How do you determine if you have enough time to watch movies?",
        "Is the user asking about getting time for movies?",
        "If you are a movie watcher, would you like to give you a good amount of time for your filmmaking needs?",
        "Is getting the time for movies the purpose of the user?",
    ],
    "Attraction": [
        "Is the intent asking about finding attractions ?",
        "Is the user asking about finding attractions ?",
        "Is the user asking about finding attractions?",
        "Is the user asking about how to find attractions?",
        "How can I find an attraction?",
        "What are some of the common questions asked by a visitor about how to find an attraction?",
        "Is it the user asking about finding attractions?",
        "Is the User Asking about Theme parks?",
        "Does the user have trouble finding attractions ?",
    ],
    "Restaurant": [
        "Is the user talking about food ?",
        "Is the user asking for restaurants ?",
        "Does the user have trouble deciding what to eat ?",
        "Is the user hungry ?",
        "Is the user talking about dinner ?",
        "Is the user talking about lunch ?",
        "Is the user talking about breakfast ?",
        "Is the user talking about meal ?",
    ],
    "Transportation": [
        "Is the user talking about transportation ?",
        "Is the user talking about traveling ?",
        "Does the user have trouble finding ways to commute ?",
        "Does the user need information about public transportation ?",
    ],
    "Hotel": [
        "Is the user looking for a place to stay ?",
        "Does the user want to find a hotel ?",
        "Is the user on a trip ?",
        "Is the user going abroad ?",
        "Is the user planning for a trip ?",
    ]
}

sgd_intents: Dict[str, str] = {
    f"{intent}-{q}": q
    for intent, questions in intent_questions.items()
    for q in questions
}

MODEL_NAME = "adamlin/distilbert-base-cased-sgd_qa-step5000"
REVISION = "negative_sample-questions"
def parse_args():
    parser = ArgumentParser()
    parser.add_argument("--device", type=int, default=-1)
    parser.add_argument("--input_dir", type=str, default="blender.jsonl")
    parser.add_argument("--ckpt_dir", type=str, default="./ckpt")
    parser.add_argument("--batch_size", type=int, default=16)
    parser.add_argument("--lr", type=float, default=1e-5)
    parser.add_argument("--num_epoch", type=int, default=5)
    args = parser.parse_args()
    return args

class SGD(Dataset):
    def __init__(self, init_dir=None) -> None:
        super().__init__()
        self.dir = init_dir
        self.tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME, revision=REVISION)

        self.ops = {'Transportation': ['Movie', 'Song', 'Restaurant', 'None'], 
                    'Hotel': ['Movie', 'Song', 'Restaurant', 'None'],
                    'Song': ['Restaurant', 'Transportation', 'Hotel', 'Attraction', 'None'],
                    'Movie': ['Restaurant', 'Transportation', 'Hotel', 'Attraction', 'None'],
                    'Attraction': ['Movie', 'Song', 'None'],
                    'Restaurant': ['Movie', 'Song', 'Transportation', 'Hotel', 'None']
                    }
        self.files = {}
        self.diag_len = {}
        self.class_len = [('', 0)]
        self.max_len = 32
        for k in intent_questions.keys():
            with open(join(self.dir, f"{k}_delex.json"), "r") as f:
                self.files[k] = json.load(f)
                random.shuffle(self.files[k])
                # self.diag_len[k] = sum(len(e['dialogue'] for e in self.files[k]))
                self.diag_len[k] = len(self.files[k])

            tmp_len = self.class_len[-1][-1] + self.diag_len[k]
            self.class_len.append((k, tmp_len))

        with open(join(self.dir, f"None_delex.json"), "r") as f:
            self.files['None'] = json.load(f)
    def __len__(self):
        return self.class_len[-1][-1]

    def __getitem__(self, index):
        res, idx = None, -1
        assert index < self.class_len[-1][-1]
         
        for i in range(1, len(self.class_len)):
            if self.class_len[i - 1][-1] <= index < self.class_len[i][-1]:
                res = self.class_len[i]
                idx = index - self.class_len[i - 1][-1]
                break
        assert res != None
        pos = random.random()
        int_pos = self.files[res[0]][idx]['intent_pos']
        if pos <= 0.4 or int_pos > 2:
            pos = np.random.random()
            return self.files[res[0]][idx]['dialogue'][max(int_pos - 2, 0)], 1, res[0]
        else:
            pos = random.random()
            if pos < 0.5:
                neg_pos = int_pos
                while neg_pos == int_pos:
                    neg_pos = np.random.randint(len(self.files[res[0]][idx]['dialogue']))
            
                return self.files[res[0]][idx]['dialogue'][neg_pos], 0, res[0]
            
            else:
                op_class = np.random.choice(self.ops[res[0]])
                assert len(self.files[op_class]) > 0
                op_d = np.random.choice(self.files[op_class])
                op_s = np.random.choice(op_d['dialogue']) if op_class != 'None' \
                                        else op_d['dialogue'][op_d['intent_pos']]
                return op_s, 0, res[0]

    def collate_fn(self, samples):
        # print('191', samples[0])
        qs = [np.random.choice(intent_questions[e[-1]]) for e in samples]
        context = [f'yes. no. {q[0]}' for q in samples]
        qc = []
        for q, c in zip(qs, context):
            qc.append((q, c))
        qc = self.tokenizer(qc, padding='max_length', max_length=self.max_len * 2, return_tensors='pt', truncation='only_second')
        # print('192', qc[0])
        sps = torch.tensor([e.char_to_token(0 if samples[i][1] else 5, sequence_index=1) for i, e in enumerate(qc._encodings)])
        eps = torch.tensor([e.char_to_token(3 if samples[i][1] else 7, sequence_index=1) for i, e in enumerate(qc._encodings)])
        
        return {'qc': qc, 'sp': sps, 'ep': eps}

def main(args):
    torch.cuda.set_device(args.device)
    model = AutoModelForQuestionAnswering.from_pretrained(MODEL_NAME, revision=REVISION)
    dataset = SGD(args.input_dir)
    loader = DataLoader(dataset, batch_size=args.batch_size, num_workers=2, collate_fn=dataset.collate_fn, shuffle=True)
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr)
    logging_step = 100
    accum_iter = 4
    accelerator = Accelerator(fp16=True)
    device = accelerator.device    
    # --- fp16 ---
    model, optimizer, loader = accelerator.prepare(model, optimizer, loader)
    for epoch in range(args.num_epoch):
        step = 1
        train_loss = train_acc = 0
        # --- train ---
        model.train()
        pbar = tqdm(loader)
        for data in pbar:	
            output = model(**(data['qc']) ,
                            start_positions=data['sp'], end_positions=data['ep'])
            # print('85', data, output)
            start_index = torch.argmax(output.start_logits, dim=1)
            end_index = torch.argmax(output.end_logits, dim=1)
            
            # Prediction is correct only if both start_index and end_index are correct
            train_acc += ((start_index == data['sp']) & (end_index == data['ep'])).float().mean()
            loss = output.loss / accum_iter 
            pbar.set_description(desc=f'loss:{loss:.4f}')
        # backward pass
            accelerator.backward(loss)
            train_loss += loss.detach().cpu()
        # weights update
            if ((step) % accum_iter == 0) or (step == len(loader)):
                optimizer.step()
                optimizer.zero_grad()
            step += 1

            ##### TODO: Apply linear learning rate decay #####
            
            # Print training loss and accuracy over past logging step
            if step % logging_step == 0:
                print(f"Epoch {epoch + 1} | Step {step} | loss = {train_loss.item() / logging_step:.3f}, acc = {train_acc / logging_step:.3f}")
                train_loss = train_acc = 0
            
            if step == len(loader):
                model.save_pretrained(join(args.ckpt_dir, 'qa_g.ckpt'))

if __name__ == '__main__':
    args = parse_args()
    main(args)
