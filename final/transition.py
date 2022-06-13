import json
import sys

import torch
from tqdm.auto import tqdm
from transformers import AutoModelForSeq2SeqLM, AutoTokenizer, AutoModelForCausalLM, BlenderbotForConditionalGeneration
from typing import Dict, List

intent_description: Dict[str, str] = {
    "Song": "Do you like to listen to music, especially rap",
    "Movie": "Do you like to watch movie in a theater",
    "Attraction": "Do you like to travel and enjoy sightseeing at famous landscape",
    "Transportation": "Do you go to somewhere else by bus, by train, or by bicycle",
    "Restaurant": "Do you like to go to a restaurant for some food or cuisine",
    "Hotel": "Do you like to go on a trip and stay in a nice hotel"
}
MODEL_NAME = "stanleychu2/t5-transition"
device = "cuda" if torch.cuda.is_available() else "cpu"
class Trans():
    def __init__(self, dev_id, ckpt=None) -> None:
        self.device = f'cuda:{dev_id}'
        checkpoint = ckpt if ckpt != None else MODEL_NAME
        self.tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
        if MODEL_NAME[-4:] == 'gpt2':
            self.tokenizer.pad_token = self.tokenizer.eos_token
        self.model = (AutoModelForSeq2SeqLM if MODEL_NAME[-4:] != 'gpt2' else AutoModelForCausalLM).from_pretrained(checkpoint).to(self.device)
        self.model.eval()

    def gen_trans(self, bot, botoken, prev, context, intent):
        prev[-1] = prev[-1].split('.')[0] + f'. {intent_description[intent]}?'
        inputs = botoken(
                    [
                        "</s> <s>".join(
                            ([context] + prev[-1:])
                        )
                    ],
                    return_tensors="pt",
                    truncation=True,
                ).to(self.device)
        reply_ids = bot.generate(**inputs)
        text = botoken.batch_decode(
                    reply_ids, skip_special_tokens=True
                )[0].strip()
        
        # prevs = ' '.join(prev)
        prevs = ' '.join(prev[:-1])
        example = (
                f"<context> {prevs} </context> <blank> <future> {text} </future>"
            )
        print('50', example, prev[-1:])
        inputs = self.tokenizer(
                example, max_length=512, truncation=True, return_tensors="pt"
            ).to(device)

        outputs = self.model.generate(
            **inputs,
            do_sample=True,
            top_k=80,
            top_p=0.95,
            max_length=64,
            repetition_penalty=0.8,
            num_return_sequences=4,
        ).squeeze(0)

        transition_sentence = [
            self.tokenizer.decode(i, skip_special_tokens=True) for i in outputs
        ]

        for e in transition_sentence[0].split('</context>'):
            if len(e.strip()) > 0:
                return e.strip()
        # return [-2]
