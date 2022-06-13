

from parrot import Parrot
import torch
import warnings
from transformers import AutoModelWithLMHead, AutoTokenizer
from typing import Dict, List
warnings.filterwarnings("ignore")
tokenizer = AutoTokenizer.from_pretrained("mrm8488/t5-small-finetuned-quora-for-paraphrasing")
model = AutoModelWithLMHead.from_pretrained("mrm8488/t5-small-finetuned-quora-for-paraphrasing")

def paraphrase(text, max_length=128):

  input_ids = tokenizer.encode(text, return_tensors="pt", add_special_tokens=True)

  generated_ids = model.generate(input_ids=input_ids, num_return_sequences=5, num_beams=5, max_length=max_length, no_repeat_ngram_size=2, repetition_penalty=3.5, length_penalty=1.0, early_stopping=True)

  preds = [tokenizer.decode(g, skip_special_tokens=True, clean_up_tokenization_spaces=True) for g in generated_ids]

  return preds
  
''' 
uncomment to get reproducable paraphrase generations
def random_state(seed):
  torch.manual_seed(seed)
  if torch.cuda.is_available():
    torch.cuda.manual_seed_all(seed)

random_state(1234)
'''

#Init models (make sure you init ONLY once if you integrate this to your code)
parrot = Parrot(model_tag="prithivida/parrot_paraphraser_on_T5", use_gpu=False)

intent_questions: Dict[str, List[str]] = {
    "Restaurants": [
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
    "Hotels": [
        "Is the user looking for a place to stay ?",
        "Does the user want to find a hotel ?",
        "Is the user on a trip ?",
        "Is the user going abroad ?",
        "Is the user planning for a trip ?",
    ]
}

phrases = sum(intent_questions.values(), [])

for phrase in phrases:
    print("-"*100)
    print("Input_phrase: ", phrase)
    print("-"*100)
    para_phrases = parrot.augment(input_phrase=phrase,
                               use_gpu=True,
    )
    for para_phrase in para_phrases:
        print(para_phrase)

    preds = paraphrase(f"paraphrase: {phrase}")

    for pred in preds:
        print(pred)