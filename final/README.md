# ADL 2022 Final Project Scripts
## Submission
Submit your codes, outputs, and report in a zip file named `team_{team_id}.zip`,
for example, `team_1.zip`.
Your submission should be unzipped into a folder with following structure:
```
team_{team_id}/
    report.pdf
    run.sh
    requirements.txt
    simulator.py
    output.jsonl
    [any other scripts you need, such as my_model.py]
    [your trained model]
    [any other resource you need, such as concept net subgraph]
```
Where `run.sh` is your command to generate `output.jsonl`,
which is the output of `simulator.py` on the test set.
For example, the `run.sh` should be:
```
pip install -r requirements.txt
python simulator.py --model_name_or_path my_train_model/ --split test
```
and `output.jsonl` should contain 980 dialogues of (at most) 5 turns.
**PLEASE DONT MODIFY THE SEED.** We need to pickup some specific dialogues for human evaluation.

## Usage
### Install requirements
```
pip install -r requirements.txt
python3 -m spacy download en_core_web_sm
```
Note: 
It is ok if you want to use toolkits of different version.
Please provide your requirements.txt in your submission.

### Run simulator
**WARNING: ONLY USE THE TEST SET TO CALCULATE HIT RATE METRIC !!!**
**WARNING: DONT USE THE TEST SET TO TRAIN OR TUNE YOUR MODEL !!!**
You can use the simulator to:
1. Collect your own training & validation data by yourself
2. Test your model by the test set and generate output (for metric calculation)
3. Have a conversation with blendorbot by interactive mode

**WARNING: YOU WOULD NEED TO MODIFY THIS SCRIPT**
To load your model in this scripts, please 
1. write/import your model structure (replace `bot = AutoModel...`)
2. make sure it can load trained weights correctly
3. make sure your tokenizer works correctly (replace `bot_tokenizer = AutoTokenizer...`)
4. make sure your model can generate sentence correctly (around `reply_ids = bot.generate(...`)

Your efforts would be minimized if your model is a variant of Seq2SeqLM model in `transformers`.
Before you submit your codes, please make sure the simulator can run smoothly, 
which should generate output file `output.jsonl` in the same format when you directly run this script.
Write your commands in `run.sh`.

### Calculate metric (hit rate)
**WARNING: PLEASE DO NOT MODIFY `hit.py` AND `keywords.json`.**
This script will calculate keyword hit rate for you by:
```
python hit.py --prediction [/path/to/your/output/prediction/from/simulator]
```
