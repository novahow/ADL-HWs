## Environment

```shell
git clone https://github.com/huggingface/transformers.git
cd transformers
git checkout t5-fp16-no-nans
pip install -e .
```





## Training

```shell
Supervised:
python3 train.py --device=cuda --g_step=2 --batch_size=2 --lr=3e-5 --max_qlen=64 --max_plen=1024 --num_epoch=20 --data_dir={data_dir}

Add RL:
python3 rl_2017.py --device=cuda --g_step=2 --batch_size=2 --lr=1e-5 --max_qlen=64 --max_plen=1024 --num_epoch=15 --data_dir={data_dir} --qa_path={ckpt_path} --init_rew=0.055
```

GPU with VRAM > 11GB is recommended

data_dir contains ['train.jsonl', 'public.jsonl']
