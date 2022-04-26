python3 run_qa.py --lr=3e-5 --doc_stride=200 --batch_size=2 --num_epoch=1 --model_name=hfl/chinese-macbert-large --device=cuda --data_dir={data_dir}

Please use GPU with at least 11GB VRAM.

The data_dir should contain:

train.json, valid.json, context.json

​	