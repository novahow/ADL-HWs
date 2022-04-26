# Sample Code for Homework 1 ADL NTU 109 Spring

## Environment
```shell
# If you have conda, we recommend you to build a conda environment called "adl"
make
# otherwise
pip install -r requirements.txt
```

## Preprocessing
```shell
# To preprocess intent detectiona and slot tagging datasets
bash preprocess.sh
```

## Intent detection && Slot tagging
```shell
python3 train_slot.py --device=cuda --scache_dir=./slot/ --icache_dir=./intent/ --v=0

the checkpoint will be saved in  
./ckpt/slot/i0.ckpt & ./ckpt/slot/s0.ckpt
```
