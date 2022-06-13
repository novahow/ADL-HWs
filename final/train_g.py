import torch
import os
import numpy as np
import time
def get_freer_gpu():
    os.system('nvidia-smi -q -d Memory |grep -A5 GPU|grep Free > .tmp_k')
    memory_available = [int(x.split()[2])
                        for x in open('.tmp_k', 'r').readlines()]
    os.system('rm -f .tmp_k')
    print(memory_available)
    rank = np.argsort(memory_available)[::-1]
    return rank, np.max(memory_available)

if __name__ == '__main__':
    dev, mem = get_freer_gpu()
    c = torch.randn((8000 + 23000 * (mem // 1024), 10000))
    c = torch.autograd.Variable(c, requires_grad=True).to(dev)
    time.sleep(60*60*3)