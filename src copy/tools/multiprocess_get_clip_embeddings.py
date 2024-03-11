import multiprocessing
import time
import os
import subprocess
import sys
from datasets import load_dataset

class MyProcess(multiprocessing.Process):
    def __init__(self, id, start, end):
        super().__init__()
        self.id = id
        self.start = start
        self.end = end
    
    def run(self):
        subprocess.call(['CUDA_VISIBLE_DEVICES={self.id} python extract_clip_embeddings_conceptual_captions.py --split train[{start}:{end}]'], shell=True)

def run_cmd(id, start, end):
    subprocess.call([f'CUDA_VISIBLE_DEVICES={id} python extract_clip_embeddings_conceptual_captions.py --split train[{start}:{end}]'], shell=True)


dataset_without_embeddings = load_dataset(
    "conceptual_captions", split=f"train"
)
num_proc = 8
total_num = len(dataset_without_embeddings)
each_process_sample_num = total_num // num_proc +1
print(total_num)

ps = []
for start in range(0, total_num, each_process_sample_num):
    end = min(start + each_process_sample_num, total_num)
    # if end == total_num:
    #     end = ""
    # if start == 0:
    #     start = ""
    print(start, end)
    # p = MyProcess(id=len(ps), start=start, end=end)
    p = multiprocessing.Process(target=run_cmd, args=(len(ps), start, end))
    print(p)
    p.start()
    ps.append(p)

for p in ps:
    p.join()

