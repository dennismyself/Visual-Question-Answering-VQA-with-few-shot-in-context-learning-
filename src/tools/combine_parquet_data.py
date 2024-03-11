import multiprocessing
import time
import os
import subprocess
import sys
from datasets import load_dataset, Dataset, concatenate_datasets
import pandas as pd




dataset_without_embeddings = load_dataset(
    "conceptual_captions", split=f"train"
)
num_proc = 8
total_num = len(dataset_without_embeddings)
each_process_sample_num = total_num // num_proc +1
print(total_num)

out_path = "/home/ubuntu/data/projects/RAVQA/data/conceptual_captions/pre-extracted-features/conceptual_captions_ViT-L_14@336px_train.parquet"

dataset_with_embeddings = None
ps = []
for start in range(0, total_num, each_process_sample_num):
    end = min(start + each_process_sample_num, total_num)
    print(start, end)
    data_path = f'/home/ubuntu/data/projects/RAVQA/data/conceptual_captions/pre-extracted-features/conceptual_captions_ViT-L_14@336px_train[{start}:{end}].parquet'
    print(data_path)
    df = pd.read_parquet(data_path, engine='pyarrow')
    ds = Dataset.from_pandas(df)
    if not dataset_with_embeddings:
        dataset_with_embeddings = ds
    else:
        dataset_with_embeddings = concatenate_datasets([dataset_with_embeddings, ds])

print('total len', len(dataset_with_embeddings))

print(f"Writing output to {out_path}...")
dataset_with_embeddings.to_parquet(out_path)
    

