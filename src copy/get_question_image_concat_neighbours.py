import pickle
from easydict import EasyDict
from pathlib import Path
from tqdm import tqdm
import pandas as pd
import faiss
import numpy as np
import random

from cribs.crib_utils import get_crib_code

data_dir = Path("../data")
vqa2_data_dir = data_dir / "vqa2"

USE_GPU = True
RERUN_INDEX = True
D_FILEPATH = 'cache/concat_nearest_neightbours_distance_20.npy'
I_FILEPATH = 'cache/concat_nearest_neighbours_index_20.npy' 
OUT_PATH = vqa2_data_dir / f"pre-extracted_features/in_context_examples/rices_concat_a1b1.pkl"
ALPHA = 1.0 # weight on text embeddings
BETA = 1.0 # weight on image embeddings
TOP_K = 20 # number of neighbours to retrieve


def load_pkl(file_path):
    with open(file_path, 'rb') as f:
        load_pickle_data = pickle.load(f)
    return EasyDict(load_pickle_data)

def dict_data_to_df(id_embed_dict, embed_field_name='text_embedding'):
    if embed_field_name == 'text_embedding':
        id_field_name = 'question_id'
    elif embed_field_name == 'image_embedding':
        id_field_name = 'question_id_prefix'
    tmp_dict = {id_field_name: [], embed_field_name: []}
    for _id, embed in id_embed_dict.items():
        tmp_dict[id_field_name].append(str(_id))
        tmp_dict[embed_field_name].append(embed)
    res_df = pd.DataFrame(tmp_dict)
    if embed_field_name == 'text_embedding':
        res_df['question_id_prefix'] = res_df['question_id'].apply(lambda x: str(x)[:-3])
    return res_df

def concat_embeddings_with_weights(df, embed_field_1, embed_field_2, alpha=1.0, beta=1.0):
    #TODO Exercise 4.2: Complete the function concat_embeddings_with_weights:
    # Args:
    #   df: A df.DataFrame object with fields for embeddings
    #   embed_field_1: field name for the first embedding
    #   embed_field_2: field name for the second embedding
    #   alpha: weighting for the first embedding
    #   beta: weighting for the second embedding
    # Returns:
    #   A numpy matrix where the embeddings from df[embed_field_1] and df[embed_field_1]  
    #   are concatanented row-wise. The embeddings should be weighted by alpha and beta, respectively
    #   i.e., The i-th row of the return matrix should be a np.array that contains the weighted concatenation of df.iloc[i][field_1] and df.iloc[i][field_2]
    # Hint:
    #   Use np.concatenate to concat the embeddings. See documentation here: https://numpy.org/doc/stable/reference/generated/numpy.concatenate.html
    #   You can change a dataframe column to an np.array by calling `df[col_name].values`. You may also find `np.stack()`, `np.squeeze()` and `DataFrame.iterrows()` useful.
    #   Use `faiss.normalize_L2(embeddings)`` to normalize vectors if you need to. What should you normalize? 
    result = None
    ##### Exercise 4.2 BEGIN ##### 
    crib_dict = {"df": df, "embed_field_1": embed_field_1, "embed_field_2": embed_field_2, "alpha": alpha, "beta": beta}
    exec(get_crib_code('4-2'), globals(), crib_dict)
    result = crib_dict.get('result', result) 
    return result
    ##### Exercise 4.2 END ##### 

def row_index_to_question_id(df, r_ind):
    return df.iloc[r_ind]['question_id']

def load_preprocessed_data(file_path):
    with open(file_path, 'rb') as f:
        load_pickle_data = pickle.load(f)["cache"]
    data_dict = {
        str(item['question_id']):item
        for item in load_pickle_data['data_items']
    }
    return data_dict 

if __name__ == '__main__':
    val_text_embeddings_df = dict_data_to_df(load_pkl(vqa2_data_dir / "pre-extracted_features/text_embeddings/coco_ViT-L_14@336px_val2014.pkl"))
    val_image_embeddings_df = dict_data_to_df(load_pkl(vqa2_data_dir / "pre-extracted_features/clip_embeddings/coco_ViT-L_14@336px_val2014.pkl"), embed_field_name='image_embedding')
    
    train_text_embeddings_df = dict_data_to_df(load_pkl(vqa2_data_dir / "pre-extracted_features/text_embeddings/coco_ViT-L_14@336px_train2014.pkl"))
    train_image_embeddings_df = dict_data_to_df(load_pkl(vqa2_data_dir / "pre-extracted_features/clip_embeddings/coco_ViT-L_14@336px_train2014.pkl"), embed_field_name='image_embedding')
    
    val_df = None
    train_df = None
    #TODO Exercise 4.1: Produce a merged dataframe that contains image and text embeddings. Do this for training and validation data
    # DataFrame merge documentation: https://pandas.pydata.org/docs/reference/api/pandas.DataFrame.merge.html 
    # Hint: consider what field should be used to merge the text and image DataFrames to set the `on` parameter of dataframe.merge(). 
    # You also need to decide the correct type of merge (left, right, cross) and specify the `how` parameter of dataframe.merge(). Note that we want to obtain concatenated embeddings for each **questions**.
    ##### Exercise 4.1 BEGIN ##### 
    crib_dict = {}
    exec(get_crib_code('4-1'), globals(), crib_dict)
    val_df = crib_dict.get('val_df', val_df)
    train_df = crib_dict.get('train_df', train_df)
    ##### Exercise 4.1 BEGIN ##### 


    val_concat_embeddings_mat = concat_embeddings_with_weights(val_df, 'text_embedding', 'image_embedding')
    train_concat_embeddings_mat = concat_embeddings_with_weights(train_df, 'text_embedding', 'image_embedding', alpha=ALPHA, beta=BETA)

    D, I = None, None
    if RERUN_INDEX:
        index = faiss.IndexFlatIP(train_concat_embeddings_mat.shape[1])
        if USE_GPU:
            res = faiss.StandardGpuResources()
            index = faiss.index_cpu_to_gpu(res, 0, index)
            print(index.is_trained)
        index.add(train_concat_embeddings_mat)

        print(f"Embeddings in the index file: {index.ntotal}")

        D, I = index.search(val_concat_embeddings_mat, TOP_K)
        with open(I_FILEPATH, 'wb') as f:
            np.save(f, I)
        with open(D_FILEPATH, 'wb') as f:
            np.save(f, D)
    else:
        with open(I_FILEPATH, 'rb') as f:
            I = np.load(f)
        with open(D_FILEPATH, 'rb') as f:
            D = np.load(f)

    val_df['nn_question_ids'] = [row_index_to_question_id(train_df, row_ind) for row_ind in tqdm(I, desc='assigning nn_question_ids')]
    val_df['nn_similarities'] = [s for s in tqdm(D, desc='unpacking similarities')]
    # load preprocessed datasets
    train_data_vqa2 = load_preprocessed_data(vqa2_data_dir / "cache/train_data_preprocessed.pkl")
    val_data_vqa2 = load_preprocessed_data(vqa2_data_dir / "cache/val_data_preprocessed.pkl")
    in_context_examples_dict = {}
    for i, row in tqdm(val_df.iterrows(), desc='Generating in_context_examples'):
        val_question_id = row['question_id']
        in_context_examples_dict[val_question_id] \
            = sorted([ 
                dict(
                    **train_data_vqa2[nn_question_id],
                    similarity=nn_sim,
                    val_question=val_data_vqa2[val_question_id]['question'],
                    val_image_key=val_data_vqa2[val_question_id]['img_key'],
                    val_image_path=val_data_vqa2[val_question_id]['img_path'],
                )
                for nn_question_id, nn_sim in zip(row['nn_question_ids'], row['nn_similarities'])
              ], key=lambda x: x['similarity'], reverse=False)

    with open(OUT_PATH, "wb") as f:
        pickle.dump(in_context_examples_dict, f)
