from trainers.metrics_processors import MetricsProcessor
from utils.vqa_tools import VQA
from utils.vqaEval import VQAEval
from easydict import EasyDict
import pickle 
import sys

# This script takes about 2 minutes to run, thus it is possible to be runed in the login node of HPC 

path_prefix = "../Experiments/"
path_postfix = "/test/test_evaluation/predictions.pkl"
# You can change which prediction file to evaluate here
experiment_to_evaluate = "YOUR EXPERIMENT NAME HERE"

path = sys.argv[1] if len(sys.argv) >= 2 else path_prefix + experiment_to_evaluate + path_postfix

def init_MP():
    # Initialize MetricsProcessors
    mp = MetricsProcessor()
    # Fake the VQA helpers here to avoid loading the whole VQA dataset
    mp.data_loader = EasyDict(
        {
            'data':{
                'vqa_data': {
                    'vqa_helpers': {
                        "train": VQA(
                            "../data/vqa2/v2_mscoco_train2014_annotations.json",
                            "../data/vqa2/v2_OpenEnded_mscoco_train2014_questions.json",
                        ),
                        "val": VQA(
                            "../data/vqa2/v2_mscoco_val2014_annotations.json",
                            "../data/vqa2/v2_OpenEnded_mscoco_val2014_questions.json",
                        ),
                    }
                }
            }
        }
    )
    return mp

def load_pkl(file_path):
    with open(file_path, 'rb') as f:
        load_pickle_data = pickle.load(f)
    return load_pickle_data


def compute_vqa_scores(data_dict):
    print("<--------------------------------------->")
    print("Starting standard VQA score evaluation...")
    log_dict = EasyDict(
                {
                    "metrics": {},
                    "artifacts": {},
                }
            )
    results = mp.compute_vqa_scores("dummy", data_dict, log_dict)
    print(results)
    print("Standard VQA score evaluation finished")
    print("<------------------------------------>")

if __name__ == '__main__':
    # Initialize MetricsProcessors
    mp = init_MP()
    
    # Load prediction data from the directory specified 
    data_dict = load_pkl(path)
    
    # Evaluate Standard VQA scores
    compute_vqa_scores(data_dict)
