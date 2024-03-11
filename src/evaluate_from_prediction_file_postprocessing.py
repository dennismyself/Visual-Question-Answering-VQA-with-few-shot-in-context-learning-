from trainers.metrics_processors import MetricsProcessor
from utils.vqa_tools import VQA
from utils.vqaEval import VQAEval
from cribs.crib_utils import get_crib_code
from easydict import EasyDict
from tqdm import tqdm
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

def postprocessing(data_dict):
    for i, data in tqdm(enumerate(data_dict['batch_predictions'])):
        # remove dot 
        
        
        #TODO Exercise 4.5: Open-Ended Investigation on postprocessing of the predictions
        # Args:
        #   data_dict: the dictionary of predictions
        # Returns:
        #  data_dict: the dictionary of predictions after postprocessing
        # Hint:
        #   Use print statements to inspect data_dict
        ##### Exercise 4.3 BEGIN ##### 
        # {'question_id': 262148000, 'answer': 'skateboarding resembling a skateboard', 'image_id': 262148, 'question_type': 'none of the a
        
        answerType = data.answer_type
        answer = data.answer
        if answerType == "yes/no":
            if "yes" in answer.lower():
                answerOut = "yes"
            else:
                answerOut = "no"
        elif answerType == "number":
            splitData = answer.split(" - ")[0]
            numbers = splitData.split(" ")
            answerOut = numbers[0].split(".")[0]
            for ele in numbers:
                ele_no_dot = ele.split(".")[0]
                if ele_no_dot.isdigit():
                    answerOut = ele_no_dot
                    break

        else:
            mainText = answer.split(" - ")[0]
            answerOut = mainText.split(".")[0]


        #data_dict['batch_predictions'][i]['answer'] = data_dict['batch_predictions'][i]['answer'].replace('.', '')
        data_dict['batch_predictions'][i]['answer'] = answerOut
        ##### Exercise 4.3 END #####
        #exec(get_crib_code('4-3'))
    return data_dict

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
    
    # Postprocessing predictions
    data_dict = postprocessing(data_dict)
    
    # Evaluate Standard VQA scores
    compute_vqa_scores(data_dict)
