local base_env = import 'few_shot_vqa_hotpotqa.jsonnet';

local train_batch_size = 32;
local valid_batch_size = 32;
local test_batch_size = 32;
local valid_step_size = 100;
local save_interval = 1;
local break_interval = 3000;
local train_epochs = 9999;
local adam_epsilon = 1e-08;
local lr = 1e-4;
local retriever_lr = 1e-5;
local gradient_accumulation_steps = 4;
local gradient_clipping = 0;
local warmup_steps = 0;

local seed=2021;


local override = {
  "model_config": {
    "input_modules": {
      "module_list":[
        {"type": "QInput",  "option": "frozen", "separation_tokens": {'start': '', 'end': ''}},
        {"type": "EmbeddingInput",  "option": "default"},          
      ],
      "postprocess_module_list": [
        {"type": "PostProcessClipEmbeddings", "option": "default"},
        {"type": "PostProcessInputTokenization", "option": "generation"},
      ],
    },
  },
};

std.mergePatch(base_env, override)
