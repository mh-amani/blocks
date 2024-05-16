## GPT3.5 Baseline

To obtain the metrics for this baseline, we need to create the requests file to be passed to the API which will write the reply from GPT3.5 in the file we specify.

Assuming one has written the requests in `scan_requests_to_parallel_process.jsonl`, then the following command will initiate many parallel requests to the API and writes the response in `scan_requests_to_parallel_process_results.jsonl`.

`python api_request_parallel_processor.py --requests_filepath cfq_requests_to_parallel_process.jsonl --save_filepath cfq_requests_to_parallel_process_results.jsonl --request_url https://api.openai.com/v1/chat/completions --max_requests_per_minute 1000 --max_tokens_per_minute 6250000`

To create a requests file, first create the corresponding datamodule

```
%cd /to/repository_root_dir
from src import utils
import hydra
from omegaconf import DictConfig
import numpy as np
import os
import torch
from src.utils import hydra_custom_resolvers
import src.utils.general as utils
import hydra

batch_size = 32
path="'/dlabdata1/masani/blocks/logs/training/runs/scan/suponly-[0.04, 0.9]-gpt2_gpt2-vqvae/2023-11-15_10-30-01/checkpoints/last.ckpt'"
datamodule_name = "scan" # "scan", "pcfg_set", "cogs", "cfq"
with hydra.initialize(config_path=configs_path, version_base="1.2"):
    config = hydra.compose(config_name=config_name, 
                           overrides=["+experiment/inference=inference", 
                                      f"datamodule={datamodule_name}", "datamodule.dataset_parameters.supervision_ratio=[0.01,0.99]",
                                      "trainer.devices=[1]", 
                                      "training_type=suponly", 
                                      f"datamodule.dataset_parameters.batch_size={batch_size}", 
                                      "sequence_to_sequence_model_key=bart", 
                                      "discretizer_key=softmax",
                                      f"model.checkpoint_path={path}"
                                      ], return_hydra_config=True)
    
    datamodule = hydra.utils.instantiate(config.datamodule, _recursive_=False)


datamodule = run_inference(config)
```

Then prepare the requests file per each dataset

```
# We need to create the following for request to openai api
# Inputs:
# - requests_filepath : str
#     - path to the file containing the requests to be processed
#     - file should be a jsonl file, where each line is a json object with API parameters and an optional metadata field
#     - e.g., {"model": "text-embedding-3-small", "input": "embed me", "metadata": {"row_id": 1}}
#     - as with all jsonl files, take care that newlines in the content are properly escaped (json.dumps does this automatically)
#     - an example file is provided at examples/data/example_requests_to_parallel_process.jsonl
#     - the code to generate the example file is appended to the bottom of this script

# We want to create a jsonl file with the requested format containing the first 100 samples from the test set
# datamodule.data_test is the test set, and each sample is a dictionary like the following:
# {'id': 0,
#  'x': 'turn opposite right thrice and turn opposite left',
#  'z': 'I_TURN_RIGHT I_TURN_RIGHT I_TURN_RIGHT I_TURN_RIGHT I_TURN_RIGHT I_TURN_RIGHT I_TURN_LEFT I_TURN_LEFT',
#  'data_type': array([ True,  True])}

# We create the jsonl file for in-context learning, i.e., we provide 5 example pairs from the training data and then 
# give the prompt from the text and request the model to return the corresponding z. Here's an example:
# prompt for test sample 1: "x: datamodule.data_train[0]['x'], z:datamodule.data_train[0]['z'], x: datamodule.data_train[1]['x'], z:datamodule.data_train[1]['z'], x: datamodule.data_train[2]['x'], z:datamodule.data_train[2]['z'], x: datamodule.data_train[3]['x'], z:datamodule.data_train[3]['z'], x: datamodule.data_train[4]['x'], z:datamodule.data_train[4]['z'], x: datamodule.data_test[0]['x']"
# expected z for test sample 1: "I_TURN_RIGHT I_TURN_RIGHT I_TURN_RIGHT I_TURN_RIGHT I_TURN_RIGHT I_TURN_RIGHT I_TURN_LEFT I_TURN_LEFT"
# Below we create the jsonl file

import json

parallel_request_data = []
for i in range(datamodule.data_test.__len__()):
    num_examples = 20
    prompt = ""
    for j in range(num_examples):
        # pick a random sample from the training data
        idx = np.random.randint(0, datamodule.data_train.__len__())
        prompt += f"x: {datamodule.data_train[idx]['x']}, z:{datamodule.data_train[idx]['z']}, "

    # Add the test data
    prompt += f"x: {datamodule.data_test[i]['x']}, z:"
    expected_z = datamodule.data_test[i]['z']
    # parallel_request_data.append({'model': 'text-embedding-3-small', 'input': f"{prompt}", 'metadata': {'expected_z': expected_z}})
    parallel_request_data.append({'model': 'gpt-3.5-turbo', 'messages':[{'role':'user', 'content': f"{prompt}"}], 'metadata': {'expected_z': expected_z}})

# write the data to a jsonl file
requests_filepath = f"/path/to/blocks/{datamodule_name}_requests_to_parallel_process.jsonl"
with open(requests_filepath, 'w') as f:
    for item in parallel_request_data:
        f.write(json.dumps(item) + "\n")

```


Once the results are all available, we can use the following snippet to calculate the metrics for GPT-generated answers.

```
results_filepath = f"/path/to/blocks/{datamodule_name}_requests_to_parallel_process_results.jsonl"
with open(results_filepath, 'r') as f:
    results = [json.loads(line) for line in f]
    
from tokenizers import Tokenizer
path = "/path/to/some/tokenizer/for/the/dataset/logs/training/runs/cfq/curriculum-[0.01, 0.99]-bart-vqvae/2024-01-14_12-29-42/tokenizer_z.json"
tokenizer = Tokenizer.from_file(path)

manual_accuracy = {}
manual_accuracy_sentence = {}
for stage in ['test']:
    for mode in ['teacherforced', 'autoreg', 'autoreg_hidden_layer']:
        for variable in ['X', 'Z']:
            acc_name = f'{stage}/{mode}/accuracy/{variable}'
            sentence_acc_name = f'{stage}/{mode}/sentence-accuracy/{variable}'
            manual_accuracy[acc_name] = {'correct': 0, 'total': 0}
            manual_accuracy_sentence[sentence_acc_name] = {'correct': 0, 'total': 0}


# convert the contents of the 100 test samples to their ids, same with hat_ids that are stored in results
ids = []
hat_ids = []
for i in range(len(results)):
    # check the number of tokens in results[i][1]['choices'][0]['message']['content'] that are in results[i][2]['expected_z']
    # make sure to keep the order, i.e., compare first tokens, second tokens and so on, until the end of the shorter list
    if datamodule_name != "cfq":
        expected_z = results[i][2]['expected_z'].split()
        hat_z = results[i][1]['choices'][0]['message']['content'].split()
        for j in range(min(len(expected_z), len(hat_z))):
            manual_accuracy[acc_name]['correct'] += int(expected_z[j] == hat_z[j])
            manual_accuracy[acc_name]['total'] += 1

        manual_accuracy_sentence[sentence_acc_name]['correct'] += int(results[i][2]['expected_z'] == results[i][1]['choices'][0]['message']['content'][:len(results[i][2]['expected_z'])])
        manual_accuracy_sentence[sentence_acc_name]['total'] += 1
    else:
        ids = tokenizer.encode(results[i][2]['expected_z']).ids
        try:
            hat_ids = tokenizer.encode(results[i][1]['choices'][0]['message']['content']).ids
            for j in range(min(len(ids), len(hat_ids))):
                manual_accuracy[acc_name]['correct'] += int(ids[j] == hat_ids[j])
                manual_accuracy[acc_name]['total'] += 1

            manual_accuracy_sentence[sentence_acc_name]['correct'] += int(ids == hat_ids[:len(results[i][2]['expected_z'])])
            manual_accuracy_sentence[sentence_acc_name]['total'] += 1
        except:
            continue

# print the accuracy
for key, value in manual_accuracy.items():
    print(key)
    if value['total'] > 0:
        print(key, value['correct']/value['total'])
        print(key, value['correct'], value['total'])
        print("=====================================")
for key, value in manual_accuracy_sentence.items():
    if value['total'] > 0:
        print(key, value['correct']/value['total'])
        print(key, value['correct'], value['total'])
        print("=====================================")