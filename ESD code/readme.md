# Installation
Setup the environment with environment.yml.

(For Windows, installing allennlp may fail, you may should download torch 1.7.0 and torchvision 0.8.1 from https://download.pytorch.org/whl/torch_stable.html manually.)
# Run the code
data_process.py is the code to generate the json file which is also contained in this folder, so you needn't run it. 

for default settings, run with `python [baseline_name].py`

add `-c` to the end of the command for cases that use the core of steps and add `-S` for cases that use SRL model to parse the steps. 

for the probing baseline, add `-model [MODEL_NAME]` to use GPT-2 of different sizes to compute the perplexity score. 

Then, you should be able to replicate our results with different scripts respectively.

# Notice
For experiments with GPT-3, Due to the double-policy policy, you need to setup an account with OPENAI and setup your own API key.
