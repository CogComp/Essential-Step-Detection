# Installation
Setup the environment with environment.yml.

(For Windows, installing allennlp may fail, you may should download torch 1.7.0 and torchvision 0.8.1 from https://download.pytorch.org/whl/torch_stable.html and use pip to install them manually.)
# Run the code
## Generation of data

data_process.py is the code to generate the json file which is also contained in this folder, so you needn't run it. 
## Settings of preprocessings
for default settings, run with `python [baseline_name].py`

add `-c` to the end of the command for cases that use the core of steps and add `-S` for cases that use SRL model to parse the steps. 

## Settings of models

for the probing baseline and nsp baseline, add `-model [MODEL_NAME]` to use model of different sizes to compute the score. 

for the probing baseline, you can use:
`-model gpt2` `-model gpt2-medium` `-model gpt2-large` `-model gpt2-xl`

for the nsp baseline, you can use:
`-model bert-base-uncased` `-model bert-large-uncased` 

(for cased berts it may also work, though they weren't used in our experiments)
## Results
Then, you should be able to replicate our results with different scripts respectively.

# Notice
For experiments with GPT-3, Due to the double-policy policy, you need to setup an account with OPENAI and setup your own API key.
