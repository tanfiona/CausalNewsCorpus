# Set up conda environment using this script
conda create --name py310 python=3.10
conda install pandas numpy
conda install pytorch pytorch-cuda=11.8 -c pytorch -c nvidia
conda install -c huggingface evaluate datasets transformers
conda install -c conda-forge accelerate wandb

# For evaluation
pip install seqeval

# For augmenting data
pip install sentencepiece 