# pytorch_utils

Required installation for all applications ---

## ---- installing pytorch geometric ----

pip3 install torch-scatter -f https://data.pyg.org/whl/torch-1.10.2+cpu.html
pip3 install torch-sparse -f https://data.pyg.org/whl/torch-1.10.2+cpu.html
pip3 install torch-geometric

## --- installing tensorboard
pip3 install tensorboard

#installing the dgl library
pip3 install tensorboardX
pip3 install dgl dglgo -f https://data.dgl.ai/wheels/repo.html
pip3 install requests nltk
pip3 install umap-learn


## -- external modules may need to add for functionality --
cd /
git clone https://github.com/celiibrendan/python_tools
git clone https://github.com/celiibrendan/machine_learning_tools.git
git clone https://github.com/celiibrendan/neuron_morphology_tools.git
git clone https://github.com/celiibrendan/pytorch_tools.git

"""
Would then add to your path as follows: 

from os import sys
#sys.path.append("/meshAfterParty/meshAfterParty")
sys.path.append("/python_tools/python_tools")
sys.path.append("/machine_learning_tools/machine_learning_tools/")
sys.path.append("/pytorch_tools/pytorch_tools/")
sys.path.append("/neuron_morphology_tools/neuron_morphology_tools/")
sys.path.append("/meshAfterParty/meshAfterParty/")
"""
