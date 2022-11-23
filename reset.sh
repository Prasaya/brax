conda deactivate
conda env remove -n two
conda create -n two python=3.6
conda activate two
pip install .
