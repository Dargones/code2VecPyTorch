# PyTorch implementation of code2vec

This repository contains PyTorch implementation of the [code2Vec](https://github.com/tech-srl/code2vec) model.
Please refer to the [Jupyter notebook file](https://github.com/Dargones/code2VecPyTorch/blob/main/Main.ipynb) for an 
example of how to set up and run the model.

The code was tested with python=3.8, cuDNN=7.6 and CUDA=10.1, but could be run without the GPU as well.
To install all the required libraries, please run the following:

```
conda install pytorch torchvision torchaudio cudatoolkit=10.1 -c pytorch
pip install -U catalyst
conda install plotly
```