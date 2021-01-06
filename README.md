CNGAT
=========
------a graph convolution network for radar precipitation estimation
<br>
## Environmental settings
a) anaconda & jupyter notebook : https://www.anaconda.com/products/individual
<br>
b)PyTorch : https://pytorch.org/
<br>
c) deep graph libary : https://www.dgl.ai/
<br>
## How to use

* ***Insecting the CNGAT graph convolution modular into DGL library***： Copy `snatconv.py` and `__init__.py` to the directory “Your:\path\to\your_anaconda\envs\your_pytorch_environment\Lib\site-packages\dgl\nn\pytorch\conv” .

<br>

* ***the dataset***: The dataset (toy) is provided by this url:https://pan.baidu.com/s/1A3qW99L_aEoJK0xN5JJQsQ , and the extracting code is `lck2`.

<br>

* ***Train the CNGAT model***： Just run `CNGAT_train.ipynb` on the jupyter notebook.
