# Tensor Train for Stochastic Optimal Control #

* `machinespirit` is a library for tt-bellman with Value Iteration with Kushner discretization.

If you use this code in your research, please cite

```
TT-QI: Faster Value Iteration in Tensor Train Format for Stochastic Optimal Control, Boyko, A.I., Oseledets, I.V. & Ferrer, G. , Comput. Math. and Math. Phys. 61, 836–846 (2021). https://doi.org/10.1134/S0965542521050043
```

Bibtex:
```
@article{tt_qi,
title = {TT-QI: Faster Value Iteration in Tensor Train Format for Stochastic Optimal Control},
author = {Boyko, Alexey I. and Oseledets, Ivan V. and Ferrer, Gonzalo},
journal = {Computational Mathematics and Mathematical Physics},
volume = {61},
pages={836–846},
year ={2021}}
```


## Installation
1) Install prerequisites

```
sudo apt-get install gfortran
conda create -n ttqi python=3.8 numpy=1.21 scipy cython jupyter matplotlib
```

2) activate conda env
```
conda activate ttqi
```

3) Recusively (!!) clone ttpy and compile it
```
git clone --recursive https://github.com/oseledets/ttpy.git
cd ttpy
python setup.py install
cd ..
rm -rf ttpy
```

4) install other dependencies
```
pip install sdeint
```

## Usage
For an example of all basic functionality run Jupyter notebook ```examples/nonperiodic_pendulum.pynb```
