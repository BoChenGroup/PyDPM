Pydpm
======
A python package focuses on constructing deep probabilistic models on GPU.
Pydpm provides efficient distribution sampling functions and has included lots of probabilistic models.

Install
=============
Temporarily support both Windows and Linux systems.

```
pip install pydpm
```
Requirements
=============
```
pycuda
scipy
numpy
```

Create Probabilistic Model
=============

>Model list
>
Model list is as following:

|Whole name                                |Model  |Paper|
|------------------------------------------|-------|-----|
|Latent Dirichlet Allocation               |LDA    |[Link](http://www.jmlr.org/papers/volume3/blei03a/blei03a.pdf)|
|Poisson Factor Analysis                   |PFA    |[Link](http://mingyuanzhou.github.io/Papers/AISTATS2012_NegBinoBeta_PFA_v19.pdf)|
|Poisson Gamma Belief Network              |PGBN   |[Link](http://mingyuanzhou.github.io/Papers/DeepPoGamma_v5.pdf )|
|Convolutional Poisson Factor Analysis     |CPFA   |[Link](http://mingyuanzhou.github.io/Papers/CPGBN_v12_arXiv.pdf)|
|Convolutional Poisson Gamma Belief Network|CPGBN  |[Link](http://mingyuanzhou.github.io/Papers/CPGBN_v12_arXiv.pdf)|
|Poisson Gamma Dynamical Systems           |PGDS   |[Link](http://mingyuanzhou.github.io/Papers/ScheinZhouWallach2016_paper.pdf )|
|Deep Poisson Gamma Dynamical Systems      |DPGDS  |[Link](http://mingyuanzhou.github.io/Papers/Guo_DPGDS_NIPS2018.pdf)|

More probabilistic models will be further included in pydpm/_model/...

>Demo

Create a PGBN model:
```
from pydpm._model import PGBN

# create the model and deploy it on gpu or cpu
model = PGBN([128, 64, 32], device='gpu')
model.initial(train_data)
train_local_params = model.train(100, train_data)
train_local_params = model.test(100, train_data)
test_local_params = model.test(100, test_data)
```
More complete demos can be found in pydpm/examples/...


Sample on GPU
=============
>Function list

The parameters of partial distribution functions are as following:

|Function        | Parameters List   | 
|----------------|-------------------|
|Normal          |mean,std,times     |
|Multinomial     |count,prob,times   |
|Poisson         |lambda,times       |
|Gamma           |shape,scale,times  |
|Beta            |alpha,beta,times   |
|F               |n1,n2,times        |
|StudentT        |n,times            |
|Dirichlet       |alpha,times        |
|Crt             |point,p,times      |
|Weibull         |shape,scale,times  |
|Chisquare       |n,times            |
|Geometric       |p,times            |
|...             |...                |

>Example

```
from pydpm._sampler import Basic_Sampler

sampler = Basic_Sampler('gpu')
a = sampler.gamma(np.ones(100)*5, 1, times=10)
b = sampler.gamma(np.ones([100, 100])*5, 1, times=10)
```
More complete demos can be found in pydpm/_sampler/...

>Compare
>
Compare the sampling speed of distribution functions with numpy:
![Image text](https://github.com/BoChenGroup/pydpm/blob/master/compare_numpy.jpg)
The compared code can be found in pydpm/example/Sample_Demo.py

Compare the sampling speed of distribution functions with tensorflow and torch:
![Image text](https://github.com/BoChenGroup/pydpm/blob/master/compare_tf2_torch.jpg)
The compared code can be found in pydpm/example/Sample_Demo2.py

Contact
========
License: Apache License Version 2.0

Contact:  Chaojie Wang <xd_silly@163.com>, Wei Zhao <13279389260@163.com>, Jiawen Wu <wjw19960807@163.com>

Copyright (c), 2020, Chaojie Wang, Wei Zhao, Jiawen Wu, Bo Chen and Mingyuan Zhou
