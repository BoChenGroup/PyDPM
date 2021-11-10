Pydpm
======
A python package focuses on constructing deep probabilistic models on GPU.
Pydpm provides efficient distribution sampling functions and has included lots of probabilistic models.

Install
=============
Temporarily support Linux system only and the Windows version will be launched soon.

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

More probabilistic models will be included further...

>Demo

Create a PGBN model:
```
from pydpm.model import PGBN
test_model = PGBN([128, 64, 32], device='gpu')
test_model.initial(train_data)
test_model.train(100)
```
More complete demos can be found in pydpm/examples/...


>Layer construction

Construct your own probabilistic model as you wish.

```
from pydpm.layer import data_base,prob_layer,model
data = data_base('./mnist_gray')
layer1 = prob_layer(128)
layer2 = prob_layer(64)
layer3 = prob_layer(32)
pgbn = model([data, layer1, layer2, layer3],'gpu')
pgbn.train(iter=100)
```

This module is under construction and will be launched soon...

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
import pydpm.distribution as dsg
a=dsg.gamma(1.5,1,100000)
b=dsg.gamma([1.5,2.5],[1,2],100000)
```
More complete demos can be found in pydpm/distribution/...

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
