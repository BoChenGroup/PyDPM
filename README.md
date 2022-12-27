[contributing-image]: https://img.shields.io/badge/contributions-welcome-brightgreen.svg?style=flat
[contributing-url]: https://github.com/BoChenGroup/PyDPM/blob/master/CONTRIBUTING.md
[stars-image]: https://img.shields.io/github/stars/BoChenGroup/PyDPM
[stars-url]: https://github.com/BoChenGroup/PyDPM/stargazers

<div align=center>
<img src="https://raw.githubusercontent.com/BoChenGroup/pydpm/master/pydpm_logo_2.png" width="70%">
<br>
</br>

[![PyPI](https://img.shields.io/pypi/v/gluonts.svg?style=flat-square)](https://pypi.org/project/gluonts/)
[![GitHub](https://img.shields.io/github/license/awslabs/gluon-ts.svg?style=flat-square)](./LICENSE)
[![Stars][stars-image]][stars-url]
[![Downloads](https://pepy.tech/badge/pydpm)](https://pepy.tech/project/pydpm)
[![Contributing][contributing-image]][contributing-url]
</div>

A python library focuses on constructing Deep Probabilistic Models (DPMs).
Our developed Pydpm not only provides efficient distribution sampling functions on GPU, but also has included the implementations of existing popular DPMs.


**[Documentation](https://dustone-mu.github.io/)** | **[Paper [Arxiv]]()** | **[Tutorials](https://dustone-mu.github.io/Getting%20Started/Introduction.html)** | **[Benchmarks](https://drive.google.com/drive/folders/1_BH_0N6wfbUvTS-CCWs4YLFpDWqGRw7w?usp=sharing)** |  **[Examples](https://dustone-mu.github.io/Getting%20Started/Mini%20example.html)** |


:fire:**Note: We have released a new version that does not depend on Pycuda.**

Install
=============
The current version of PyDPM can be installed under either Windows or Linux system with PyPI. 

```
$ pip install pydpm
```

For Windows system, we recommed to install Visual Studio 2019 as the compiler equipped with CUDA 11.5 toolkit;
For Linux system, we recommed to install the latest version of CUDA toolkit.

Overview
=============
![Image text](https://raw.githubusercontent.com/BoChenGroup/pydpm/master/pydpm_framework_old.png)


Usage
=============

>Example: a few lines of code to construct a 3-layer PGBN [Link](http://mingyuanzhou.github.io/Papers/DeepPoGamma_v5.pdf).

```python
from pydpm.model import PGBN

# create the model and deploy it on gpu or cpu
model = PGBN([128, 64, 32], device='gpu')
model.initial(train_data)
train_local_params = model.train(train_data, iter_all=100)
train_local_params = model.test(train_data, iter_all=100)
test_local_params = model.test(test_data, iter_all=100)

# evaluate the model with classification accuracy
# the demo accuracy can achieve 0.8549
results = ACC(train_local_params.Theta[0], test_local_params.Theta[0], train_label, test_label, 'SVM')

# save the model after training
model.save()
```
More model demos can be found in pydpm/examples/...



Sample on GPU
=============


>Example

```python
from pydpm._sampler import Basic_Sampler

sampler = Basic_Sampler('gpu')
a = sampler.gamma(np.ones(100)*5, 1, times=10)
b = sampler.gamma(np.ones([100, 100])*5, 1, times=10)
```
More sampler demos can be found in pydpm/_sampler/...

>Compare
>
Compare the sampling speed of distribution functions with numpy:
![Image text](https://raw.githubusercontent.com/BoChenGroup/Pydpm/master/compare_numpy.png)  
The compared code can be found in pydpm/example/Sampler_Speed_Demo.py

Compare the sampling speed of distribution functions with tensorflow and torch:
![Image text](https://raw.githubusercontent.com/BoChenGroup/Pydpm/master/compare_tf2_torch.png)  
The compared code can be found in pydpm/example/Sampler_Speed_Demo.py

Contact
========
License: Apache License Version 2.0

Contact:  Chaojie Wang <xd_silly@163.com>, Wei Zhao <13279389260@163.com>, Xinyang Liu <lxy771258012@163.com>, Jiawen Wu <wjw19960807@163.com>

Copyright (c), 2020, Chaojie Wang, Wei Zhao, Xinyang Liu, Jiawen Wu, Jie Ren, Yewen Li, Hao Zhang, Bo Chen and Mingyuan Zhou
