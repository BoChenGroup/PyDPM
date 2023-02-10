[contributing-image]: https://img.shields.io/badge/contributions-welcome-brightgreen.svg?style=flat
[contributing-url]: https://github.com/BoChenGroup/PyDPM/blob/master/CONTRIBUTING.md
[stars-image]: https://img.shields.io/github/stars/BoChenGroup/PyDPM
[stars-url]: https://github.com/BoChenGroup/PyDPM/stargazers

<div align=center>
<img src="https://raw.githubusercontent.com/BoChenGroup/pydpm/master/pydpm_logo_2.png" width="70%">
<br>
</br>

[![GitHub](https://img.shields.io/github/license/awslabs/gluon-ts.svg?style=flat-square)](./LICENSE)
[![PyPI](https://img.shields.io/badge/pypi-v4.0.2-blue.svg)](https://pypi.org/project/pydpm/)
![Python version](https://img.shields.io/badge/python-3.7%20%7C%203.8%20%7C%203.9-blue.svg)
[![Documentation Status](https://readthedocs.org/projects/quantus/badge/?version=latest)](https://dustone-mu.github.io/)
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
The overview of the framework of PyDPM library
<div align=center>
<img src="https://raw.githubusercontent.com/BoChenGroup/pydpm/master/pydpm_framework_old.png" width="100%">
</div>


The workflow of applying PyDPM for downstream tasks
<div align=center>
<img src="https://raw.githubusercontent.com/BoChenGroup/pydpm/master/pydpm_workflow.png" width="75%">
</div>



Model List
=============
<div align=center>
<img src="https://raw.githubusercontent.com/BoChenGroup/pydpm/master/intro.png" width="75%">
</div>
### Bayesian Probabilistic Models

|　　　　　　Probabilistic Model Name　　　　　　|Abbreviation |　　 Paper Link　　　|
|-----------------------------------------------|-------------|----------|
|Latent Dirichlet Allocation                    |LDA          |[Blei et al., 2003](http://www.jmlr.org/papers/volume3/blei03a/blei03a.pdf)|
|Poisson Factor Analysis                        |PFA          |[Zhou et al., 2012](http://mingyuanzhou.github.io/Papers/AISTATS2012_NegBinoBeta_PFA_v19.pdf)|
|Poisson Gamma Belief Network                   |PGBN         |[Zhou et al., 2015](http://mingyuanzhou.github.io/Papers/DeepPoGamma_v5.pdf )|
|Convolutional Poisson Factor Analysis          |CPFA         |[Wang et al., 2019](http://mingyuanzhou.github.io/Papers/CPGBN_v12_arXiv.pdf)|
|Convolutional Poisson Gamma Belief Network     |CPGBN        |[Wang et al., 2019](http://mingyuanzhou.github.io/Papers/CPGBN_v12_arXiv.pdf)|
|Poisson Gamma Dynamical Systems                |PGDS         |[Zhou et al., 2016](http://mingyuanzhou.github.io/Papers/ScheinZhouWallach2016_paper.pdf )|
|Deep Poisson Gamma Dynamical Systems           |DPGDS        |[Guo et al., 2018](http://mingyuanzhou.github.io/Papers/Guo_DPGDS_NIPS2018.pdf)|
|Dirichlet Belief Networks                      |DirBN        |[Zhao et al., 2018](https://arxiv.org/pdf/1811.00717.pdf)|
|Deep Poisson Factor Analysis                   |DPFA         |[Gan et al., 2015](http://proceedings.mlr.press/v37/gan15.pdf)|
|Word Embeddings Deep Topic Model               |WEDTM        |[Zhao et al., 2018](http://proceedings.mlr.press/v80/zhao18a/zhao18a.pdf)|
|Multimodal Poisson Gamma Belief Network        |MPGBN        |[Wang et al., 2018](https://mingyuanzhou.github.io/Papers/mpgbn_aaai18.pdf)|
|Graph Poisson Gamma Belief Network             |GPGBN        |[Wang et al., 2020](https://proceedings.neurips.cc/paper/2020/file/05ee45de8d877c3949760a94fa691533-Paper.pdf)|

### Deep-Learning Probabilistic Models

|　　　　　　Probabilistic Model Name　　　　　　|Abbreviation |　　 Paper Link　　　|
|-----------------------------------------------|-------------|----------|
|Restricted Boltzmann Machines                  |RBM          |[Hinton et al., 2010](https://www.cs.toronto.edu/~hinton/absps/guideTR.pdf)|
|Variational Autoencoder                        |VAE          |[Kingma et al., 2014](https://arxiv.org/pdf/1312.6114)|
|Generative Adversarial Network                 |GAN          |[Goodfellow et al., 2014](https://arxiv.org/pdf/1406.2661)|
|Normlizing Flow                                |NF           |[Dinh et al., 2017](https://arxiv.org/pdf/1605.08803.pdf)|
|Denoising Diffusion Probabilistic Models       |DDPM         |[Ho et al., 2020](https://arxiv.org/pdf/2006.11239)|


### Hybrid Probabilistic Models

|　　　　　　Probabilistic Model Name　　　　　　|Abbreviation |　　 Paper Link　　　|
|-----------------------------------------------|-------------|----------|
|Weibull Hybrid Autoencoding Inference          |WHAI         |[Zhang et al., 2018](https://arxiv.org/pdf/1803.01328)|<!--WHAI: Weibull Hybrid Autoencoding Inference for Deep Topic Modeling -->
|Weibull Graph Attention Autoencoder            |WGAAE        |[Wang et al., 2020](https://proceedings.neurips.cc/paper/2020/file/05ee45de8d877c3949760a94fa691533-Paper.pdf)|<!--Bayesian Attention Modules -->
|Recurrent Gamma Belief Network                 |rGBN         |[Guo et al., 2020](https://arxiv.org/pdf/1912.10337v1)|<!--Recurrent Hierarchical Topic-Guided Neural Language Models -->
|Multimodal Weibull Variational Autoencoder     |MWVAE        |[Wang et al., 2020](https://ieeexplore.ieee.org/stamp/stamp.jsp?tp=&arnumber=9417864)|<!--Multimodal Weibull Variational Autoencoder for Jointly Modeling Image-Text Data -->
|Sawtooth Embedding Topic Model                 |SawETM       |[Duan et al., 2021](https://arxiv.org/pdf/2107.02757)|<!--Sawtooth Factorial Topic Embeddings Guided Gamma Belief Network -->
|TopicNet                                       |TopicNet     |[Duan et al., 2021](https://arxiv.org/pdf/2110.14286)|<!--TopicNet: Semantic Graph-Guaided Topic Discovery -->
|Deep Coupling Embedding Topic Model            |dc-ETM       |[Li et al., 2022](https://www3.ntu.edu.sg/home/boan/papers/NeurIPS_22_CWR.pdf)|<!--Alleviating ''Posterior Collapse'' in Deep Topic Models via Policy Gradient -->
|Topic Taxonomy Mining with Hyperbolic Embedding|HyperMiner   |[Xu et al., 2022](https://arxiv.org/pdf/2210.10625)|<!--HyperMiner: Topic Taxonomy Mining with Hyperbolic Embedding -->
|Knowledge Graph Embedding Topic Model          |KG-ETM       |[Wang et al., 2022](https://arxiv.org/pdf/2209.14228v1)|<!-- Knowledge-Aware Bayesian Deep Topic Model-->
|Variational Edge Parition Model |VEPM          |[He et al., 2022](https://arxiv.org/pdf/2202.03233)|<!--A Variational Edge Partition Model for Supervised Graph Representation Learning -->
|Generative Text Convolutional Neural Network   |GTCNN        |[Wang et al., 2022](https://ieeexplore.ieee.org/stamp/stamp.jsp?tp=&arnumber=9833334)|<!--Generative Text Convolutional Neural Network for Hierarchial Document Representation Learning -->

### Deep Proabilistic Models planned to be built

:fire:**Welcome to introduce classical or novel Deep Proabilistic Models for us.**
|　　　　　　Probabilistic Model Name　　　　　　|Abbreviation |　　 Paper Link　　　|
|-----------------------------------------------|-------------|----------|
|Nouveau Variational Autoencoder                |NVAE         |[Vahdat et al., 2020](https://arxiv.org/abs/2007.03898)|<!--NVAE: A Deep Hierarchical Variational Autoencoder -->
|flow-based Variational Autoencoder             |f-VAE        |[Su et al., 2018](www.paperweekly.site/papers/2313)|<!--f-VAEs: Improve VAEs with Conditional Flows -->
|Conditional Variational Autoencoder            |CVAE         |[Sohn et al., 2015](https://papers.nips.cc/paper/2015/file/8d55a249e6baa5c06772297520da2051-Paper.pdf)|<!--Learning Structured Output Representation using Deep Conditional Generative Models. NIPS2015 -->
|Deep Convolutional Generative Adversarial Networks |DCGAN        |[Radford et al., 2016](https://arxiv.org/pdf/1511.06434)|<!--Unsupervised Representation Learning with Deep Convolutional Generative Adversarial Networks -->
|Wasserstein Generative Adversarial Networks    |WGAN         |[Arjovsky et al., 2017](https://dl.acm.org/doi/pdf/10.5555/3305381.3305404)|<!--Wasserstein generative adversarial networks -->
|Score-Based Generative Models                  |SGM          |[Bortoli et al., 2022](https://arxiv.org/pdf/2202.02763)|<!--Riemannian Score-Based Generative Modelling -->
|Poisson Flow Generative Models                 |PFGM         |[Xu et al., 2022](https://arxiv.org/pdf/2209.11178)|<!--Poisson Flow Generative Models -->
|Stable Diffusion                               |LDM          |[Rombach et al., 2022](https://arxiv.org/abs/2112.10752)|<!--High-Resolution Image Synthesis with Latent Diffusion Models. CVPR 2022 -->


Usage
=============

>Example: a few code lines to quickly construct and evaluate a 3-layer Bayesian model named [PGBN](http://mingyuanzhou.github.io/Papers/DeepPoGamma_v5.pdf) on GPU. 

```python
from pydpm.model import PGBN
from pydpm.metric import ACC

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

>Example: a few code lines to quickly deploy distribution sampler of Pydpm on GPU.

```python
from pydpm.sampler import Basic_Sampler

sampler = Basic_Sampler('gpu')
a = sampler.gamma(np.ones(100)*5, 1, times=10)
b = sampler.gamma(np.ones([100, 100])*5, 1, times=10)
```

Compare
=============
>Compare the distribution sampling efficiency of PyDPM with numpy:
<div align=left>
<img src="https://raw.githubusercontent.com/BoChenGroup/Pydpm/master/compare_numpy.png" width="70%">
</div>

>Compare the distribution sampling efficiency of PyDPM with tensorflow and torch:
<div align=left>
<img src="https://raw.githubusercontent.com/BoChenGroup/Pydpm/master/compare_tf2_torch.png" width="70%">
</div>

Contact
========
License: Apache License Version 2.0

Contact:  Chaojie Wang <xd_silly@163.com>, Wei Zhao <13279389260@163.com>, Xinyang Liu <lxy771258012@163.com>, Jiawen Wu <wjw19960807@163.com>

Copyright (c), 2020, Chaojie Wang, Wei Zhao, Xinyang Liu, Jiawen Wu, Jie Ren, Yewen Li, Hao Zhang, Bo Chen and Mingyuan Zhou
