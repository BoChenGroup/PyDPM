"""
===========================================
Sample_Demo
===========================================

"""
# Author: Jiawen Wu <wjw19960807@163.com>; Chaojie Wang <xd_silly@163.com>
# License: Apache License Version 2.0

import pydpm.distribution as dsg
import matplotlib.pyplot as plt
import numpy as np
import tensorflow as tf
import torch
import time


times=10000000
func_name = ["gamma","normal","poisson",'laplace','exponential','beta']
timecost=np.array([[0,0,0]for i in range(6)],dtype=np.float64)

input_a = np.ones([times, 1])
input_b = np.ones([times, 1])

tf.random.gamma([times,1],1,1)
torch.distributions.Gamma(torch.tensor(input_a).cuda(), torch.tensor(input_a).cuda())
dsg.gamma(1,1,times)

for i in range(30):

    print(i)

    #gamma
    start_time=time.time()
    #dsg sample
    a=dsg.gamma(input_a,input_b,1)
    dsg_time=time.time()
    #tensorflow sample
    c=tf.compat.v1.distributions.Gamma(input_a, input_b)
    c=c.sample()
    # c=sess.run(tf1)
    tf_time=time.time()
    #pytorch sample
    temp = torch.distributions.Gamma(torch.tensor(input_a).cuda(), torch.tensor(input_b).cuda())
    d = temp.sample()
    d = d.cpu().numpy()
    torch_time=time.time()
    timecost[0]+=np.array([dsg_time-start_time,tf_time-dsg_time,torch_time-tf_time])

    #normal
    start_time=time.time()
    #dsg sample
    a=dsg.normal(input_a, input_b, 1)
    dsg_time=time.time()
    #tensorflow sample
    c=tf.compat.v1.distributions.Normal(input_a, input_b)
    c = c.sample()
    # c=sess.run(tf2)
    tf_time=time.time()
    #pytorch sample
    temp = torch.distributions.Normal(torch.tensor(input_a).cuda(), torch.tensor(input_b).cuda())
    d = temp.sample()
    d = d.cpu().numpy()
    torch_time=time.time()
    timecost[1]+=np.array([dsg_time-start_time, tf_time-dsg_time,torch_time-tf_time])

    #poisson
    start_time=time.time()
    #dsg sample
    a=dsg.poisson(input_a,1)
    dsg_time=time.time()
    #tensorflow sample
    c=tf.random.poisson([times,1],1)
    tmp = c.numpy()
    # c=sess.run(tf3)
    tf_time=time.time()
    #pytorch sample
    temp = torch.distributions.Poisson(torch.tensor(input_a*1).cuda())
    d = temp.sample()
    d=d.cpu().numpy()
    torch_time=time.time()
    timecost[2]+=np.array([dsg_time-start_time,tf_time-dsg_time,torch_time-tf_time])

    # laplace
    start_time = time.time()
    # dsg sample
    a = dsg.laplace(input_a,input_a, 1)
    dsg_time = time.time()
    # tensorflow sample
    c = tf.compat.v1.distributions.Laplace(input_a * 1.0,input_a * 1.0)
    c = c.sample()
    # c=sess.run(tf3)
    tf_time = time.time()
    # pytorch sample
    temp = torch.distributions.Laplace(torch.tensor(input_a * 1.0).cuda(),torch.tensor(input_a * 1.0).cuda())
    d = temp.sample()
    d = d.cpu().numpy()
    torch_time = time.time()
    timecost[3] += np.array([dsg_time - start_time,  tf_time - dsg_time, torch_time - tf_time])

    # exponential
    start_time = time.time()
    # dsg sample
    a = dsg.exponential(input_a*1.2, 1)
    dsg_time = time.time()
    # tensorflow sample
    c = tf.compat.v1.distributions.Exponential( input_a * 1.2)
    c = c.sample()
    # c=sess.run(tf3)
    tf_time = time.time()
    # pytorch sample
    temp = torch.distributions.Exponential(
                                       torch.tensor(input_a * 1.2).cuda())
    d = temp.sample()
    d = d.cpu().numpy()
    torch_time = time.time()
    timecost[4] += np.array([dsg_time - start_time, tf_time - dsg_time, torch_time - tf_time])

    #beta
    start_time = time.time()
    # dsg sample
    a = dsg.beta(input_a*1.0, input_a*1.0,1)
    dsg_time = time.time()
    # tensorflow sample
    c = tf.compat.v1.distributions.Beta(input_a * 1.0,input_a * 1.0)
    c = c.sample()
    # c=sess.run(tf3)
    tf_time = time.time()
    # pytorch sample
    temp = torch.distributions.Beta(torch.tensor(input_a).cuda(), torch.tensor(input_a).cuda())
    d = temp.sample()
    d = d.cpu().numpy()
    torch_time = time.time()
    timecost[5] += np.array([dsg_time - start_time, tf_time - dsg_time, torch_time - tf_time])



bar_width = 0.2

index = np.arange(len(timecost))

rects1 = plt.bar(index, np.array(timecost)[:,0], bar_width, color='#0072BC', label='dsg')

rects2 = plt.bar(index + 1*bar_width, np.array(timecost)[:,1], bar_width, color='#00FF00', label='tensorflow')

rects3 = plt.bar(index + 2*bar_width, np.array(timecost)[:,2], bar_width, color='#00FFFF', label='pytorch')

plt.legend(loc='upper center', bbox_to_anchor=(0.5, -0.03), fancybox=True, ncol=5)

plt.xticks(index + bar_width, func_name)
plt.xticks(fontsize=10)
plt.yticks(fontsize=10)
plt.show()
