"""
===========================================
Sample_Demo
===========================================

"""

# Author: Jiawen Wu <wjw19960807@163.com>; Chaojie Wang <xd_silly@163.com>
# License: Apache License Version 2.0

import pydpm.distribution as dsg
import numpy.random as nr
import matplotlib.pyplot as plt
import numpy as np
times=1000000
func_name = ["exponential","gamma","chisquare","laplace","logistic",
             "lognormal","normal","poisson","weibull","gumbel",
             "dirichlet","multinomial","f","negative_binomial","zipf",
             "power","geometric","pareto"]

func_num=len(func_name)
timecost1=np.zeros(func_num)
timecost2=np.zeros(func_num)
import time
#exponential
for i in range(30):
    print(i)
    timecost=[]
    start_time=time.time()
    a=dsg.exponential(1.5,times)
    mid_time=time.time()
    b=nr.exponential(1.5,times)
    
    timecost.append([mid_time-start_time,time.time()-mid_time])

    #gamma
    start_time=time.time()
    a=dsg.gamma(1.5,1,times)
    mid_time=time.time()
    b=nr.gamma(1.5,1,times)
    
    timecost.append([mid_time-start_time,time.time()-mid_time])

    #chisquare
    start_time=time.time()
    a=dsg.chisquare(5,times)
    mid_time=time.time()
    b=nr.chisquare(5,times)
    
    timecost.append([mid_time-start_time,time.time()-mid_time])

    #laplace
    start_time=time.time()
    a=dsg.laplace(2,2,times)
    mid_time=time.time()
    b=nr.laplace(2,2,times)
    
    timecost.append([mid_time-start_time,time.time()-mid_time])

    #logistic
    start_time=time.time()
    a=dsg.logistic(2,2,times)
    mid_time=time.time()
    b=nr.logistic(2,2,times)
    
    timecost.append([mid_time-start_time,time.time()-mid_time])

    #lognormal
    start_time=time.time()
    a=dsg.lognormal(1,1,times)
    mid_time=time.time()
    b=nr.lognormal(1,1,times)
    
    timecost.append([mid_time-start_time,time.time()-mid_time])

    #normal
    start_time=time.time()
    a=dsg.normal(1,1,times)
    mid_time=time.time()
    b=nr.normal(1,1,times)
    
    timecost.append([mid_time-start_time,time.time()-mid_time])

    #poisson
    start_time=time.time()
    a=dsg.poisson(1.5,times)
    mid_time=time.time()
    b=nr.poisson(1.5,times)
    
    timecost.append([mid_time-start_time,time.time()-mid_time])

    #weibull
    start_time=time.time()
    a=dsg.weibull(2,1,times)
    mid_time=time.time()
    b=nr.weibull(2,times)
    timecost.append([mid_time-start_time,time.time()-mid_time])

    #gumbel
    start_time=time.time()
    a=dsg.gumbel(2,3,times)
    mid_time=time.time()
    b=nr.gumbel(2,3,times)
    
    timecost.append([mid_time-start_time,time.time()-mid_time])

    #dirichlet
    start_time=time.time()
    a=dsg.dirichlet([1,2,3,4,5],times)
    mid_time=time.time()
    b=nr.dirichlet([1,2,3,4,5],times)

    timecost.append([mid_time-start_time,time.time()-mid_time])

    #multinomial
    start_time=time.time()
    a=dsg.multinomial(2,[0.2,0.3,0.5],1000000)
    mid_time=time.time()
    b=nr.multinomial(2,[0.2,0.3,0.5],1000000)

    timecost.append([mid_time-start_time,time.time()-mid_time])

    #f
    start_time=time.time()
    a=dsg.f(4,5,times)
    mid_time=time.time()
    b=nr.f(4,5,times)
    
    timecost.append([mid_time-start_time,time.time()-mid_time])

    #negative_binomial
    start_time=time.time()
    a=dsg.negative_binomial(5,0.5,times)
    mid_time=time.time()
    b=nr.negative_binomial(5,0.5,times)
    
    timecost.append([mid_time-start_time,time.time()-mid_time])

    #zipf
    start_time=time.time()
    a=dsg.zipf(1.25,times)
    mid_time=time.time()
    b=nr.poisson(1.25,times)

    timecost.append([mid_time-start_time,time.time()-mid_time])

    #power
    start_time=time.time()
    a=dsg.power(1.5,times)
    mid_time=time.time()
    b=nr.power(1.5,times)
    
    timecost.append([mid_time-start_time,time.time()-mid_time])

    #geometric
    start_time=time.time()
    a=dsg.geometric(0.4,times)
    mid_time=time.time()
    b=nr.geometric(0.4,times)
    
    timecost.append([mid_time-start_time,time.time()-mid_time])

    #pareto
    start_time=time.time()
    a=dsg.pareto(1.25,times)
    mid_time=time.time()
    b=nr.pareto(1.25,times)
    
    timecost.append([mid_time-start_time,time.time()-mid_time])

    timecost=np.array(timecost)
    timecost1+=timecost[:,0]
    timecost2+=timecost[:,1]

bar_width = 0.4
index = np.arange(len(timecost))
rects1 = plt.bar(index, timecost1, bar_width, color='#0072BC', label='dsg')
rects2 = plt.bar(index + bar_width, timecost2, bar_width, color='#ED1C24', label='numpy')
plt.legend(loc='upper center', bbox_to_anchor=(0.5, -0.03), fancybox=True, ncol=5)
plt.xticks(index + bar_width/2, func_name)
plt.xticks(fontsize=9)
plt.yticks(fontsize=9)
plt.show()
