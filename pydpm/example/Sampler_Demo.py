import numpy as np
import seaborn as sns
import scipy.stats as stats
from collections import Counter
import matplotlib.pyplot as plt

from pydpm._sampler import Basic_Sampler

def debug_sampler_and_plot():

    sampler = Basic_Sampler('gpu')

    # gamma
    output = sampler.gamma(np.ones(1000)*4.5, 5)
    plt.figure()
    plt.hist(output, bins=20, density=True)
    plt.plot(np.linspace(0, 100, 100), stats.gamma.pdf(np.linspace(0, 100, 100), 4.5, scale=5))
    plt.title('gamma(4.5, 5)')
    plt.show()

    # standard_gamma
    output = sampler.standard_gamma(np.ones(1000)*4.5)
    plt.figure()
    plt.hist(output, bins=20, density=True)
    plt.plot(np.linspace(0, 20, 100), stats.gamma.pdf(np.linspace(0, 20, 100), 4.5))
    plt.title('standard_gamma(4.5)')
    plt.show()

    # dirichlet
    output = sampler.dirichlet(np.ones(1000)*4.5)
    plt.figure()
    plt.hist(output, bins=20, density=True)
    # x = np.linspace(np.min(output), np.max(output), 100)
    # plt.plot(x, stats.dirichlet.pdf(x, alpha=np.ones(100)*4.5))
    plt.title('dirichlet(4.5)')
    plt.show()

    # beta
    output = sampler.beta(np.ones(1000)*0.5, 0.5)
    plt.figure()
    plt.hist(output, bins=20, density=True)
    plt.plot(np.linspace(0, 1, 100), stats.beta.pdf(np.linspace(0, 1, 100), 0.5, 0.5))
    plt.title('beta(0.5, 0.5)')
    plt.show()

    # beta(2, 5)
    output = sampler.beta(np.ones(1000)*2, 5)
    plt.figure()
    plt.hist(output, bins=20, density=True)
    plt.plot(np.linspace(0, 1, 100), stats.beta.pdf(np.linspace(0, 1, 100), 2, 5))
    plt.title('beta(2, 5)')
    plt.show()

    # normal
    output = sampler.normal(np.ones(1000)*5, np.ones(1000)*2)
    plt.figure()
    plt.hist(output, bins=20, density=True)
    plt.plot(np.linspace(-2, 13, 100), stats.norm.pdf(np.linspace(-2, 13, 100), 5, scale=2))
    plt.title('normal(5, 2)')
    plt.show()

    # standard_normal
    output = sampler.standard_normal(1000)
    plt.figure()
    plt.hist(output, bins=20, density=True)
    plt.plot(np.linspace(-3, 3, 100), stats.norm.pdf(np.linspace(-3, 3, 100)))
    plt.title('standard_normal()')
    plt.show()

    # uniform
    output = sampler.uniform(np.ones(1000)*(-2), np.ones(1000)*5)
    plt.figure()
    plt.hist(output, bins=20, density=True)
    plt.plot(np.linspace(-3, 6, 100), stats.uniform.pdf(np.linspace(-3, 6, 100), -2, 7))
    plt.title('uniform(-2, 5)')
    plt.show()

    # standard_uniform
    output = sampler.standard_uniform(1000)
    plt.figure()
    plt.hist(output, bins=20, density=True)
    plt.plot(np.linspace(-0.3, 1.3, 100), stats.uniform.pdf(np.linspace(-0.3, 1.3, 100)))
    plt.title('standard_uniform()')
    plt.show()

    # binomial
    output = sampler.binomial(np.ones(1000)*10, np.ones(1000)*0.5)
    plt.figure()
    plt.hist(output, bins=np.max(output)-np.min(output), density=True, range=(np.min(output)-0.5, np.max(output)-0.5))
    # plt.scatter(np.arange(10), stats.binom._pmf(np.arange(10), 10, 0.5), c='orange', zorder=10)
    plt.title('binomial(10, 0.5)')
    plt.show()

    # negative_binomial
    output = sampler.negative_binomial(np.ones(1000)*10, 0.5)
    plt.figure()
    plt.hist(output, bins=np.max(output)-np.min(output), density=True, range=(np.min(output)-0.5, np.max(output)-0.5))
    plt.scatter(np.arange(30), stats.nbinom._pmf(np.arange(30), 10, 0.5), c='orange', zorder=10)
    plt.title('negative_binomial(10, 0.5)')
    plt.show()

    # multinomial
    output = sampler.multinomial(5, [0.8, 0.2], 1000)
    # output = sampler.multinomial([10]*4, [[0.8, 0.2]]*4, 3)
    plt.figure()
    plt.hist(output[0], bins=15, density=True)
    plt.title('multinomial(5, [0.8, 0.2])')
    plt.show()

    a = np.array([np.array([[i] * 6 for i in range(6)]).reshape(-1), np.array(list(range(6)) * 6)]).T
    output = stats.multinomial(n=5, p=[0.8, 0.2]).pmf(a)
    sns.heatmap(output.reshape(6, 6), annot=True)
    plt.ylabel('number of the 1 kind(p=0.8)')
    plt.xlabel('number of the 2 kind(p=0.2)')
    plt.title('stats.multinomial(n=5, p=[0.8, 0.2])')
    plt.show()

    # poisson
    output = sampler.poisson(np.ones(1000)*10)
    plt.figure()
    plt.hist(output, bins=22, density=True, range=(-0.5, 21.5))
    plt.scatter(np.arange(20), stats.poisson.pmf(np.arange(20), 10), c='orange', zorder=10)
    plt.title('poisson(10)')
    plt.show()

    # cauchy
    output = sampler.cauchy(np.ones(1000)*1, 0.5)
    plt.figure()
    plt.hist(output, bins=20, density=True, range=(-5, 7))
    plt.plot(np.linspace(-5, 7, 100), stats.cauchy.pdf(np.linspace(-5, 7, 100), 1, 0.5))
    plt.title('cauchy(1, 0.5)')
    plt.show()

    # standard_cauchy
    output = sampler.standard_cauchy(1000)
    plt.figure()
    plt.hist(output, bins=20, density=True, range=(-7, 7))
    plt.plot(np.linspace(-7, 7, 100), stats.cauchy.pdf(np.linspace(-7, 7, 100)))
    plt.title('standard_cauchy()')
    plt.show()

    # chisquare
    output = sampler.chisquare(np.ones(1000)*10)
    plt.figure()
    plt.hist(output, bins=20, density=True)
    plt.plot(np.linspace(0, 30, 100), stats.chi2.pdf(np.linspace(0, 30, 100), 10))
    plt.title('chisquare(10)')
    plt.show()

    # noncentral_chisquare
    output = sampler.noncentral_chisquare(np.ones(1000)*10, 5)
    plt.figure()
    plt.hist(output, bins=20, density=True)
    # nocentral_chi2 = scale^2 * (chi2 + 2*loc*chi + df*loc^2)
    # E(Z) = nonc + df
    # Var(Z) = 2(df+2nonc)
    plt.title('noncentral_chisquare(df=10, nonc=5)')
    plt.show()

    # exponential
    lam = 0.5
    output = sampler.exponential(np.ones(1000)*lam)
    plt.figure()
    plt.hist(output, bins=20, density=True)
    plt.plot(np.linspace(0.01, 4, 100), stats.expon.pdf(np.linspace(0.01, 4, 100), scale=0.5))
    plt.title('exponential(0.5)')
    plt.show()

    # standard_exponential
    output = sampler.standard_exponential(1000)
    plt.figure()
    plt.hist(output, bins=20, density=True)
    plt.plot(np.linspace(0.01, 8, 100), stats.expon.pdf(np.linspace(0.01, 8, 100)))
    plt.title('standard_exponential()')
    plt.show()

    # f
    output = sampler.f(np.ones(1000)*10, 10)
    plt.figure()
    plt.hist(output, bins=20, density=True)
    plt.plot(np.linspace(0, 8, 100), stats.f.pdf(np.linspace(0, 8, 100), 10, 10))
    plt.title('f(10, 10)')
    plt.show()

    # noncentral_f
    output = sampler.noncentral_f(np.ones(1000)*10, 10, 5)
    plt.figure()
    plt.hist(output, bins=20, density=True)
    # E(F) = (m+nonc)*n / (m*(n-2)), n>2.
    # Var(F) = 2*(n/m)**2 * ((m+nonc)**2 + (m+2*nonc)*(n-2)) / ((n-2)**2 * (n-4))
    plt.title('noncentral_f(dfnum=10, dfden=10, nonc=5)')
    plt.show()

    # geometric
    output = sampler.geometric(np.ones(1000)*0.1)
    plt.figure()
    plt.hist(output, bins=20, density=True)
    plt.scatter(np.arange(50), stats.geom.pmf(np.arange(50), p=0.1), c='orange', zorder=10)
    plt.title('geometric(0.1)')
    plt.show()

    # gumbel
    output = sampler.gumbel(np.ones(1000)*5, np.ones(1000)*2)
    plt.figure()
    plt.hist(output, bins=20, density=True)
    plt.plot(np.linspace(0, 20, 100), stats.gumbel_r.pdf(np.linspace(0, 20, 100)+0.01, 5, scale=2))
    plt.title('gumbel(5, 2)')
    plt.show()
    np.random.gumbel()

    # hypergeometric
    output = sampler.hypergeometric(np.ones(1000)*5, 10, 10)
    plt.figure()
    plt.hist(output, bins=np.max(output)-np.min(output), density=True, range=(np.min(output)+0.5, np.max(output)+0.5))
    plt.scatter(np.arange(10), stats.hypergeom(15, 5, 10).pmf(np.arange(10)), c='orange', zorder=10)  # hypergeom(M, n, N), total, I, tiems
    plt.title('hypergeometric(5, 10, 10)')
    plt.show()

    # laplace
    output = sampler.laplace(np.ones(1000)*5, np.ones(1000)*2)
    plt.figure()
    plt.hist(output, bins=20, density=True)
    plt.plot(np.linspace(-10, 20, 100), stats.laplace.pdf(np.linspace(-10, 20, 100), 5, scale=2))
    plt.title('laplace(5, 2)')
    plt.show()

    # logistic
    output = sampler.logistic(np.ones(1000)*5, np.ones(1000)*2)
    plt.figure()
    plt.hist(output, bins=20, density=True)
    plt.plot(np.linspace(-10, 20, 100), stats.logistic.pdf(np.linspace(-10, 20, 100), 5, scale=2))
    plt.title('logistic(5, 2)')
    plt.show()

    # power
    output = sampler.power(np.ones(1000)*0.5)
    plt.figure()
    plt.hist(output, bins=20, density=True)
    plt.plot(np.linspace(0, 1.5, 100), stats.powerlaw.pdf(np.linspace(0, 1.5, 100), 0.5))
    plt.title('power(0.5)')
    plt.show()

    # zipf
    output = sampler.zipf(np.ones(1000)*1.1)
    counter = Counter(output)
    filter = np.array([[key, counter[key]] for key in counter.keys() if key < 50])
    plt.figure()
    plt.scatter(filter[:, 0], filter[:, 1] / 1000)
    plt.plot(np.arange(1, 50), stats.zipf(1.1).pmf(np.arange(1, 50)))
    plt.title('zipf(1.1)')
    plt.show()

    # pareto
    output = sampler.pareto(np.ones(1000) * 2, np.ones(1000) * 5)
    plt.figure()
    count, bins, _ = plt.hist(output, bins=50, density=True, range=(np.min(output), 100))
    a, m = 2., 5.  # shape and mode
    fit = a * m ** a / bins ** (a + 1)
    plt.plot(bins, max(count) * fit / max(fit), linewidth=2, color='r')
    plt.title('pareto(2, 5)')
    plt.show()

    # rayleigh
    output = sampler.rayleigh(np.ones(1000)*2.0)
    plt.figure()
    plt.hist(output, bins=20, density=True)
    plt.plot(np.linspace(0, 8, 100), stats.rayleigh(scale=2).pdf(np.linspace(0, 8, 100)))
    plt.title('rayleigh(2)')
    plt.show()

    # t
    output = sampler.t(np.ones(1000)*2.0)
    plt.figure()
    plt.hist(output, bins=20, density=True, range=(-6, 6))
    plt.plot(np.linspace(-6, 6, 100), stats.t(2).pdf(np.linspace(-6, 6, 100)))
    plt.title('t(2)')
    plt.show()

    # triangular
    output = sampler.triangular(np.ones(1000)*0.0, 0.3, 1)
    plt.figure()
    plt.hist(output, bins=20, density=True)
    plt.plot(np.linspace(0, 1, 100), stats.triang.pdf(np.linspace(0, 1, 100), 0.3))
    plt.title('triangular(0, 0.3, 1)')
    plt.show()

    # weibull
    output = sampler.weibull(np.ones(1000)*4.5, 5)
    plt.figure()
    plt.hist(output, bins=20, density=True)
    plt.plot(np.linspace(0, 10, 100), stats.weibull_min.pdf(np.linspace(0, 10, 100), 4.5, scale=5))
    plt.title('weibull(4.5, 5)')
    plt.show()


# -----------------test the accuracy --------------------
if __name__ == "__main__":
    plot_all_distribution_example = True  # plot all distribution samplers' example and compare with its pdf/pmf.

    if plot_all_distribution_example:
        debug_sampler_and_plot()


