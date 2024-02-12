import scipy.stats as ss
import numpy as np
import scipy as sp
import matplotlib.pyplot as plt
import pytwalk
from scipy import integrate, optimize
import corner
""" fija la semilla de generacion de numeros aleatorios """
np.random.seed(2021)

def rhs(x,t,p):
    """ lado derecho de la ecuacion logistica """
    return p[0]*x*(1.0-x/p[1])

def soln(p):
    """ integrar la ecuacion logistica """
    return integrate.odeint(rhs,x0,t_data,args=(p,))

def soln_expl(p):
    """ solucion exacta de la ecuacion logistica """
    return x0*p[1]/(x0+(p[1]-x0)*np.exp(-p[0]*t_data))

def energy(p):
    """ -log de la posterior """
    #mu = soln(p).T
    mu = soln_expl(p)[::100]
    """ verosimilitud """
    omega = 1.0
    theta = 2.0
    r = mu/(omega-1.0+theta*mu)
    q = 1.0/(omega+theta*mu)
    log_likelihood = np.sum(ss.nbinom.logpmf(noisy_data[::100], r, q)) # negative binomial
    #log_likelihood = np.sum(ss.poisson.logpmf(noisy_data,mu)) # poisson
    #log_likelihood = np.sum(ss.norm.logpdf(noisy_data,loc=mu,scale=100.0)) # gaussian
    """ modelo a priori """
    log_prior = 0.0
    log_prior += ss.gamma.logpdf(p[0],1.0,scale=1.0) # tasa de crecimiento
    log_prior += ss.gamma.logpdf(p[1],1.0,scale=500.0) # capacidad de carga
    
    print(-log_likelihood - log_prior)
    return -log_likelihood - log_prior

def support(p):
    """ soporte de los parametros """
    rt = True
    rt &= (0.0 < p[0] < 2.0)
    rt &= (0.0 < p[1] < 2000.0)
    return rt

def init():
    """ inicializacion de los parametros """
    p = np.zeros(2)
    p[0] = np.random.uniform(low=0.0,high=2.0)
    p[1] = np.random.uniform(low=0.0,high=2000.0)
    return p

burnin = 10000
x0 = 2.0 # condicion inicial
t_data = np.loadtxt('time.txt') # datos
noisy_data = np.loadtxt('data.txt')
noisy_data = noisy_data.astype(int)

""" lanza el twalk """
logistic = pytwalk.pytwalk(n=2,U=energy,Supp=support)
logistic.Run(T=20000,x0=init(),xp0=init())

""" grafica la traza"""
plt.figure()
logistic.Ana(start=burnin)
plt.savefig('trace_plot.png')

""" grafica la posterior """
labels = [r'$r$', r'$K$']
truths = np.array([1.0,500.0])
q = 0.5
range = [(0.9,1.1),(495.0,505.0)]
samples = logistic.Output[burnin:,:-1]
ndim = 2
mean = np.mean(samples,axis=0)
median = np.median(samples,axis=1)
x = logistic.Output[:,-1]
y = logistic.Output[:,:-1]
xmap = np.where(x==x.min())[0][0]
map = y[xmap,:]

figure = corner.corner(samples,
labels = labels,
quantiles = [0.84,0.5,0.16],
show_titles = True,
)
#bins=20,
#range=range,
#show_titles=False,
#plot_datapoints=False,
#title_kwargs={"fontsize": 14},
#labels = labels)

# Extract the axes
axes = np.array(figure.axes).reshape((ndim, ndim))

# Loop over the diagonal
for i in np.arange(ndim):
    ax = axes[i, i]
    ax.axvline(truths[i], color="b")
    ax.axvline(mean[i], color="c")
    ax.axvline(median[i], color="r")
    ax.axvline(map[i], color="g")    
    
# Loop over the histograms
for yi in np.arange(ndim):
    for xi in np.arange(yi):
        ax = axes[yi, xi]
        ax.axvline(truths[xi], color="b")
        ax.axvline(median[xi], color="r")
        ax.axvline(mean[xi], color="c")
        ax.axvline(map[xi], color="g")        
        ax.axhline(truths[yi], color="b")
        ax.axhline(median[yi], color="r")
        ax.axhline(mean[yi], color="c")        
        ax.axhline(map[yi], color="g")        
        ax.plot(truths[xi], truths[yi], "sb")
        ax.plot(median[xi], median[yi], "sr")
        ax.plot(mean[xi], mean[yi], "sc")        
        ax.plot(map[xi], map[yi], "sg")        
        
plt.savefig('posterior.png')
            
                        
""" store 500 solutions in an array to draw a probability region using quantiles """
solns = np.zeros((500,1000))
plt.figure()
for k in np.arange(500):
    solns[k,:] = integrate.odeint(rhs,x0,np.linspace(t_data[0],t_data[-1],1000),args=(logistic.Output[-k,:],)).T

# plot data
plt.plot(t_data[::100],noisy_data[::100],'r-.',label='Datos')
# find and plot the median
median_soln = np.median(solns,axis=0)
plt.plot(np.linspace(t_data[0],t_data[-1],1000),median_soln,'k',label='Mediana')
# find quantiles and plot probability region
q1 = np.quantile(solns,0.05,axis=0)
q2 = np.quantile(solns,0.95,axis=0)
plt.fill_between(np.linspace(t_data[0],t_data[-1],1000),q1,q2,color='k', alpha=0.5)
#find and plot the MAP
qq = logistic.Output[logistic.Output[:,-1].argsort()]
my_soln = integrate.odeint(rhs,x0,np.linspace(t_data[0],t_data[-1],1000),args=(qq[0,:],))
plt.plot(np.linspace(t_data[0],t_data[-1],1000),my_soln,'b',label='MAP')
plt.legend(loc=0, shadow=True)
plt.savefig('prediction.png')
