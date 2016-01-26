from matplotlib.pylab import *
from numpy import *

N = 10000					# number of terms in fourier series
x = linspace(0,1,N)			# mesh fineness

def ck(k):						# fourier coefficients
	return (-2/(k*pi))*(cos(k*pi*(2.0/3.0))-cos(k*pi*(1.0/3.0)))

def f(x):						# function for approximation
	f_ = zeros(len(x))
	for k in range(1,N):
		f_+= ck(k)*sin(k*pi*x)
	return f_

f = f(x)						# fourier approximation

g = zeros(len(x))									# exact function
g[where( (x > (1./3) ) & (x < (2./3) )  )] = 1		# g(x)=1, for 1/3 < x < 2/3

def dudx(u): 
  du = zeros(len(u))

  for i in range(0, len(u)-2):
    du[i] = (u[i+1] - u[i]) * (len(du)-1) 

  return du

print "Num. fourier terms: ", N 

# norms
L2 = sum((g-f**2)/len(f))
print  "L2_norm: ", sqrt(L2) 

L_max = max(f)-1
print "L_max: ", L_max

Du = dudx(g-f)
H1 = sum(Du**2 + (g-f)**2 )/len(Du)
print "H1: ", sqrt(H1) 

plot(x,g,label="Exact solution")
plot(x,f,label="Fourier solution")
xlabel('x')
title('Gibbs phenomenon, terms in fourier series: {}'.format(N))
legend()
show()
