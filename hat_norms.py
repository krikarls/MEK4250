
# MEK4250: Exercise 3.4 
from numpy import *
import matplotlib.pyplot as plt
import sympy as sym

def FEM_function_norms(N):

	h = 1.0/N
	print "N = ", N, " |  h = ", h

	# set up mesh points
	x = linspace(0,1,N+1)

	# set up function u(x)
	u = zeros(N+1)
	u[int(N/2)] = 1.0

	print "**** Analytical norms ****"
	import sympy as sym
	X = sym.Symbol('X')
	u1 = sym.Rational(1,h)*X - sym.Rational(1,h)*(0.5-h) 
	u2 = sym.Rational(-1,h)*X + sym.Rational(1,h)*(0.5+h) 

	L2 = sym.integrate(u1*u1,(X,0.5-h,0.5)) + sym.integrate(u2*u2,(X,0.5,0.5+h))
	print "L2: ", sqrt(float(L2))

	H1 = L2 + sym.integrate(sym.diff(u1,X)*sym.diff(u1,X),(X,0.5-h,0.5)) + sym.integrate(sym.diff(u1,X)*sym.diff(u1,X),(X,0.5,0.5+h))
	print "H1: ", sqrt(float(H1))

	print "**** Numerical norms ****"

	num_L2 = sum((u**2)/len(u))
	print  "L2: ", sqrt(num_L2) 

	dudx = (u[1:] - u[:-1])/(x[1]-x[0]) 
	num_H1 = sum(u[1:]**2 + dudx**2 )/len(u)
	print "H1: ", sqrt(num_H1) 

	return L2, H1


N_list = [10,100,1000,10000]
H = zeros(4)-3
L = zeros(4)-3

for i in range(0,4):
	L[i], H[i] = FEM_function_norms(N_list[i])


plt.subplot(2, 1, 1)
plt.plot(N_list,L)
plt.title('$L^2 -norm$ for increasing $N$')

plt.subplot(2, 1, 2)
plt.plot(N_list,H)
plt.title('$H^1 -norm$ for increasing $N$')
plt.xlabel('N')

plt.show()



