from dolfin import *
import numpy as np
import matplotlib.pyplot as plt

def SOLVER(lmbda,N):
	mesh = UnitSquareMesh(N,N)

	# Define function spaces
	V = VectorFunctionSpace(mesh, "CG", 2)

	# Analytical solution
	u_e = Expression(('pi*x[0]*cos(x[0]*x[1]*pi)','-pi*x[1]*cos(x[0]*x[1]*pi)'))
	u_E = interpolate(u_e,V)
	
	#file = File('analytical_displacement.pvd')
	#file << u_E
	#plot(u_E,range_min=0.0,range_max=2.0, mode='displacement',title='Analytical')

	class Bndry(SubDomain):
		def inside(self, x, on_boundry):
			return on_boundry

	bnd = Bndry()
	mf = FacetFunction("size_t", mesh); mf.set_all(0)
	bnd.mark(mf, 1)
	bcs = DirichletBC(V, u_e, bnd)

	# Define variational problem
	u = TrialFunction(V)
	v = TestFunction(V)

	# External force
	mu = Constant(1.0) 
	lmbda = Constant(lmbda)
	
	f = Expression(("pi*pi*(pi*x[0]*(x[0]*x[0]+x[1]*x[1])*cos(pi*x[0]*x[1])+2*x[1]*sin(pi*x[0]*x[1]))",\
	       " -pi*pi*(pi*x[1]*(x[0]*x[0]+x[1]*x[1])*cos(pi*x[0]*x[1])+2*x[0]*sin(pi*x[0]*x[1]))"))

	a = inner(grad(u), grad(v))*dx + lmbda*inner(div(u),div(v))*dx 
	L = inner(f,v)*dx  

	# Compute solution
	u_ = Function(V)
	solve(a == L,u_, bcs)

	#file = File('numerical_displacement.pvd')
	#file << u_
	
	#H1 = errornorm(analytical,u_,norm_type='h1', degree_rise=3, mesh=None)
	L2 = errornorm(u_e,u_,norm_type='l2', degree_rise=3, mesh=None)

	return L2



import numpy as np
lmbda_list = [1,10,100,1000,10000]
N_list = [8,16,32,64,128,256]
H_list = [1/8.,1/16.,1/32.,1/64.,1/128.,1./256]

L2_list = np.zeros([5,6]) 

for i in range(0,5):
	for j in range(0,6):
		L2_list[i,j] = SOLVER(lmbda_list[i] ,N_list[j])


log_E = np.log(L2_list)
log_h = np.log(H_list)

plt.plot(log_h,log_E[0,:],label='$\lambda = 1$')
plt.plot(log_h,log_E[1,:],label='$\lambda = 10$')
plt.plot(log_h,log_E[2,:],label='$\lambda = 100$')
plt.plot(log_h,log_E[3,:],label='$\lambda = 1000$')
plt.plot(log_h,log_E[4,:],label='$\lambda = 10000$')
plt.ylabel('$log(||E||_{L^2})$', fontsize=18)
plt.xlabel('$log(h)$',fontsize=18)
plt.title('log-log plot of error for second order polynomials')
plt.legend(loc=4)
plt.show()





print '----------- Numerical L2-errors -----------'
print ''
print 'Data-format:'
print 'L = lambda'
print 'L\N   | 8  |  16 |  32 |  64 |' 
print '1     | *  |  *  |  *  |  *  |' 
print '10    | *  |  *  |  *  |  *  |' 
print '100   | *  |  *  |  *  |  *  |' 
print '1000  | *  |  *  |  *  |  *  |' 
print '10000 | *  |  *  |  *  |  *  |' 

print ''
print 'L2-errors:'
print L2_list

E = L2_list
R = np.zeros([5,5])

for i in range(0,5):
	for j in range(0,5):
		R[i,j] = ln(E[i,j+1]/E[i,j])/ln(N_list[j+1]**-1/N_list[j]**-1)


print ''
print '----------- Convergence rate -----------'
print ''
print 'Data-format:'
print 'L = lambda'
print 'L\N   | 8  |  16 |  32 |  64 |' 
print '1     | *  |  *  |  *  |  *  |' 
print '10    | *  |  *  |  *  |  *  |' 
print '100   | *  |  *  |  *  |  *  |' 
print '1000  | *  |  *  |  *  |  *  |' 
print '10000 | *  |  *  |  *  |  *  |' 

print R



"""
from TablePrint import *

caption1 = 'L2-errors'
TablePrint(E,lmbda_list,N_list,caption1)

H = ['$r_1$','$r_2$','$r_3$']
caption2 = 'Convergence-rates'
TablePrint(R,lmbda_list,H,caption2)
"""