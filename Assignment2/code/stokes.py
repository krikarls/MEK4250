import numpy as np

def SOLVER(P_u,P_p,n):
	from dolfin import *
	mesh = UnitSquareMesh(n,n)

	# Define function spaces
	V = VectorFunctionSpace(mesh, "CG", P_u)
	Q = FunctionSpace(mesh, "CG", P_p)
	W = V * Q

	# Analytical solution
	u_e = Expression(('sin(pi*x[1])','cos(x[0]*pi)'))
	p_e = Expression('sin(2*pi*x[0])')
	u_E = interpolate(u_e,V)
	#p_E = interpolate(p_e,Q)
	#plot(u_E,title='Analytical u')
	#plot(p_E,title='Analytical p')

	class LeftWall(SubDomain):
		def inside(self, x, on_boundry):
			return on_boundry and x[0] < 1e-6

	class RightWall(SubDomain):
		def inside(self, x, on_boundry):
			return on_boundry and x[0] > 0.9999

	class TopWall(SubDomain):
		def inside(self, x, on_boundry):
			return on_boundry and x[1] > 0.9999

	class BottomWall(SubDomain):
		def inside(self, x, on_boundry):
			return on_boundry and x[1] < 1e-6

	mf = FacetFunction("size_t", mesh) 
	mf.set_all(0)
	left = LeftWall(); left.mark(mf, 1)
	right = RightWall(); right.mark(mf, 2)
	top = TopWall(); top.mark(mf, 3)
	bottom = BottomWall(); bottom.mark(mf, 4)
	bc1 = DirichletBC(W.sub(0), u_e, left)
	bc2 = DirichletBC(W.sub(0), u_e, right)
	bc3 = DirichletBC(W.sub(0), u_e, top)
	bc4 = DirichletBC(W.sub(0), u_e, bottom)
	bc5 = DirichletBC(W.sub(1), p_e, left)
	bc6 = DirichletBC(W.sub(1), p_e, right)
	bc7 = DirichletBC(W.sub(1), p_e, top)
	bc8 = DirichletBC(W.sub(1), p_e, bottom)

	# Define variational problem
	(u, p) = TrialFunctions(W)
	(v, q) = TestFunctions(W)

	# External force
	f =  Expression(("pi*pi*sin(pi*x[1])+2*pi*cos(2*pi*x[0])", "pi*pi*cos(pi*x[0])"))

	a = (inner(grad(u), grad(v)) - div(v)*p + q*div(u))*dx
	L = inner(f, v)*dx

	# Compute solution
	w = Function(W)
	solve(a == L, w, bcs=[bc1,bc2,bc3,bc4,bc5,bc6,bc7,bc8])

	(u, p) = w.split()

	#H1 = errornorm(u_,u_,norm_type='h1', degree_rise=3, mesh=None)
	Lu = errornorm(u_e,u,norm_type='h1', degree_rise=3, mesh=None)
	Lp = errornorm(p_e,p,norm_type='l2', degree_rise=3, mesh=None)

	#plot(u, title='Numerical u')
	#plot(p, title='Numerical p')
	#interactive()

	# Wall shear stress
	
	#n = FacetNormal(mesh)
	ds = ds[mf]
	#shear_stress = assemble(dot(grad(u)+nabla_grad(u),n)[0]*ds) 
	#shear_stress = assemble(dot(sym(grad(u)),n)[0]*ds(2)) 

	#left_shear_stress = assemble(u.dx(0)[1]*ds(1))
	#right_shear_stress = assemble(u.dx(0)[1]*ds(2))
	#top_shear_stress = assemble(u.dx(1)[0]*ds(3))
	#bottom_shear_stress = assemble(u.dx(1)[0]*ds(4))

	Es = sqrt(assemble((u.dx(1)[0]-u_E.dx(1)[0])**2*ds(4)))

	#print 'Wall shear stress on left wall: ', left_shear_stress
	#print 'Wall shear stress on right wall: ', right_shear_stress
	#print 'Wall shear stress on top wall: ', top_shear_stress
	#print 'Wall shear stress on bottom wall: ', bottom_shear_stress
	#err_wallstress = abs(left_shear_stress+right_shear_stress+top_shear_stress+bottom_shear_stress)


	return Lu, Lp, Es


N = [8,16,32,64]
h = [1./8., 1./16., 1./32., 1./64.]


E_u = np.zeros(len(N))
E_p = np.zeros(len(N))

N = [8,16,32,64]
h = [1./8., 1./16., 1./32., 1./64.]

E_u = np.zeros(len(N))
E_p = np.zeros(len(N))
E1_s = np.zeros(len(N))
E2_s = np.zeros(len(N))
E3_s = np.zeros(len(N))
E4_s = np.zeros(len(N))
r = np.zeros([3,4])

Pu = 4; Pp = 3
for i in range(0,len(N)):
	E_u[i], E_p[i], E1_s[i] = SOLVER(Pu,Pp,N[i])

Pu = 4; Pp = 2
for i in range(0,len(N)):
	E_u[i], E_p[i], E2_s[i] = SOLVER(Pu,Pp,N[i])

Pu = 3; Pp = 2
for i in range(0,len(N)):
	E_u[i], E_p[i], E3_s[i] = SOLVER(Pu,Pp,N[i])

Pu = 3; Pp = 1
for i in range(0,len(N)):
	E_u[i], E_p[i], E4_s[i] = SOLVER(Pu,Pp,N[i])

print r


import matplotlib.pyplot as plt
log_h = np.log(np.array(h))

plt.plot(log_h,np.log(E1_s),label='$P_4-P_3$')
plt.plot(log_h,np.log(E2_s),label='$P_4-P_2$')
plt.plot(log_h,np.log(E3_s),label='$P_3-P_2$')
plt.plot(log_h,np.log(E4_s),label='$P_3-P_1$')
plt.ylabel('$log(||E||_{L^2})$', fontsize=18)
plt.xlabel('$log(h)$',fontsize=18)
plt.title('log-log plot of wall shear stress error')
plt.legend(loc=4)
plt.show()

"""
from TablePrint import TablePrint

V = ['$r_1$','$r_2$','$r_3$']
H = ['1','2','3','4']
name = 'Very nice table'
TablePrint(r,V,H,name)

"""



"""
E1 = np.zeros(len(N))
E2 = np.zeros(len(N))
E3 = np.zeros(len(N))
E4 = np.zeros(len(N))
log_h = np.log(np.array(h))

Pu = 4; Pp = 3
for i in range(0,len(N)):
	E_u[i], E_p[i] = SOLVER(Pu,Pp,N[i])
	E1[i] = E_u[i]+E_p[i]

Pu = 4; Pp = 2
for i in range(0,len(N)):
	E_u[i], E_p[i] = SOLVER(Pu,Pp,N[i])
	E2[i] = E_u[i]+E_p[i]

Pu = 3; Pp = 2
for i in range(0,len(N)):
	E_u[i], E_p[i] = SOLVER(Pu,Pp,N[i])
	E3[i] = E_u[i]+E_p[i]

Pu = 3; Pp = 1
for i in range(0,len(N)):
	E_u[i], E_p[i] = SOLVER(Pu,Pp,N[i])
	E4[i] = E_u[i]+E_p[i]


import matplotlib.pyplot as plt
plt.plot(log_h,np.log(E1),label='$P_4-P_3$')
plt.plot(log_h,np.log(E2),label='$P_4-P_2$')
plt.plot(log_h,np.log(E3),label='$P_3-P_2$')
plt.plot(log_h,np.log(E4),label='$P_3-P_1$')
plt.ylabel('$log(||E_e||_{H^2}+||E_p||_{L^2})$', fontsize=18)
plt.xlabel('$log(h)$',fontsize=18)
plt.title('log-log plot of error in velocity + pressure')
plt.legend(loc=4)
plt.show()

# This part is used for computing the convergence rates for all combinations of elements 
r = np.zeros([3,8])

E_u = np.zeros(len(N))
E_p = np.zeros(len(N))

Pu = 4; Pp = 3
for i in range(0,len(N)):
	E_u[i], E_p[i] = SOLVER(Pu,Pp,N[i])

for j in range(0,len(N)-1):
	r[j,0] = ln(E_u[j+1]/E_u[j])/ln(N[j+1]**-1/N[j]**-1)
	r[j,1] = ln(E_p[j+1]/E_p[j])/ln(N[j+1]**-1/N[j]**-1)

Pu = 4; Pp = 2
for i in range(0,len(N)):
	E_u[i], E_p[i] = SOLVER(Pu,Pp,N[i])

for j in range(0,len(N)-1):
	r[j,2] = ln(E_u[j+1]/E_u[j])/ln(N[j+1]**-1/N[j]**-1)
	r[j,3] = ln(E_p[j+1]/E_p[j])/ln(N[j+1]**-1/N[j]**-1)

Pu = 3; Pp = 2
for i in range(0,len(N)):
	E_u[i], E_p[i] = SOLVER(Pu,Pp,N[i])

for j in range(0,len(N)-1):
	r[j,4] = ln(E_u[j+1]/E_u[j])/ln(N[j+1]**-1/N[j]**-1)
	r[j,5] = ln(E_p[j+1]/E_p[j])/ln(N[j+1]**-1/N[j]**-1)

Pu = 3; Pp = 1
for i in range(0,len(N)):
	E_u[i], E_p[i] = SOLVER(Pu,Pp,N[i])

for j in range(0,len(N)-1):
	r[j,6] = ln(E_u[j+1]/E_u[j])/ln(N[j+1]**-1/N[j]**-1)
	r[j,7] = ln(E_p[j+1]/E_p[j])/ln(N[j+1]**-1/N[j]**-1)

print r

from TablePrint import TablePrint

V = ['$r_1$','$r_2$','$r_3$']
H = ['1','2','3','4','5','6','7','8']
name = 'Very nice table'
TablePrint(r,V,H,name)
"""



"""
log_E = np.zeros(len(N))
log_H = np.zeros(len(N))

for i in range(0,len(N)):
	E_u[i], E_p[i], E_S[i] = SOLVER(Pu,Pp,N[i])
	log_E[i] = ln(E_u[i]+E_p[i])
	log_H[i] = ln(h[i])

import matplotlib.pyplot as plt
plt.plot(log_H, log_E)
plt.show()

#### Check error estimate, that is LHS less-or-equal to RHS ####

def u_norm(l):
	S = 0
	for i in range(0,l+2):
		S += pi**i
	return sqrt(S)

def p_norm(k):
	S = 0
	for i in range(0,k+1):
		S += 0.5*(2*pi)**(2*i)
	return sqrt(S)

for i in range(0,len(N)):
	LHS = E_u[i] + E_p[i] 
	RHS = u_norm(Pu)*h[i]**Pu + p_norm(Pp)*h[i]**(Pp+1)

	print N[i],' & ' ,'%.4e' %LHS, ' & ' , '%.4e' %RHS,'&', '$\checkmark$'  ,'\\''\\' 
	print '\\hline'
"""





