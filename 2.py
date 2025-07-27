import numpy as np
import ufl
from dolfinx import geometry
from dolfinx.fem import Function, functionspace, assemble_scalar, form, Constant
from dolfinx.fem.petsc import LinearProblem
from dolfinx.io import XDMFFile
from dolfinx.mesh import create_unit_square
from ufl import dx, grad, inner
from mpi4py import MPI
from petsc4py import PETSc
import time
from ufl import dx, grad, inner, ds

# approximation space polynomial degree
deg = 2

# number of elements in each direction of msh
n_elem = 64

msh = create_unit_square(MPI.COMM_SELF, n_elem, n_elem)
n = ufl.FacetNormal(msh)

ts = time.time()
#frequency range definition
f_axis = np.arange(50, 2005, 5)

#Mic position definition
mic = np.array([0.9, 0.9, 0])


# Test and trial function space
V = functionspace(msh, ("CG", deg))
u = ufl.TrialFunction(V)
v = ufl.TestFunction(V)
f = Function(V)

# Source amplitude
Q = 0.0001

#Source definition position = (Sx,Sy)
Sx = 0.1
Sy = 0.1

# fluid definition
c0 = 340
rho_0 = 1.225
omega = Constant(msh, PETSc.ScalarType(1))
k0 = Constant(msh, PETSc.ScalarType(1))

alfa = 0.015
delta_tmp = Function(V)
delta_tmp.interpolate(lambda x : 1/(np.abs(alfa)*np.sqrt(np.pi))*np.exp(-(((x[0]-Sx)**2+(x[1]-Sy)**2)/(alfa**2))))
int_delta_tmp = assemble_scalar(form(delta_tmp*dx)) #form() l'ho scoperto per sbaglio. Senza esplode.
delta = delta_tmp/int_delta_tmp

r = Function(V)
r.interpolate(lambda x : np.sqrt((x[0]-Sx)**2 + (x[1]-Sy)**2))
Z = rho_0*c0/(1+1/(1j*k0*r))


f = 1j*rho_0*omega*Q*delta
g = 1j*rho_0*omega/Z
a = inner(grad(u), grad(v)) * dx - k0**2 * inner(u, v) * dx +  g * inner(u , v) * ds
L = inner(f, v) * dx
