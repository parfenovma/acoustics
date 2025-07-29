import numpy as np
import ufl
from dolfinx import geometry
from dolfinx.fem import Function, functionspace, assemble_scalar, form, Constant
from dolfinx.fem.petsc import LinearProblem
from dolfinx.mesh import create_unit_square
from ufl import dx, grad, inner, ds
from mpi4py import MPI
from petsc4py import PETSc
import time
import matplotlib.pyplot as plt
import matplotlib.tri as tri
import matplotlib.animation as animation

# approximation space polynomial degree
deg = 2

# number of elements in each direction of msh
n_elem = 64

msh = create_unit_square(MPI.COMM_SELF, n_elem, n_elem)
n = ufl.FacetNormal(msh)

# Выберите частоту для визуализации
freq = 100.0  # Гц
print(f"Computing for frequency: {freq} Hz")

# Test and trial function space
V = functionspace(msh, ("CG", deg))
u = ufl.TrialFunction(V)
v = ufl.TestFunction(V)

# Source amplitude
Q = 0.0001

# Source definition position = (Sx, Sy)
Sx = 0.1
Sy = 0.1

# fluid definition
c0 = 340
rho_0 = 1.225
omega = Constant(msh, PETSc.ScalarType(1))
k0 = Constant(msh, PETSc.ScalarType(1))

alfa = 0.015
delta_tmp = Function(V)
delta_tmp.interpolate(lambda x: 1/(np.abs(alfa)*np.sqrt(np.pi)) * 
                     np.exp(-(((x[0]-Sx)**2 + (x[1]-Sy)**2)/(alfa**2))))
int_delta_tmp = assemble_scalar(form(delta_tmp*dx))
delta = delta_tmp / int_delta_tmp

r = Function(V)
r.interpolate(lambda x: np.sqrt((x[0]-Sx)**2 + (x[1]-Sy)**2) + 1e-8)
Z = rho_0 * c0 / (1 + 1/(1j * k0 * r))

# Set frequency parameters
omega.value = freq * 2 * np.pi
k0.value = 2 * np.pi * freq / c0

f = 1j * rho_0 * omega * Q * delta
g = 1j * rho_0 * omega / Z
a = inner(grad(u), grad(v)) * dx - k0**2 * inner(u, v) * dx + g * inner(u, v) * ds
L = inner(f, v) * dx

# Solve the problem
uh = Function(V)
problem = LinearProblem(a, L, u=uh, petsc_options={"ksp_type": "preonly", "pc_type": "lu"})
problem.solve()

# Get solution at mesh vertices
vertex_coords = msh.geometry.x
dof_coords = V.tabulate_dof_coordinates()

# Find DOFs that correspond to mesh vertices
dof_idx_on_vertex = []
for v in vertex_coords:
    # Find DOF with minimum distance to vertex
    distances = np.linalg.norm(dof_coords - v, axis=1)
    closest_dof = np.argmin(distances)
    dof_idx_on_vertex.append(closest_dof)
dof_idx_on_vertex = np.array(dof_idx_on_vertex)

# Get pressure values at vertices
p_complex = uh.x.array[dof_idx_on_vertex]

# Create triangulation for plotting
x = vertex_coords[:, 0]
y = vertex_coords[:, 1]
triangles = msh.topology.connectivity(msh.topology.dim, 0).array.reshape(-1, 3)
triang = tri.Triangulation(x, y, triangles)

# Create animation
num_frames = 30
phases = np.linspace(0, 2*np.pi, num_frames, endpoint=False)
abs_max = np.abs(p_complex).max()

# Создаем фигуру
fig, ax = plt.subplots(figsize=(8, 6))
ax.set_xlim(0, 1)
ax.set_ylim(0, 1)
ax.set_aspect('equal')
ax.set_xlabel('X [m]')
ax.set_ylabel('Y [m]')

# Добавляем постоянные элементы (источник и микрофон)
ax.plot(Sx, Sy, 'ro', markersize=8, label='Source')
ax.plot(0.9, 0.9, 'go', markersize=8, label='Mic')
ax.legend(loc='upper right')

# Создаем первый кадр
phase0 = phases[0]
p_real0 = np.real(p_complex * np.exp(1j * phase0))
contour = ax.tricontourf(triang, p_real0, levels=50, cmap='RdBu_r', 
                        vmin=-abs_max, vmax=abs_max)

# Добавляем цветовую шкалу
cbar = fig.colorbar(contour, ax=ax, shrink=0.8)
cbar.set_label('Pressure [Pa]')
ax.set_title(f'Pressure Distribution at {freq} Hz\nPhase: {phase0/np.pi:.2f}π rad')

# Функция обновления анимации
def update(frame):
    # Удаляем предыдущие контуры
    for coll in ax.collections:
        coll.remove()
    
    phase = phases[frame]
    p_real = np.real(p_complex * np.exp(1j * phase))
    
    # Создаем новый контур
    contour = ax.tricontourf(triang, p_real, levels=50, cmap='RdBu_r', 
                            vmin=-abs_max, vmax=abs_max)
    
    # Обновляем заголовок
    ax.set_title(f'Pressure Distribution at {freq} Hz\nPhase: {phase/np.pi:.2f}π rad')
    
    return contour

# Создаем анимацию
ani = animation.FuncAnimation(
    fig, update, frames=num_frames, interval=100, repeat=True)

# Сохраняем GIF
gif_filename = f"pressure_field_{freq}Hz.gif"
ani.save(gif_filename, writer='pillow', fps=10, dpi=100)
print(f"GIF saved as {gif_filename}")

# Также сохраняем статичный график для справки
plt.figure(figsize=(8, 6))
p_real = np.real(p_complex)
contour_static = plt.tricontourf(triang, p_real, levels=50, cmap='RdBu_r')
plt.colorbar(contour_static, label='Pressure [Pa]')
plt.plot(Sx, Sy, 'ro', markersize=8, label='Source')
plt.plot(0.9, 0.9, 'go', markersize=8, label='Mic')
plt.title(f'Pressure Distribution at {freq} Hz (Real Part)')
plt.xlabel('X [m]')
plt.ylabel('Y [m]')
plt.legend()
plt.savefig(f"pressure_static_{freq}Hz.png")
print(f"Static plot saved as pressure_static_{freq}Hz.png")
