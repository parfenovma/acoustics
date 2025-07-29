import numpy as np
import ufl
from dolfinx import geometry, fem
from dolfinx.fem import Function, functionspace, assemble_scalar, form, Constant
from dolfinx.fem.petsc import LinearProblem
from dolfinx.mesh import create_unit_square, locate_entities_boundary, meshtags
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
freq = 800.0  # Гц
print(f"Computing for frequency: {freq} Hz")

# Test and trial function space
V = functionspace(msh, ("CG", deg))
u = ufl.TrialFunction(V)
v = ufl.TestFunction(V)

# Source amplitude
Q = 0.0001

# Source definition position = (Sx, Sy)
Sx = 0.1
Sy = 0.5

# fluid definition
c0 = 340
rho_0 = 1.225
omega = Constant(msh, PETSc.ScalarType(1))
k0 = Constant(msh, PETSc.ScalarType(1))

# Определяем границы:
# 1 - нижняя (y=0)
# 2 - правая (x=1)
# 3 - верхняя (y=1)
# 4 - левая (x=0)

# Жесткие границы (условие Неймана) на верхней и нижней границах
bottom_facets = locate_entities_boundary(msh, msh.topology.dim-1, lambda x: np.isclose(x[1], 0.0))
top_facets = locate_entities_boundary(msh, msh.topology.dim-1, lambda x: np.isclose(x[1], 1.0))

# Импедансные границы на левой и правой границах
left_facets = locate_entities_boundary(msh, msh.topology.dim-1, lambda x: np.isclose(x[0], 0.0))
right_facets = locate_entities_boundary(msh, msh.topology.dim-1, lambda x: np.isclose(x[0], 1.0))

# Создаем мезотеги для границ
facets = np.hstack([bottom_facets, top_facets, left_facets, right_facets])
markers = np.hstack([
    np.full_like(bottom_facets, 1),
    np.full_like(top_facets, 3),
    np.full_like(left_facets, 4),
    np.full_like(right_facets, 2)
])
sorted_facets = np.argsort(facets)
facet_tag = meshtags(msh, msh.topology.dim-1, facets[sorted_facets], markers[sorted_facets])

# Создаем мезу ds для разных типов границ
ds_impedance = ds(subdomain_data=facet_tag)((2,4))  # правая (2) и левая (4) границы
ds_neumann = ds(subdomain_data=facet_tag)((1,3))    # нижняя (1) и верхняя (3) границы

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

# Формулировка задачи с разными граничными условиями
# Жесткие границы (условие Неймана) не добавляют вклада в билинейную форму
a = inner(grad(u), grad(v)) * dx - k0**2 * inner(u, v) * dx + g * inner(u, v) * ds_impedance
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
ax.set_title(f'Pressure Distribution at {freq} Hz (Hard Boundaries)')

# Добавляем постоянные элементы (источник и микрофон)
ax.plot(Sx, Sy, 'ro', markersize=8, label='Source')
ax.plot(0.9, 0.9, 'go', markersize=8, label='Mic')
ax.legend(loc='upper right')

# Добавляем маркеры для границ
ax.plot([0, 1], [0, 0], 'k-', linewidth=2, label='Hard Boundary')
ax.plot([0, 1], [1, 1], 'k-', linewidth=2)

# Создаем первый кадр
phase0 = phases[0]
p_real0 = np.real(p_complex * np.exp(1j * phase0))
contour = ax.tricontourf(triang, p_real0, levels=50, cmap='RdBu_r', 
                        vmin=-abs_max, vmax=abs_max)

# Добавляем цветовую шкалу
cbar = fig.colorbar(contour, ax=ax, shrink=0.8)
cbar.set_label('Pressure [Pa]')

# Функция обновления анимации
def update(frame):
    # Удаляем предыдущие контуры
    for coll in ax.collections:
        if hasattr(coll, 'remove'):
            coll.remove()
    
    phase = phases[frame]
    p_real = np.real(p_complex * np.exp(1j * phase))
    
    # Создаем новый контур
    contour = ax.tricontourf(triang, p_real, levels=50, cmap='RdBu_r', 
                            vmin=-abs_max, vmax=abs_max)
    
    # Обновляем заголовок
    ax.set_title(f'Pressure at {freq} Hz, Phase: {phase/np.pi:.2f}π rad\nHard Boundaries Top/Bottom')
    
    return contour.collections if hasattr(contour, 'collections') else [contour]

# Создаем анимацию
ani = animation.FuncAnimation(
    fig, update, frames=num_frames, interval=100, repeat=True, blit=False)

# Сохраняем GIF
gif_filename = f"pressure_field_hard_boundaries_{freq}Hz.gif"
ani.save(gif_filename, writer='pillow', fps=10, dpi=100)
print(f"GIF saved as {gif_filename}")

# Также сохраняем статичный график для справки
plt.figure(figsize=(8, 6))
p_real = np.real(p_complex)
contour_static = plt.tricontourf(triang, p_real, levels=50, cmap='RdBu_r')
plt.colorbar(contour_static, label='Pressure [Pa]')
plt.plot(Sx, Sy, 'ro', markersize=8, label='Source')
plt.plot(0.9, 0.9, 'go', markersize=8, label='Mic')
plt.plot([0, 1], [0, 0], 'k-', linewidth=2, label='Hard Boundary')
plt.plot([0, 1], [1, 1], 'k-', linewidth=2)
plt.title(f'Pressure Distribution at {freq} Hz (Hard Boundaries)')
plt.xlabel('X [m]')
plt.ylabel('Y [m]')
plt.legend()
plt.savefig(f"pressure_hard_boundaries_{freq}Hz.png")
print(f"Static plot saved as pressure_hard_boundaries_{freq}Hz.png")
