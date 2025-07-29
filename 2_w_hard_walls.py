import numpy as np
import ufl
import gmsh
import tempfile
import os
from dolfinx import io, fem, geometry
from dolfinx.fem import Function, FunctionSpace, assemble_scalar, form, Constant
from dolfinx.fem.petsc import LinearProblem
from dolfinx.mesh import locate_entities_boundary
from ufl import dx, grad, inner, ds
from mpi4py import MPI
from petsc4py import PETSc
import matplotlib.pyplot as plt
import matplotlib.tri as tri
import matplotlib.animation as animation

# Параметры стенок
wall_positions = [0.3, 0.5, 0.7]  # Y-координаты стенок
wall_length = 0.5                   # Длина стенок
wall_thickness = 0.01               # Толщина стенок для визуализации

# Создаем геометрию в Gmsh
gmsh.initialize()
gmsh.model.add("domain_with_walls")

# Параметры области
domain_size = 1.0
base_tag = 100  # Базовый тег для всех групп (чтобы избежать конфликтов)

# Создаем основную область
rectangle = gmsh.model.occ.addRectangle(0, 0, 0, domain_size, domain_size)

# Создаем стенки
walls = []
for i, y in enumerate(wall_positions):
    wall = gmsh.model.occ.addRectangle(0, y - wall_thickness/2, 0, 
                                      wall_length, wall_thickness)
    walls.append(wall)
    
    # Вычитаем стенку из основной области
    rectangle, _ = gmsh.model.occ.cut([(2, rectangle)], [(2, wall)], 
                                     removeObject=True, removeTool=True)
    rectangle = rectangle[0][1]  # Обновляем ID основной области

# Синхронизируем
gmsh.model.occ.synchronize()

# Получаем все границы
all_boundaries = gmsh.model.getBoundary([(2, rectangle)], oriented=False)

# Создаем физические группы для внешних границ
external_boundaries = []
for boundary in all_boundaries:
    dim, tag = boundary
    com = gmsh.model.occ.getCenterOfMass(dim, tag)
    x, y = com[0], com[1]
    
    # Проверяем, является ли граница внешней
    if np.isclose(x, 0) or np.isclose(x, domain_size) or \
       np.isclose(y, 0) or np.isclose(y, domain_size):
        external_boundaries.append(tag)

# Создаем физическую группу для внешних границ
external_group = gmsh.model.addPhysicalGroup(1, external_boundaries, base_tag)

# Создаем физические группы для стенок
wall_groups = []
for i, wall in enumerate(walls):
    wall_edges = gmsh.model.getBoundary([(2, wall)], oriented=False)
    wall_tags = [e[1] for e in wall_edges]
    wall_group = gmsh.model.addPhysicalGroup(1, wall_tags, base_tag + i + 1)
    wall_groups.append(wall_group)

# Генерируем сетку
gmsh.option.setNumber("Mesh.CharacteristicLengthMin", 0.02)
gmsh.option.setNumber("Mesh.CharacteristicLengthMax", 0.05)
gmsh.model.mesh.generate(2)

# Сохраняем сетку во временный файл
with tempfile.NamedTemporaryFile(suffix=".msh", delete=False) as tmpfile:
    gmsh_file = tmpfile.name
    gmsh.write(gmsh_file)

# Импортируем сетку в DOLFINx
gmsh.finalize()
msh, cell_tags, facet_tags = io.gmshio.read_from_msh(
    gmsh_file, MPI.COMM_SELF, 0, gdim=2
)
os.unlink(gmsh_file)  # Удаляем временный файл

# Выбираем частоту для визуализации
freq = 1000.0  # Гц
print(f"Computing for frequency: {freq} Hz")

# Test and trial function space
V = FunctionSpace(msh, ("CG", 2))
u = ufl.TrialFunction(V)
v = ufl.TestFunction(V)

# Source amplitude
Q = 0.0001
Sx = 0.1
Sy = 0.1

# Fluid properties
c0 = 340
rho_0 = 1.225
omega = Constant(msh, PETSc.ScalarType(1))
k0 = Constant(msh, PETSc.ScalarType(1))

# Delta function source
alfa = 0.015
delta_tmp = Function(V)
delta_tmp.interpolate(lambda x: 1/(np.abs(alfa)*np.sqrt(np.pi)) * 
                     np.exp(-(((x[0]-Sx)**2 + (x[1]-Sy)**2)/(alfa**2))))
int_delta_tmp = assemble_scalar(form(delta_tmp*dx))
delta = delta_tmp / int_delta_tmp

# Impedance calculation
r = Function(V)
r.interpolate(lambda x: np.sqrt((x[0]-Sx)**2 + (x[1]-Sy)**2) + 1e-8)
Z = rho_0 * c0 / (1 + 1/(1j * k0 * r))

# Set frequency parameters
omega.value = freq * 2 * np.pi
k0.value = 2 * np.pi * freq / c0

f = 1j * rho_0 * omega * Q * delta
g = 1j * rho_0 * omega / Z

# Формулировка задачи с разными граничными условиями
# Импедансные условия на всех внешних границах
ds_impedance = ds(subdomain_data=facet_tags, subdomain_id=external_group)

# Условия Неймана на стенках (каждая стенка - отдельный subdomain_id)
ds_walls = []
for i, group in enumerate(wall_groups):
    wall_ds = ds(subdomain_data=facet_tags, subdomain_id=group)
    ds_walls.append(wall_ds)

# Билинейная форма
a = inner(grad(u), grad(v)) * dx - k0**2 * inner(u, v) * dx
a += g * inner(u, v) * ds_impedance  # Импеданс на внешних границах

# Линейная форма
L = inner(f, v) * dx

# Solve the problem
uh = Function(V)
problem = LinearProblem(a, L, u=uh, petsc_options={"ksp_type": "preonly", "pc_type": "lu"})
problem.solve()

# Визуализация результатов
vertex_coords = msh.geometry.x
dof_coords = V.tabulate_dof_coordinates()

# Find DOFs that correspond to mesh vertices
dof_idx_on_vertex = []
for v in vertex_coords:
    distances = np_point = np.array([v[0], v[1]])
    distances = np.linalg.norm(dof_coords - point, axis=1)
    closest_dof = np.argmin(distances)
    dof_idx_on_vertex.append(closest_dof)
dof_idx_on_vertex = np.array(dof_idx_on_vertex)

# Get pressure values at vertices
p_complex = uh.x.array[dof_idx_on_vertex]

# Create triangulation
x = vertex_coords[:, 0]
y = vertex_coords[:, 1]
triangles = msh.topology.connectivity(msh.topology.dim, 0).array.reshape(-1, 3)
triang = tri.Triangulation(x, y, triangles)

# Create animation
num_frames = 30
phases = np.linspace(0, 2*np.pi, num_frames, endpoint=False)
abs_max = np.abs(p_complex).max()

fig, ax = plt.subplots(figsize=(10, 8))
ax.set_xlim(0, domain_size)
ax.set_ylim(0, domain_size)
ax.set_aspect('equal')
ax.set_xlabel('X [m]')
ax.set_ylabel('Y [m]')
ax.set_title(f'Pressure Distribution at {freq} Hz with Walls')

# Source and mic
ax.plot(Sx, Sy, 'ro', markersize=10, label='Source')
ax.plot(0.9, 0.9, 'go', markersize=10, label='Mic')

# Highlight walls
for y in wall_positions:
    ax.plot([0, wall_length], [y, y], 'k-', linewidth=3, label='Hard Wall' if y == wall_positions[0] else "")

ax.legend(loc='upper right')

# First frame
phase0 = phases[0]
p_real0 = np.real(p_complex * np.exp(1j * phase0))
contour = ax.tricontourf(triang, p_real0, levels=50, cmap='RdBu_r', 
                        vmin=-abs_max, vmax=abs_max)
cbar = fig.colorbar(contour, ax=ax, shrink=0.8)
cbar.set_label('Pressure [Pa]')

def update(frame):
    for coll in ax.collections:
        if isinstance(coll, plt.collections.PolyCollection):
            coll.remove()
    
    phase = phases[frame]
    p_real = np.real(p_complex * np.exp(1j * phase))
    
    contour = ax.tricontourf(triang, p_real, levels=50, cmap='RdBu_r', 
                            vmin=-abs_max, vmax=abs_max)
    ax.set_title(f'Pressure at {freq} Hz, Phase: {phase/np.pi:.2f}π rad')
    
    return [contour]

ani = animation.FuncAnimation(
    fig, update, frames=num_frames, interval=100, blit=False)

gif_filename = f"pressure_with_walls_{freq}Hz.gif"
ani.save(gif_filename, writer='pillow', fps=10, dpi=150)
print(f"GIF saved as {gif_filename}")

# Static plot
plt.figure(figsize=(10, 8))
p_real = np.real(p_complex)
contour_static = plt.tricontourf(triang, p_real, levels=50, cmap='RdBu_r')
plt.colorbar(contour_static, label='Pressure [Pa]')
plt.plot(Sx, Sy, 'ro', markersize=10, label='Source')
plt.plot(0.9, 0.9, 'go', markersize=10, label='Mic')
for y in wall_positions:
    plt.plot([0, wall_length], [y, y], 'k-', linewidth=3, label='Hard Wall' if y == wall_positions[0] else "")
plt.title(f'Pressure Distribution at {freq} Hz with Walls')
plt.xlabel('X [m]')
plt.ylabel('Y [m]')
plt.legend()
plt.savefig(f"pressure_with_wallMeshTagss_{freq}Hz.png")
print(f"Static plot saved as pressure_with_walls_{freq}Hz.png")
