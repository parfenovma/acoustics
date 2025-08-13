# TODO@parfenovma fix this script

import numpy as np
import ufl
import gmsh
import tempfile
import os
from dolfinx import io, fem
from dolfinx.fem import Function, functionspace, assemble_scalar, form, Constant
from dolfinx.fem.petsc import LinearProblem
from ufl import dx, grad, inner, ds
from mpi4py import MPI
from petsc4py import PETSc
import matplotlib.pyplot as plt
import matplotlib.tri as tri
import matplotlib.animation as animation

# Параметры
wall_positions = [0.3, 0.5, 0.7]
wall_length = 0.5
wall_thickness = 0.01
domain_size = 1.0
base_tag = 100
freq = 1000.0

# Инициализация Gmsh
gmsh.initialize()
gmsh.model.add("walls_domain")

# Создаём основной прямоугольник
main_rect = gmsh.model.occ.addRectangle(0, 0, 0, domain_size, domain_size)

# Список для хранения тегов стенок
wall_surfaces = []

# Физические группы для стенок
for i, y in enumerate(wall_positions):
    # Создаём стенку
    wall = gmsh.model.occ.addRectangle(0, y - wall_thickness/2, 0, wall_length, wall_thickness)
    wall_surfaces.append(wall)

# Синхронизируем, чтобы геометрия была доступна
gmsh.model.occ.synchronize()

# === ВАЖНО: Теперь получаем границы стенок ДО вычитания ===
wall_physical_groups = []
for i, wall_tag in enumerate(wall_surfaces):
    # Получаем границы (линии) стенки
    dim_tags = gmsh.model.getBoundary([(2, wall_tag)], oriented=False)
    wall_line_tags = [tag for (dim, tag) in dim_tags if dim == 1]
    
    # Создаём физическую группу для жёсткой границы
    pg = gmsh.model.addPhysicalGroup(1, wall_line_tags, tag=base_tag + i + 1)
    gmsh.model.setPhysicalName(1, pg, f"Wall_{i+1}")
    wall_physical_groups.append(pg)

# Теперь вычитаем все стенки из основного прямоугольника
cut_result = [(2, main_rect)]
for wall in wall_surfaces:
    cut_result, _ = gmsh.model.occ.cut(cut_result, [(2, wall)], 
                                      removeObject=True, removeTool=True)
# Обновляем тег основной области
main_surface_tag = cut_result[0][1]

# Синхронизируем после операций CSG
gmsh.model.occ.synchronize()

# === Внешние границы ===
all_boundary = gmsh.model.getBoundary([(2, main_surface_tag)], oriented=False)
external_lines = []
for (dim, tag) in all_boundary:
    if dim != 1:
        continue
    com = gmsh.model.occ.getCenterOfMass(1, tag)
    x, y = com[0], com[1]
    if np.isclose(x, 0) or np.isclose(x, domain_size) or \
       np.isclose(y, 0) or np.isclose(y, domain_size):
        external_lines.append(tag)

# Физическая группа для внешних границ (импеданс)
external_group = gmsh.model.addPhysicalGroup(1, external_lines, tag=base_tag)
gmsh.model.setPhysicalName(1, external_group, "External")

# Настройка сетки
gmsh.option.setNumber("Mesh.CharacteristicLengthMin", 0.02)
gmsh.option.setNumber("Mesh.CharacteristicLengthMax", 0.05)
gmsh.model.mesh.generate(2)

# Сохраняем временный файл
with tempfile.NamedTemporaryFile(suffix=".msh", delete=False) as tmp:
    msh_file = tmp.name
    gmsh.write(msh_file)

# Завершаем Gmsh
gmsh.finalize()

# Загружаем в DOLFINx
msh, cell_tags, facet_tags = io.gmshio.read_from_msh(msh_file, MPI.COMM_SELF, 0, gdim=2)
os.unlink(msh_file)

# === Решение задачи ===
V = functionspace(msh, ("CG", 2))
u = ufl.TrialFunction(V)
v = ufl.TestFunction(V)

Sx, Sy = 0.1, 0.1
Q = 0.0001
c0 = 340
rho_0 = 1.225
omega = Constant(msh, PETSc.ScalarType(1))
k0 = Constant(msh, PETSc.ScalarType(1))

# Источник
alfa = 0.015
delta_tmp = Function(V)
delta_tmp.interpolate(lambda x: 1/(np.abs(alfa)*np.sqrt(np.pi)) * 
                     np.exp(-(((x[0]-Sx)**2 + (x[1]-Sy)**2)/(alfa**2))))
int_delta = assemble_scalar(form(delta_tmp * dx))
delta = delta_tmp / int_delta

# Импеданс
r = Function(V)
r.interpolate(lambda x: np.sqrt((x[0]-Sx)**2 + (x[1]-Sy)**2) + 1e-8)
Z = rho_0 * c0 / (1 + 1/(1j * k0 * r))

omega.value = freq * 2 * np.pi
k0.value = 2 * np.pi * freq / c0

f = 1j * rho_0 * omega * Q * delta
g = 1j * rho_0 * omega / Z

# Граничные условия
ds_ext = ds(subdomain_data=facet_tags, subdomain_id=base_tag)  # внешние
ds_walls = sum((ds(subdomain_data=facet_tags, subdomain_id=base_tag + i + 1) 
                for i in range(len(wall_positions))), ds(domain=msh))

# Формулировка
a = inner(grad(u), grad(v)) * dx - k0**2 * inner(u, v) * dx + g * inner(u, v) * ds_ext
L = inner(f, v) * dx

# Решение
uh = Function(V)
problem = LinearProblem(a, L, u=uh, petsc_options={"ksp_type": "preonly", "pc_type": "lu"})
problem.solve()

# === Визуализация ===
x = msh.geometry.x
dof_c = V.tabulate_dof_coordinates()
dof_to_vert = np.array([np.argmin(np.linalg.norm(dof_c - xi, axis=1)) for xi in x])
p_complex = uh.x.array[dof_to_vert]

triang = tri.Triangulation(x[:, 0], x[:, 1], 
                          msh.topology.connectivity(2, 0).array.reshape(-1, 3))

# Анимация
fig, ax = plt.subplots(figsize=(10, 8))
ax.set_xlim(0, 1)
ax.set_ylim(0, 1)
ax.set_aspect('equal')
ax.plot(Sx, Sy, 'ro', ms=10, label='Source')
ax.plot(0.9, 0.9, 'go', ms=10, label='Mic')
for y in wall_positions:
    ax.plot([0, wall_length], [y, y], 'k-', lw=3, label='Wall' if y == wall_positions[0] else "")
ax.legend()

phases = np.linspace(0, 2*np.pi, 30)
abs_max = np.abs(p_complex).max()

def update(frame):
    for coll in ax.collections:
        coll.remove()
    p_real = np.real(p_complex * np.exp(1j * phases[frame]))
    ax.tricontourf(triang, p_real, levels=50, cmap='RdBu_r', vmin=-abs_max, vmax=abs_max)
    ax.set_title(f'Pressure at {freq} Hz, phase: {phases[frame]/np.pi:.2f}π')
    return ax.collections

ani = animation.FuncAnimation(fig, update, frames=30, interval=100, blit=False)
ani.save(f"pressure_with_walls_{freq}Hz.gif", writer='pillow', fps=10)
plt.close()

# Статичный график
plt.figure(figsize=(10, 8))
p_real = np.real(p_complex)
plt.tricontourf(triang, p_real, levels=50, cmap='RdBu_r', vmin=-abs_max, vmax=abs_max)
plt.colorbar(label='Pressure [Pa]')
plt.plot(Sx, Sy, 'ro', ms=10, label='Source')
plt.plot(0.9, 0.9, 'go', ms=10, label='Mic')
for y in wall_positions:
    plt.plot([0, wall_length], [y, y], 'k-', lw=3, label='Wall' if y == wall_positions[0] else "")
plt.legend()
plt.title(f'Pressure at {freq} Hz')
plt.savefig(f"pressure_with_walls_{freq}Hz.png")
plt.show()

print("✅ Готово: сетка с жёсткими стенками создана, решение найдено, визуализация сохранена.")
