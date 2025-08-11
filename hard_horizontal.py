import numpy as np
import ufl
from dolfinx.fem import Function, functionspace, assemble_scalar, form, Constant
from dolfinx.fem.petsc import LinearProblem
from dolfinx.mesh import create_unit_square, locate_entities_boundary, meshtags
from ufl import dx, grad, inner, ds
from mpi4py import MPI
from petsc4py import PETSc
import matplotlib.pyplot as plt
import matplotlib.tri as tri
import matplotlib.animation as animation

deg = 2
n_elem = 64

msh = create_unit_square(MPI.COMM_SELF, n_elem, n_elem)
n = ufl.FacetNormal(msh)

# Select frequency for visualization
freq = 800.0  # Hz
print(f"Computing for frequency: {freq} Hz")

V = functionspace(msh, ("CG", deg))
u = ufl.TrialFunction(V)
v = ufl.TestFunction(V)

Q = 0.0001
Sx = 0.1
Sy = 0.5

c0 = 340
rho_0 = 1.225
omega = Constant(msh, PETSc.ScalarType(1))
k0 = Constant(msh, PETSc.ScalarType(1))

# Define boundaries:
# 1 - bottom (y=0)
# 2 - right (x=1)
# 3 - top (y=1)
# 4 - left (x=0)

# Hard boundaries (Neumann condition) on top and bottom boundaries
bottom_facets = locate_entities_boundary(msh, msh.topology.dim-1, lambda x: np.isclose(x[1], 0.0))
top_facets = locate_entities_boundary(msh, msh.topology.dim-1, lambda x: np.isclose(x[1], 1.0))

# Impedance boundaries on left and right boundaries
left_facets = locate_entities_boundary(msh, msh.topology.dim-1, lambda x: np.isclose(x[0], 0.0))
right_facets = locate_entities_boundary(msh, msh.topology.dim-1, lambda x: np.isclose(x[0], 1.0))

# Create meshtags for boundaries
facets = np.hstack([bottom_facets, top_facets, left_facets, right_facets])
markers = np.hstack([
    np.full_like(bottom_facets, 1),
    np.full_like(top_facets, 3),
    np.full_like(left_facets, 4),
    np.full_like(right_facets, 2)
])
sorted_facets = np.argsort(facets)
facet_tag = meshtags(msh, msh.topology.dim-1, facets[sorted_facets], markers[sorted_facets])

# Create ds measures for different boundary types
ds_impedance = ds(subdomain_data=facet_tag)((2,4))  # right (2) and left (4) boundaries
ds_neumann = ds(subdomain_data=facet_tag)((1,3))    # bottom (1) and top (3) boundaries

alfa = 0.015
delta_tmp = Function(V)
delta_tmp.interpolate(lambda x: 1/(np.abs(alfa)*np.sqrt(np.pi)) * 
                     np.exp(-(((x[0]-Sx)**2 + (x[1]-Sy)**2)/(alfa**2))))
int_delta_tmp = assemble_scalar(form(delta_tmp*dx))
delta = delta_tmp / int_delta_tmp

r = Function(V)
r.interpolate(lambda x: np.sqrt((x[0]-Sx)**2 + (x[1]-Sy)**2) + 1e-8)
Z = rho_0 * c0 / (1 + 1/(1j * k0 * r))

omega.value = freq * 2 * np.pi
k0.value = 2 * np.pi * freq / c0

f = 1j * rho_0 * omega * Q * delta
g = 1j * rho_0 * omega / Z

# Problem formulation with different boundary conditions
# Hard boundaries (Neumann condition) do not contribute to the bilinear form
a = inner(grad(u), grad(v)) * dx - k0**2 * inner(u, v) * dx + g * inner(u, v) * ds_impedance
L = inner(f, v) * dx

uh = Function(V)
problem = LinearProblem(a, L, u=uh, petsc_options={"ksp_type": "preonly", "pc_type": "lu"})
problem.solve()

vertex_coords = msh.geometry.x
dof_coords = V.tabulate_dof_coordinates()

dof_idx_on_vertex = []
for v in vertex_coords:
    distances = np.linalg.norm(dof_coords - v, axis=1)
    closest_dof = np.argmin(distances)
    dof_idx_on_vertex.append(closest_dof)
dof_idx_on_vertex = np.array(dof_idx_on_vertex)

p_complex = uh.x.array[dof_idx_on_vertex]

x = vertex_coords[:, 0]
y = vertex_coords[:, 1]
triangles = msh.topology.connectivity(msh.topology.dim, 0).array.reshape(-1, 3)
triang = tri.Triangulation(x, y, triangles)

num_frames = 30
phases = np.linspace(0, 2*np.pi, num_frames, endpoint=False)
abs_max = np.abs(p_complex).max()

# Create figure
fig, ax = plt.subplots(figsize=(8, 6))
ax.set_xlim(0, 1)
ax.set_ylim(0, 1)
ax.set_aspect('equal')
ax.set_xlabel('X [m]')
ax.set_ylabel('Y [m]')
ax.set_title(f'Pressure Distribution at {freq} Hz (Hard Boundaries)')

# Add constant elements (source and microphone)
ax.plot(Sx, Sy, 'ro', markersize=8, label='Source')
ax.legend(loc='upper right')

# Add boundary markers
ax.plot([0, 1], [0, 0], 'k-', linewidth=2, label='Hard Boundary')
ax.plot([0, 1], [1, 1], 'k-', linewidth=2)

# Create first frame
phase0 = phases[0]
p_real0 = np.real(p_complex * np.exp(1j * phase0))
contour = ax.tricontourf(triang, p_real0, levels=50, cmap='RdBu_r', 
                        vmin=-abs_max, vmax=abs_max)

# Add colorbar
cbar = fig.colorbar(contour, ax=ax, shrink=0.8)
cbar.set_label('Pressure [Pa]')

# Animation update function
def update(frame):
    # Remove previous contours
    for coll in ax.collections:
        if hasattr(coll, 'remove'):
            coll.remove()
    
    phase = phases[frame]
    p_real = np.real(p_complex * np.exp(1j * phase))
    
    # Create new contour
    contour = ax.tricontourf(triang, p_real, levels=50, cmap='RdBu_r', 
                            vmin=-abs_max, vmax=abs_max)
    
    # Update title
    ax.set_title(f'Pressure at {freq} Hz, Phase: {phase/np.pi:.2f}π rad\nHard Boundaries Top/Bottom')
    
    return contour.collections if hasattr(contour, 'collections') else [contour]

ani = animation.FuncAnimation(
    fig, update, frames=num_frames, interval=100, repeat=True, blit=False)

gif_filename = f"animations/pressure_field_hard_boundaries_{freq}Hz.gif"
ani.save(gif_filename, writer='pillow', fps=10, dpi=100)
print(f"GIF saved as {gif_filename}")

# Also save static plot for reference
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
plt.savefig(f"pic/pressure_hard_boundaries_{freq}Hz.png")
print(f"Static plot saved as pic/pressure_hard_boundaries_{freq}Hz.png")