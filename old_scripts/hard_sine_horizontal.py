import numpy as np
import ufl
from dolfinx.fem import Function, functionspace, assemble_scalar, form, Constant
from dolfinx.fem.petsc import LinearProblem
from ufl import dx, grad, inner
from mpi4py import MPI
from petsc4py import PETSc
import matplotlib.pyplot as plt
import matplotlib.tri as tri
import matplotlib.animation as animation
import gmsh
from dolfinx.io import gmshio

deg = 2

n_elem = 64

width = 1.0
height = 1.0
A = 0.05
num_waves = 3
num_points_boundary = 50

# Generate mesh with wavy boundaries using gmsh
gmsh.initialize()
gmsh.model.add("wavy_channel")

# Points for bottom boundary
bottom_points = []
for i in range(num_points_boundary + 1):
    x = width * i / num_points_boundary
    y = A * np.sin(2 * np.pi * num_waves * x / width)
    p = gmsh.model.geo.addPoint(x, y, 0)
    bottom_points.append(p)

# Points for top boundary
top_points = []
for i in range(num_points_boundary + 1):
    x = width * i / num_points_boundary
    y = height + A * np.sin(2 * np.pi * num_waves * x / width)
    p = gmsh.model.geo.addPoint(x, y, 0)
    top_points.append(p)

# Splines for boundaries
bottom_spline = gmsh.model.geo.addSpline(bottom_points)
top_spline = gmsh.model.geo.addSpline(top_points)

# Left and right boundaries (straight lines)
left_line = gmsh.model.geo.addLine(bottom_points[0], top_points[0])
right_line = gmsh.model.geo.addLine(bottom_points[-1], top_points[-1])

# Loop and surface
loop = gmsh.model.geo.addCurveLoop([bottom_spline, right_line, -top_spline, -left_line])
surface = gmsh.model.geo.addPlaneSurface([loop])

gmsh.model.geo.synchronize()

# Physical groups for tags
gmsh.model.addPhysicalGroup(1, [bottom_spline], 1)  # bottom (1)
gmsh.model.addPhysicalGroup(1, [top_spline], 3)     # top (3)
gmsh.model.addPhysicalGroup(1, [left_line], 4)      # left (4)
gmsh.model.addPhysicalGroup(1, [right_line], 2)     # right (2)
gmsh.model.addPhysicalGroup(2, [surface], 1)        # domain

# Generate mesh
lc = width / n_elem
gmsh.option.setNumber("Mesh.CharacteristicLengthMin", lc)
gmsh.option.setNumber("Mesh.CharacteristicLengthMax", lc)
gmsh.model.mesh.generate(2)

# Import to dolfinx
msh, cell_markers, facet_markers = gmshio.model_to_mesh(gmsh.model, MPI.COMM_WORLD, 0, gdim=2)
gmsh.finalize()

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

# Create ds measures for different boundary types
ds_impedance = ufl.Measure("ds", domain=msh, subdomain_data=facet_markers)((2, 4))  # right (2) and left (4)
ds_neumann = ufl.Measure("ds", domain=msh, subdomain_data=facet_markers)((1, 3))    # bottom (1) and top (3)

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
rhs = inner(f, v) * dx

uh = Function(V)
problem = LinearProblem(a, rhs, u=uh, petsc_options={"ksp_type": "preonly", "pc_type": "lu"})
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
x_lin = np.linspace(0, 1, 100)
y_bottom_min = np.min(A * np.sin(2 * np.pi * num_waves * x_lin / width)) - 0.05
y_top_max = height + np.max(A * np.sin(2 * np.pi * num_waves * x_lin / width)) + 0.05
ax.set_ylim(y_bottom_min, y_top_max)
ax.set_aspect('equal')
ax.set_xlabel('X [m]')
ax.set_ylabel('Y [m]')
ax.set_title(f'Pressure Distribution at {freq} Hz (Wavy Hard Boundaries)')

# Add constant elements (source and microphone)
ax.plot(Sx, Sy, 'ro', markersize=8, label='Source')
ax.legend(loc='upper right')

# Add wavy boundaries
x_bound = np.linspace(0, 1, 100)
y_bottom = A * np.sin(2 * np.pi * num_waves * x_bound / width)
y_top = height + A * np.sin(2 * np.pi * num_waves * x_bound / width)
ax.plot(x_bound, y_bottom, 'k-', linewidth=2, label='Wavy Hard Boundary')
ax.plot(x_bound, y_top, 'k-', linewidth=2)

# Add left and right boundaries (impedance)
ax.plot([0, 0], [y_bottom[0], y_top[0]], 'b-', linewidth=1, label='Impedance Boundary')
ax.plot([1, 1], [y_bottom[-1], y_top[-1]], 'b-', linewidth=1)

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
    ax.set_title(f'Pressure at {freq} Hz, Phase: {phase/np.pi:.2f}π rad\nWavy Hard Boundaries Top/Bottom')
    
    return contour.collections if hasattr(contour, 'collections') else [contour]

ani = animation.FuncAnimation(
    fig, update, frames=num_frames, interval=100, repeat=True, blit=False)

gif_filename = f"animations/pressure_field_wavy_boundaries_{freq}Hz.gif"
ani.save(gif_filename, writer='pillow', fps=10, dpi=100)
print(f"GIF saved as {gif_filename}")

# Also save static plot for reference
plt.figure(figsize=(8, 6))
p_real = np.real(p_complex)
contour_static = plt.tricontourf(triang, p_real, levels=50, cmap='RdBu_r')
plt.colorbar(contour_static, label='Pressure [Pa]')
plt.plot(Sx, Sy, 'ro', markersize=8, label='Source')
plt.plot(0.9, 0.9, 'go', markersize=8, label='Mic')
plt.plot(x_bound, y_bottom, 'k-', linewidth=2, label='Wavy Hard Boundary')
plt.plot(x_bound, y_top, 'k-', linewidth=2)
plt.plot([0, 0], [y_bottom[0], y_top[0]], 'b-', linewidth=1, label='Impedance Boundary')
plt.plot([1, 1], [y_bottom[-1], y_top[-1]], 'b-', linewidth=1)
plt.title(f'Pressure Distribution at {freq} Hz (Wavy Hard Boundaries)')
plt.xlabel('X [m]')
plt.ylabel('Y [m]')
plt.xlim(0, 1)
plt.ylim(y_bottom_min, y_top_max)
plt.legend()
plt.savefig(f"pic/pressure_wavy_boundaries_{freq}Hz.png")
print(f"Static plot saved as pic/pressure_wavy_boundaries_{freq}Hz.png")
