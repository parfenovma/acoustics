import numpy as np
import ufl
from dolfinx import fem, io
from dolfinx.fem import Function, functionspace, assemble_scalar, form, Constant
from dolfinx.fem.petsc import LinearProblem
from ufl import dx, grad, inner
from mpi4py import MPI
from petsc4py import PETSc
import matplotlib.pyplot as plt
import matplotlib.tri as tri
import matplotlib.animation as animation
import gmsh
from abc import ABC, abstractmethod
from dataclasses import dataclass

# =============================================================================
# 1. Configuration Data Class
# =============================================================================
# @dataclass
# class SimulationConfig:
#     """A single container for all simulation parameters."""
#     width: float = 1.0
#     height: float = 1.0
#     A: float = 0.1
#     num_waves: int = 3
#     num_points_boundary: int = 50
#     n_elem: int = 64
#     deg: int = 2
#     freq: float = 800.0
#     Sx: float = 0.1
#     Sy: float = 0.5
#     boundary_type: str = 'wavy'  # 'wavy' or 'rigid'
    
#     # Physical constants
#     c0: float = 340.0
#     rho_0: float = 1.225
    
#     # Source properties
#     Q: float = 0.0001
#     alfa: float = 0.015 # for Gaussian source approximation

# =============================================================================
# 1. Configuration Data Class (Updated)
# =============================================================================
@dataclass
class SimulationConfig:
    """A single container for all simulation parameters."""
    width: float = 1.0; height: float = 1.0; A: float = 0.0; num_waves: int = 0
    num_points_boundary: int = 50; n_elem: int = 64; deg: int = 2
    boundary_type: str = 'rigid'

    # Time-domain specific parameters
    t_end: float = 0.006  # End time of simulation [s]
    dt: float = 2e-6      # Time step [s] - MUST satisfy CFL condition

    # Source parameters (for Ricker pulse)
    source_freq: float = 2000.0 # Center frequency of the Ricker pulse [Hz]
    source_delay: float = 0.0005 # Time delay for the pulse peak [s]
    Sx: float = 0.5; Sy: float = 0.5
    
    # Physical constants
    c0: float = 340.0

# =============================================================================
# 2. Interfaces (Abstract Base Classes)
# =============================================================================

class IMeshGenerator(ABC):
    @abstractmethod
    def generate(self):
        pass

class ISourceTerm(ABC):
    @abstractmethod
    def get_form(self, V: fem.FunctionSpace, v: ufl.TestFunction, omega: Constant) -> ufl.Form:
        pass

class IBoundaryCondition(ABC):
    # CHANGED: Interface now gets more context for accurate physics
    @abstractmethod
    def get_lhs_term(self, V: fem.FunctionSpace, u: ufl.TrialFunction, v: ufl.TestFunction, ds: ufl.Measure, k0: Constant, omega: Constant) -> ufl.Form:
        pass

class IProblemSolver(ABC):
    @abstractmethod
    def solve(self, frequency: float):
        pass

class IVisualizer(ABC):
    @abstractmethod
    def create_animation(self, output_filename: str = None):
        pass

    @abstractmethod
    def create_static_plot(self, output_filename: str = None):
        pass

# =============================================================================
# 3. Concrete Implementations
# =============================================================================

# --- MESH IMPLEMENTATION ---
class GmshChannelMesh(IMeshGenerator):
    def __init__(self, config: SimulationConfig):
        self.config = config

    def generate(self):
        cfg = self.config
        gmsh.initialize()
        gmsh.model.add("channel")

        if cfg.boundary_type == 'wavy':
            bottom_points = self._create_boundary_points(lambda x: cfg.A * np.sin(2 * np.pi * cfg.num_waves * x / cfg.width))
            top_points = self._create_boundary_points(lambda x: cfg.height + cfg.A * np.sin(2 * np.pi * cfg.num_waves * x / cfg.width))
            bottom_curve = gmsh.model.geo.addSpline(bottom_points)
            top_curve = gmsh.model.geo.addSpline(top_points)
            left_line = gmsh.model.geo.addLine(bottom_points[0], top_points[0])
            right_line = gmsh.model.geo.addLine(bottom_points[-1], top_points[-1])
            # The loop for wavy boundaries needs to reverse the top and left curves
            curve_loop_tags = [bottom_curve, right_line, -top_curve, -left_line]
            
        elif cfg.boundary_type == 'rigid':
            p1, p2, p3, p4 = (gmsh.model.geo.addPoint(x, y, 0) for x, y in [(0, 0), (cfg.width, 0), (cfg.width, cfg.height), (0, cfg.height)])
            bottom_curve = gmsh.model.geo.addLine(p1, p2)
            right_line = gmsh.model.geo.addLine(p2, p3)
            top_curve = gmsh.model.geo.addLine(p3, p4)
            left_line = gmsh.model.geo.addLine(p4, p1)
            # FIX: For rigid boundaries, all curves are already in order. No reversal needed.
            curve_loop_tags = [bottom_curve, right_line, top_curve, left_line]
        else:
            raise ValueError("boundary_type must be 'wavy' or 'rigid'")

        # --- THIS IS THE CORRECTED PART ---
        loop = gmsh.model.geo.addCurveLoop(curve_loop_tags)
        surface = gmsh.model.geo.addPlaneSurface([loop])
        gmsh.model.geo.synchronize()

        gmsh.model.addPhysicalGroup(1, [bottom_curve], 1); gmsh.model.addPhysicalGroup(1, [right_line], 2)
        gmsh.model.addPhysicalGroup(1, [top_curve], 3); gmsh.model.addPhysicalGroup(1, [left_line], 4)
        gmsh.model.addPhysicalGroup(2, [surface], 1)

        lc = cfg.width / cfg.n_elem
        gmsh.option.setNumber("Mesh.CharacteristicLengthMin", lc); gmsh.option.setNumber("Mesh.CharacteristicLengthMax", lc)
        gmsh.model.mesh.generate(2)

        msh, _, facet_markers = io.gmshio.model_to_mesh(gmsh.model, MPI.COMM_WORLD, 0, gdim=2)
        gmsh.finalize()
        return msh, facet_markers

    def _create_boundary_points(self, y_func):
        points = []
        for i in range(self.config.num_points_boundary + 1):
            x = self.config.width * i / self.config.num_points_boundary
            y = y_func(x)
            points.append(gmsh.model.geo.addPoint(x, y, 0))
        return points

# --- SOURCE IMPLEMENTATION ---
class GaussianPointSource(ISourceTerm):
    def __init__(self, config: SimulationConfig):
        self.config = config

    def get_form(self, V: fem.FunctionSpace, v: ufl.TestFunction, omega: Constant) -> ufl.Form:
        cfg = self.config
        delta_tmp = Function(V)
        delta_tmp.interpolate(lambda x: 1/(np.abs(cfg.alfa)*np.sqrt(np.pi)) * 
                             np.exp(-(((x[0]-cfg.Sx)**2 + (x[1]-cfg.Sy)**2)/(cfg.alfa**2))))
        int_delta = assemble_scalar(form(delta_tmp * dx))
        delta = delta_tmp / int_delta
        f = 1j * cfg.rho_0 * omega * cfg.Q * delta
        return inner(f, v) * dx

# --- BOUNDARY CONDITION IMPLEMENTATIONS ---
class ImpedanceBC(IBoundaryCondition):
    def __init__(self, config: SimulationConfig, tags: list[int]):
        self.config = config
        # Убедимся, что tags - это всегда список, даже если передано одно число
        if not isinstance(tags, (list, tuple)):
            tags = [tags]
        self.tags = tags

    def get_lhs_term(self, V: fem.FunctionSpace, u: ufl.TrialFunction, v: ufl.TestFunction, ds: ufl.Measure, k0: Constant, omega: Constant) -> ufl.Form:
        cfg = self.config
        r = Function(V)
        r.interpolate(lambda x: np.sqrt((x[0]-cfg.Sx)**2 + (x[1]-cfg.Sy)**2) + 1e-8)
        
        Z = cfg.rho_0 * cfg.c0 / (1 + 1/(1j * k0 * r))
        g = 1j * cfg.rho_0 * omega / Z
        
        # --- FIX: Iterate over tags and sum the integrals ---
        integral_term = 0
        for tag in self.tags:
            integral_term += g * inner(u, v) * ds(tag)
        return integral_term

class HardWallBC(IBoundaryCondition):
    def get_lhs_term(self, V, u, v, ds, k0, omega) -> ufl.Form:
        # Возвращает нулевую форму правильного типа, не влияющую на сборку
        zero = Constant(V.mesh, PETSc.ScalarType(0.0))
        return inner(zero * u, v) * ds(1) # ds(1) здесь формальность, т.к. множитель 0

# --- PROBLEM SOLVER IMPLEMENTATION ---
class HelmholtzProblemSolver(IProblemSolver):
    def __init__(self, mesh, facet_markers, source: ISourceTerm, bcs: list[IBoundaryCondition], config: SimulationConfig):
        self.mesh = mesh
        self.facet_markers = facet_markers
        self.source = source
        self.bcs = bcs
        self.config = config
        
        self.V = functionspace(mesh, ("CG", config.deg))
        self.u, self.v = ufl.TrialFunction(self.V), ufl.TestFunction(self.V)
        self.ds = ufl.Measure("ds", domain=mesh, subdomain_data=facet_markers)
        self.omega = Constant(mesh, PETSc.ScalarType(1))
        self.k0 = Constant(mesh, PETSc.ScalarType(1))

    def solve(self, frequency: float) -> fem.Function:
        print(f"Assembling and solving for frequency: {frequency} Hz")
        self.omega.value = frequency * 2 * np.pi
        self.k0.value = self.omega.value / self.config.c0

        a = inner(grad(self.u), grad(self.v)) * dx - self.k0**2 * inner(self.u, self.v) * dx
        L = self.source.get_form(self.V, self.v, self.omega)
 
        for bc in self.bcs:
            a += bc.get_lhs_term(self.V, self.u, self.v, self.ds, self.k0, self.omega)
        
        uh = Function(self.V)
        problem = LinearProblem(a, L, u=uh, petsc_options={"ksp_type": "preonly", "pc_type": "lu"})
        problem.solve()
        return uh

# --- VISUALIZER IMPLEMENTATION (Unchanged from previous version) ---
class MatplotlibVisualizer(IVisualizer):
    def __init__(self, mesh, solution, config):
        self.mesh = mesh
        self.uh = solution
        self.config = config
        # Use a P1 projection for robust visualization
        V_vis = functionspace(mesh, ("Lagrange", 1))
        p_vis = Function(V_vis)
        p_vis.interpolate(self.uh)
        
        self.p_complex = p_vis.x.array
        
        # We need the DoF coordinates from the P1 space for triangulation
        vertex_coords = V_vis.tabulate_dof_coordinates()
        self.triang = tri.Triangulation(vertex_coords[:, 0], vertex_coords[:, 1])

    def create_animation(self, output_filename=None):
        fig, ax = plt.subplots(figsize=(8, 6))
        phases = np.linspace(0, 2 * np.pi, 30, endpoint=False)
        abs_max = np.abs(self.p_complex).max()

        def update(frame):
            ax.clear()  # Clear the axes for the new frame
            self._setup_plot_aesthetics(ax)
            phase = phases[frame]
            p_real = np.real(self.p_complex * np.exp(1j * phase))
            ax.tricontourf(self.triang, p_real, levels=50, cmap='RdBu_r', vmin=-abs_max, vmax=abs_max)
            ax.set_title(f"Pressure at {self.config.freq} Hz, Phase: {phase/np.pi:.2f}π")
            return [] # Return an empty list when blit=False and ax.clear() is used

        # Create animation with blit=False for simplicity and robustness
        ani = animation.FuncAnimation(fig, update, frames=len(phases), interval=100, blit=False)

        if output_filename:
            ani.save(output_filename, writer='pillow', fps=10, dpi=100)
            print(f"Animation saved to {output_filename}")
        
        return ani, fig

    def create_static_plot(self, output_filename=None):
        fig, ax = plt.subplots(figsize=(8, 6))
        self._setup_plot_aesthetics(ax)
        
        p_real = self.p_complex.real
        abs_max = np.abs(p_real).max()
        
        contour = ax.tricontourf(self.triang, p_real, levels=50, cmap='RdBu_r', vmin=-abs_max, vmax=abs_max)
        fig.colorbar(contour, ax=ax, label="Pressure [Pa]")
        
        if output_filename:
            plt.savefig(output_filename, dpi=300, bbox_inches='tight')
            print(f"Static plot saved to {output_filename}")
        
        return fig

    def _setup_plot_aesthetics(self, ax):
        cfg = self.config
        ax.set_aspect('equal')
        ax.set_xlabel('X [m]')
        ax.set_ylabel('Y [m]')
        ax.set_title(f"Pressure at {cfg.freq} Hz, Type: {cfg.boundary_type.capitalize()}")
        
        if cfg.boundary_type == 'wavy':
            x = np.linspace(0, cfg.width, 200)
            y_b = cfg.A * np.sin(2 * np.pi * cfg.num_waves * x / cfg.width)
            y_t = cfg.height + cfg.A * np.sin(2 * np.pi * cfg.num_waves * x / cfg.width)
            ax.plot(x, y_b, 'k-', lw=2)
            ax.plot(x, y_t, 'k-', lw=2)
        else:
            ax.add_patch(plt.Rectangle((0, 0), cfg.width, cfg.height, fill=False, color='k', lw=2))
        
        ax.plot(cfg.Sx, cfg.Sy, 'ro', label='Source')
        ax.legend()


# =============================================================================
# 4. Main Application (Composition Root)
# =============================================================================

# def main():
#     config = SimulationConfig(freq=500.0, boundary_type='rigid')
#     # To run the other case, simply change the line above:
#     # config = SimulationConfig(freq=800.0, boundary_type='wavy', A=0.08)

#     mesh_generator = GmshChannelMesh(config)
#     mesh, facet_markers = mesh_generator.generate()
    
#     source_term = GaussianPointSource(config)
    
#     boundary_conditions = [
#         # HardWallBC is the default (natural) condition, no need to add it explicitly
#         ImpedanceBC(config, tags=[2, 4]), # Applies to right (2) and left (4) boundaries
#     ]

#     solver = HelmholtzProblemSolver(mesh, facet_markers, source_term, boundary_conditions, config)
#     solution = solver.solve(config.freq)
    
#     visualizer = MatplotlibVisualizer(mesh, solution, config)
    
#     viz_anim, _ = visualizer.create_animation(f"{config.boundary_type}_anim.gif")
#     viz_static = visualizer.create_static_plot(f"{config.boundary_type}_static.png")
    
#     plt.show()

# if __name__ == "__main__":
#     main()


import numpy as np
import ufl
from dolfinx import fem, io, plot
from dolfinx.fem import Function, functionspace, assemble_scalar, form, Constant
from dolfinx.fem.petsc import LinearProblem, assemble_vector
from ufl import dx, grad, inner
from mpi4py import MPI
from petsc4py import PETSc
import matplotlib.pyplot as plt
import matplotlib.animation as animation
import gmsh
from abc import ABC, abstractmethod
from dataclasses import dataclass
from tqdm import tqdm # Для красивого индикатора прогресса


# --- SOURCE IMPLEMENTATION ---
class RickerPulseSource(ISourceTerm):
    def __init__(self, V: fem.FunctionSpace, config: SimulationConfig):
        self.config = config
        self._spatial_component = Function(V)
        alfa = 0.3
        self._spatial_component.interpolate(
            lambda x: 1/(np.abs(alfa)*np.sqrt(np.pi)) * np.exp(-(((x[0]-config.Sx)**2 + (x[1]-config.Sy)**2)/(alfa**2)))
        )
        self._spatial_component /= assemble_scalar(form(self._spatial_component * dx))

    def get_spatial_component(self) -> fem.Function:
        return self._spatial_component

    def get_value_at_time(self, t: float) -> float:
        cfg = self.config
        f, t0 = cfg.source_freq, cfg.source_delay
        arg = (np.pi * f * (t - t0))**2
        return (1.0 - 2.0 * arg) * np.exp(-arg)

    def get_form(self, V, v, omega):
        return super().get_form(V, v, omega)

class TimeDomainWaveSolver(IProblemSolver):
    def __init__(self, mesh, source_term: ISourceTerm, config: SimulationConfig):
        self.mesh = mesh
        self.source_term = source_term
        self.config = config
        
        self.V = functionspace(mesh, ("Lagrange", config.deg))
        self.p, self.p_n, self.p_n_1 = Function(self.V), Function(self.V), Function(self.V)
        
        u, v = ufl.TrialFunction(self.V), ufl.TestFunction(self.V)
        c0, dt = config.c0, config.dt
        
        self.source_amplitude = Constant(mesh, PETSc.ScalarType(0.0))
        self.source_spatial = source_term.get_spatial_component()

        self.a = inner(u, v) * dx
        L_form = (inner(2 * self.p_n - self.p_n_1, v) * dx
                  - (c0 * dt)**2 * inner(grad(self.p_n), grad(v)) * dx
                  + (c0 * dt)**2 * inner(self.source_amplitude * self.source_spatial, v) * dx)
        self.L = form(L_form)

        self.A = fem.petsc.assemble_matrix(form(self.a))
        self.A.assemble()
        
        # --- FIX: Create solver-compatible PETSc vectors from the matrix ---
        self.b = self.A.createVecLeft()  # RHS vector
        self.p_petsc = self.A.createVecRight() # Solution vector
        
        self.solver = PETSc.KSP().create(mesh.comm)
        self.solver.setOperators(self.A)
        self.solver.setType(PETSc.KSP.Type.PREONLY)
        self.solver.getPC().setType(PETSc.PC.Type.LU)
        
    def solve(self, frames_to_store=10): # Added optional argument
        times = np.arange(0, self.config.t_end, self.config.dt)
        pressure_data = []

        print("Starting time-domain simulation...")
        for i, t in enumerate(tqdm(times)):
            self.source_amplitude.value = self.source_term.get_value_at_time(t)
            with self.b.localForm() as loc_b: loc_b.set(0)
            fem.petsc.assemble_vector(self.b, self.L)
            self.solver.solve(self.b, self.p_petsc)
            self.p.x.array[:] = self.p_petsc.array; self.p.x.scatter_forward()
            self.p_n_1.x.array[:] = self.p_n.x.array; self.p_n.x.array[:] = self.p.x.array
            if i % frames_to_store == 0:
                pressure_data.append(self.p.x.array.copy())
        
        stored_times = times[::frames_to_store]
        return np.array(pressure_data), stored_times
   

# --- VISUALIZER IMPLEMENTATION ---
class MatplotlibTimeVisualizer(IVisualizer):
    def __init__(self, mesh, config: SimulationConfig):
        self.mesh = mesh
        self.config = config

        # Setup for visualization (P1 space)
        V_vis = functionspace(mesh, ("Lagrange", 1))
        self.V_sol = functionspace(mesh, ("Lagrange", config.deg))
        vertex_coords = V_vis.tabulate_dof_coordinates()
        self.triang = tri.Triangulation(vertex_coords[:, 0], vertex_coords[:, 1])

        # We need a P1 function to project the high-order solution onto for visualization
        self.p_vis = Function(V_vis)
        self.deg = config.deg

    def create_animation(self, pressure_over_time: np.ndarray, times: np.ndarray, output_filename=None):
        fig, ax = plt.subplots(figsize=(8, 6))
        
        # --- DEBUG FIX 2: Let matplotlib choose the color scale for each frame ---
        # vmax = np.max(np.abs(pressure_over_time)) * 0.5 
        
        p_sol_frame = Function(self.V_sol)

        def update(frame):
            ax.clear()
            self._setup_plot_aesthetics(ax)
            p_sol_frame.x.array[:] = pressure_over_time[frame]
            self.p_vis.interpolate(p_sol_frame)

            # --- DEBUG FIX 2 (cont.): Remove fixed vmin/vmax ---
            ax.tricontourf(self.triang, self.p_vis.x.array, levels=40, cmap='RdBu_r')
            ax.set_title(f"Pressure at t = {times[frame]*1000:.3f} ms") # More precision in title
            return []

        ani = animation.FuncAnimation(fig, update, frames=len(times), interval=50, blit=False)
        if output_filename:
            print(f"Saving animation to {output_filename}...")
            ani.save(output_filename, writer='pillow', fps=20, dpi=120)
            print("Animation saved.")
        
        return ani, fig

    def _setup_plot_aesthetics(self, ax):
        # A simplified version of the previous setup
        cfg = self.config
        ax.set_aspect('equal')
        ax.set_xlabel('X [m]'); ax.set_ylabel('Y [m]')
        ax.add_patch(plt.Rectangle((0, 0), cfg.width, cfg.height, fill=False, color='k', lw=2))
        ax.plot(cfg.Sx, cfg.Sy, 'black', markersize=20, label='Source')
        ax.legend(loc="upper right")
        ax.set_xlim(-0.1, cfg.width + 0.1)
        ax.set_ylim(-0.1, cfg.height + 0.1)

    def create_static_plot(self, output_filename = None):
        return super().create_static_plot(output_filename)

# =============================================================================
# 4. Main Application
# =============================================================================

# Define a concrete GmshChannelMesh for completeness
# class GmshChannelMesh(IMeshGenerator):
#     def __init__(self, config: SimulationConfig): self.config = config
#     def generate(self):
#         # Simplified version for rigid rectangle
#         cfg = self.config
#         gmsh.initialize()
#         gmsh.model.add("channel")
#         gmsh.model.occ.addRectangle(0, 0, 0, cfg.width, cfg.height)
#         gmsh.model.occ.synchronize()
#         lc = 1.0 / cfg.n_elem
#         gmsh.option.setNumber("Mesh.CharacteristicLengthMax", lc)
#         gmsh.model.mesh.generate(2)
#         msh, _, _ = io.gmshio.model_to_mesh(gmsh.model, MPI.COMM_WORLD, 0, gdim=2)
#         gmsh.finalize()
#         return msh, None # No facet markers needed for this simple case

def main_time_domain():
    # --- DEBUG FIX 1: Change simulation parameters ---
    config = SimulationConfig(
        t_end=0.001,       # Stop before the first reflection
        dt=2e-6,           # Keep dt small for stability
        source_delay=0.0003  # Make the pulse appear earlier
    )

    h_min = 1.0 / config.n_elem
    cfl = config.c0 * config.dt / h_min
    print(f"Mesh size (h): ~{h_min:.4f} m"); print(f"CFL number (estimated): {cfl:.1f}")
    if cfl > 0.5:
        print("WARNING: Estimated CFL > 0.5. Simulation might be unstable.")

    mesh_generator = GmshChannelMesh(config); mesh, _ = mesh_generator.generate()
    
    # We need to tell the solver to save frames more frequently
    # This is a bit of a hack, we can modify the solver or just do it here
    # For now, let's modify the solver to accept this parameter.
    
    # Let's modify TimeDomainWaveSolver to accept `frames_to_store`
    V_solver_space = functionspace(mesh, ("Lagrange", config.deg))
    source = RickerPulseSource(V_solver_space, config)
    
    # We will modify the solver class to accept this
    solver = TimeDomainWaveSolver(mesh, source, config)
    
    # --- DEBUG FIX 3: Let's modify what is returned to save more frames ---
    # To do this cleanly, let's add an argument to the solve() method.
    pressure_history, times_stored = solver.solve(frames_to_store=2) # Store every 2nd frame
    
    visualizer = MatplotlibTimeVisualizer(mesh, config)
    visualizer.create_animation(pressure_history, times_stored, "initial_wave.gif")
    
    plt.show()

if __name__ == "__main__":
    main_time_domain()