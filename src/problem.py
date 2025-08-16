import abc
import numpy as np
import dolfinx.fem as fem
import dolfinx.fem.petsc as petsc
import ufl
import tqdm
import petsc4py.PETSc as PETSc
import src.config as cfg


class ISourceTerm(abc.ABC):
    @abc.abstractmethod
    def get_form(self, V: fem.FunctionSpace, v: ufl.TestFunction, omega: fem.Constant) -> ufl.Form:
        pass


class IBoundaryCondition(abc.ABC):
    @abc.abstractmethod
    def get_lhs_term(self, V: fem.FunctionSpace, u: ufl.TrialFunction, v: ufl.TestFunction, ds: ufl.Measure, k0: fem.Constant, omega: fem.Constant) -> ufl.Form:
        pass


class IProblemSolver(abc.ABC):
    @abc.abstractmethod
    def solve(self, frequency: float):
        pass


class FrequencyDomainSimulationConfig(cfg.ConfigProtocol):
    rho_0: float
    freq: float
    # Source properties
    Q: float = 0.0001
    alfa: float = 0.015 # for Gaussian source approximation


class GaussianPointSource(ISourceTerm):
    def __init__(self, config: FrequencyDomainSimulationConfig):
        self.config = config

    def get_form(self, V: fem.FunctionSpace, v: ufl.TestFunction, omega: fem.Constant) -> ufl.Form:
        cfg = self.config
        delta_tmp = fem.Function(V)
        delta_tmp.interpolate(lambda x: 1/(np.abs(cfg.alfa)*np.sqrt(np.pi)) * 
                             np.exp(-(((x[0]-cfg.Sx)**2 + (x[1]-cfg.Sy)**2)/(cfg.alfa**2))))
        int_delta = fem.assemble_scalar(fem.form(delta_tmp * ufl.dx))
        delta = delta_tmp / int_delta
        f = 1j * cfg.rho_0 * omega * cfg.Q * delta
        return ufl.inner(f, v) * ufl.dx


class ImpedanceBC(IBoundaryCondition):
    def __init__(self, config: FrequencyDomainSimulationConfig, tags: list[int]):
        self.config = config
        if not isinstance(tags, (list, tuple)):
            tags = [tags]
        self.tags = tags

    def get_lhs_term(self, V: fem.FunctionSpace, u: ufl.TrialFunction, v: ufl.TestFunction, ds: ufl.Measure, k0: fem.Constant, omega: fem.Constant) -> ufl.Form:
        cfg = self.config
        r = fem.Function(V)
        r.interpolate(lambda x: np.sqrt((x[0]-cfg.Sx)**2 + (x[1]-cfg.Sy)**2) + 1e-8)
        
        Z = cfg.rho_0 * cfg.c0 / (1 + 1/(1j * k0 * r))
        g = 1j * cfg.rho_0 * omega / Z
        
        integral_term = 0
        for tag in self.tags:
            integral_term += g * ufl.inner(u, v) * ds(tag)
        return integral_term

class HardWallBC(IBoundaryCondition):
    def get_lhs_term(self, V, u, v, ds, k0, omega) -> ufl.Form:
        zero = fem.Constant(V.mesh, PETSc.ScalarType(0.0))
        return ufl.inner(zero * u, v) * ds(1)


class HelmholtzProblemSolver(IProblemSolver):
    def __init__(self, mesh, facet_markers, source: ISourceTerm, bcs: list[IBoundaryCondition], config: FrequencyDomainSimulationConfig):
        self.mesh = mesh
        self.facet_markers = facet_markers
        self.source = source
        self.bcs = bcs
        self.config = config
        
        self.V = fem.functionspace(mesh, ("CG", config.deg))
        self.u, self.v = ufl.TrialFunction(self.V), ufl.TestFunction(self.V)
        self.ds = ufl.Measure("ds", domain=mesh, subdomain_data=facet_markers)
        self.omega = fem.Constant(mesh, PETSc.ScalarType(1))
        self.k0 = fem.Constant(mesh, PETSc.ScalarType(1))

    def solve(self, frequency: float) -> fem.Function:
        print(f"Assembling and solving for frequency: {frequency} Hz")
        self.omega.value = frequency * 2 * np.pi
        self.k0.value = self.omega.value / self.config.c0

        a = ufl.inner(ufl.grad(self.u), ufl.grad(self.v)) * ufl.dx - self.k0**2 * ufl.inner(self.u, self.v) * ufl.dx
        L = self.source.get_form(self.V, self.v, self.omega)
 
        for bc in self.bcs:
            a += bc.get_lhs_term(self.V, self.u, self.v, self.ds, self.k0, self.omega)
        
        uh = fem.Function(self.V)
        problem = petsc.LinearProblem(a, L, u=uh, petsc_options={"ksp_type": "preonly", "pc_type": "lu"})
        problem.solve()
        return uh

class PMLWaveSolver(IProblemSolver):
    def __init__(self, mesh, cell_markers, source, config):
        self.mesh, self.cell_markers, self.source, self.config = mesh, cell_markers, source, config
        cfg = config
        
        # --- 1. Define Mixed Function Space ---
        P_el = ufl.FiniteElement("Lagrange", mesh.ufl_cell(), cfg.deg)

    # Создаем смешанное пространство, просто передавая КОРТЕЖ из элементов в functionspace
        self.V = fem.functionspace(mesh, (P_el, P_el, P_el))
        
        # --- 2. Define trial and test functions ---
        # p, px, py are for step n+1
        # q, qx, qy are test functions
        (self.p, self.px, self.py) = ufl.TrialFunctions(self.V)
        (q, qx, qy) = ufl.TestFunctions(self.V)

        # --- 3. Define functions for previous time steps ---
        self.sol_n = fem.Function(self.V)   # Solution at step n
        self.sol_n_1 = fem.Function(self.V) # Solution at step n-1
        p_n, px_n, py_n = ufl.split(self.sol_n)
        p_n_1, px_n_1, py_n_1 = ufl.split(self.sol_n_1)

        # --- 4. Define PML damping function sigma ---
        # sigma is non-zero only in the PML region
        V_sigma = fem.functionspace(mesh, ("DG", 0))
        self.sigma_x = fem.Function(V_sigma)
        self.sigma_y = fem.Function(V_sigma)

        pml_cells = cell_markers.find(2) # Find all cells with tag 2 (PML)
        # Простая функция затухания, нарастающая от границ основной области
        def sigma_func(x, dim, boundary):
            val = np.zeros_like(x[0])
            dist = np.abs(x[dim] - boundary)
            val[:] = cfg.pml_sigma_max * (dist / cfg.pml_thickness)**2
            return val
        
        self.sigma_x.interpolate(lambda x: sigma_func(x, 0, 0.0), pml_cells)
        self.sigma_x.interpolate(lambda x: sigma_func(x, 0, cfg.width), pml_cells)
        self.sigma_y.interpolate(lambda x: sigma_func(x, 1, 0.0), pml_cells)
        self.y_sigma.interpolate(lambda x: sigma_func(x, 1, cfg.height), pml_cells)

        # --- 5. Define Variational Formulation ---
        c0, dt = cfg.c0, cfg.dt
        source_spatial = source.get_spatial_component()
        self.source_amplitude = fem.Constant(mesh, PETSc.ScalarType(0.0))

        # Time discretization coeffs
        c1 = (2 - dt*self.sigma_x)/(2 + dt*self.sigma_x)
        c2 = (2*dt)/(2 + dt*self.sigma_x)
        d1 = (2 - dt*self.sigma_y)/(2 + dt*self.sigma_y)
        d2 = (2*dt)/(2 + dt*self.sigma_y)
        
        # Weak form for p, px, py
        dx_measure = ufl.Measure("dx", domain=mesh)
        
        F = ( (self.p - p_n - self.px - self.py) * q * dx_measure
            + (self.px - c1*px_n - c2*c0**2 * ufl.Dx(p_n,0)*ufl.Dx(qx,0)) * qx * dx_measure
            + (self.py - d1*py_n - d2*c0**2 * ufl.Dx(p_n,1)*ufl.Dx(qy,1)) * qy * dx_measure )

        self.a, self.L = ufl.system.lhs(F), ufl.system.rhs(F)

        # Source term added to RHS 'p' equation
        self.L -= ufl.inner(self.source_amplitude * source_spatial, q) * dx_measure
        
        # --- 6. Solver Setup ---
        # (similar to before, but for a mixed system)
        self.A = fem.petsc.assemble_matrix(ufl.form(self.a))
        self.A.assemble()
        self.b = fem.petsc.create_vector(ufl.form(self.L))
        self.solver = PETSc.KSP().create(mesh.comm)
        self.solver.setOperators(self.A)
        self.solver.setType(PETSc.KSP.Type.PREONLY)
        self.solver.getPC().setType(PETSc.PC.Type.LU)
        self.sol = fem.Function(self.V)

    def solve(self, frames_to_store=20):
        # Time-stepping loop
        times = np.arange(0, self.config.t_end, self.config.dt)
        pressure_data = [] # Will store only the 'p' component
        
        # Subspace for extracting the pressure component 'p'
        V_p, _ = self.V.sub(0).collapse()

        print("Starting PML simulation...")
        for i, t in enumerate(tqdm(times)):
            self.source_amplitude.value = self.source.get_value_at_time(t)
            with self.b.localForm() as loc_b: loc_b.set(0)
            fem.petsc.assemble_vector(self.b, ufl.form(self.L))

            self.solver.solve(self.b, self.sol.vector)
            
            self.sol_n_1.x.array[:] = self.sol_n.x.array
            self.sol_n.x.array[:] = self.sol.x.array

            if i % frames_to_store == 0:
                p_component = self.sol.sub(0).collapse()
                pressure_data.append(p_component.x.array.copy())
        
        stored_times = times[::frames_to_store]
        return np.array(pressure_data), stored_times


class TimeDomainConfigProtocol(cfg.ConfigProtocol):
    t_end: float  # End time of simulation [s]
    dt: float  # Time step [s] - MUST satisfy CFL condition


class RickerPulseConfigProtocol(TimeDomainConfigProtocol):
    # Source parameters (for Ricker pulse)
    source_freq: float  # Center frequency of the Ricker pulse [Hz]
    source_delay: float  # Time delay for the pulse peak [s]


class RickerPulseSource(ISourceTerm):
    def __init__(self, V: fem.FunctionSpace, config: RickerPulseConfigProtocol):
        self.config = config
        self._spatial_component = fem.Function(V)
        alfa = 0.3
        self._spatial_component.interpolate(
            lambda x: 1/(np.abs(alfa)*np.sqrt(np.pi)) * np.exp(-(((x[0]-config.Sx)**2 + (x[1]-config.Sy)**2)/(alfa**2)))
        )
        self._spatial_component /= fem.assemble_scalar(fem.form(self._spatial_component * ufl.dx))

    def get_spatial_component(self) -> fem.Function:
        return self._spatial_component

    def get_value_at_time(self, t: float) -> float:
        cfg = self.config
        f, t0 = cfg.source_freq, cfg.source_delay
        arg = (np.pi * f * (t - t0))**2
        return (1.0 - 2.0 * arg) * np.exp(-arg)

    def get_form(self, V, v, omega):
        return super().get_form(V, v, omega)
    

class HammingPulseConfigProtocol(TimeDomainConfigProtocol):
    # Source parameters (for Hamming pulse)
    pulse_duration: float  # Hamming window duration [s]
    pulse_carrier_freq: float  # main sinusoidal frequency [Hz]


class HammingPulseSource(ISourceTerm):
    """
    Источник, модулированный окном Хэмминга.
    S(t) = sin(2*pi*f_c*t) * (0.54 - 0.46*cos(2*pi*t/T)) для 0 <= t <= T
    """
    def __init__(self, V: fem.FunctionSpace, config: HammingPulseConfigProtocol):
        self.config = config
        # Пространственная часть точно такая же, как раньше
        self._spatial_component = fem.Function(V)
        alfa = 0.015
        self._spatial_component.interpolate(
            lambda x: 1/(np.abs(alfa)*np.sqrt(np.pi)) * np.exp(-(((x[0]-config.Sx)**2 + (x[1]-config.Sy)**2)/(alfa**2)))
        )
        self._spatial_component /= fem.assemble_scalar(fem.form(self._spatial_component * ufl.dx))

    def get_spatial_component(self) -> fem.Function:
        return self._spatial_component

    def get_value_at_time(self, t: float) -> float:
        cfg = self.config
        T = cfg.pulse_duration
        
        # Если время t находится вне окна, источник равен нулю
        if not (0 <= t <= T):
            return 0.0
        
        # Несущая синусоида
        carrier = np.sin(2 * np.pi * cfg.pulse_carrier_freq * t)
        
        # Окно Хэмминга
        window = 0.54 - 0.46 * np.cos(2 * np.pi * t / T)
        
        return carrier * window
    
    def get_form(self, V, v, omega):
        return super().get_form(V, v, omega)


class TimeDomainWaveSolver(IProblemSolver):
    def __init__(self, mesh, source_term: ISourceTerm, config: TimeDomainConfigProtocol):
        self.mesh = mesh
        self.source_term = source_term
        self.config = config
        
        self.V = fem.functionspace(mesh, ("Lagrange", config.deg))
        self.p, self.p_n, self.p_n_1 = fem.Function(self.V), fem.Function(self.V), fem.Function(self.V)
        
        u, v = ufl.TrialFunction(self.V), ufl.TestFunction(self.V)
        c0, dt = config.c0, config.dt
        
        self.source_amplitude = fem.Constant(mesh, PETSc.ScalarType(0.0))
        self.source_spatial = source_term.get_spatial_component()

        self.a = ufl.inner(u, v) * ufl.dx
        L_form = (ufl.inner(2 * self.p_n - self.p_n_1, v) * ufl.dx
                  - (c0 * dt)**2 * ufl.inner(ufl.grad(self.p_n), ufl.grad(v)) * ufl.dx
                  + (c0 * dt)**2 * ufl.inner(self.source_amplitude * self.source_spatial, v) * ufl.dx)
        self.L = fem.form(L_form)

        self.A = petsc.assemble_matrix(fem.form(self.a))
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
        for i, t in enumerate(tqdm.tqdm(times)):
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
