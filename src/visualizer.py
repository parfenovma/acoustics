import abc
import dolfinx.fem as fem
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.tri as tri
import matplotlib.animation as animation
import src.config as cfg

class IVisualizer(abc.ABC):
    @abc.abstractmethod
    def create_animation(self, output_filename: str = None):
        pass

    @abc.abstractmethod
    def create_static_plot(self, output_filename: str = None):
        pass


class MatplotlibVisualizer(IVisualizer):
    def __init__(self, mesh, solution, config):
        self.mesh = mesh
        self.uh = solution
        self.config = config
        # Use a P1 projection for robust visualization
        V_vis = fem.functionspace(mesh, ("Lagrange", 1))
        p_vis = fem.Function(V_vis)
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


class MatplotlibTimeVisualizer(IVisualizer):
    def __init__(self, mesh, config: cfg.ConfigProtocol):
        self.mesh = mesh
        self.config = config

        # Setup for visualization (P1 space)
        V_vis = fem.functionspace(mesh, ("Lagrange", 1))
        self.V_sol = fem.functionspace(mesh, ("Lagrange", config.deg))
        vertex_coords = V_vis.tabulate_dof_coordinates()
        self.triang = tri.Triangulation(vertex_coords[:, 0], vertex_coords[:, 1])

        # We need a P1 function to project the high-order solution onto for visualization
        self.p_vis = fem.Function(V_vis)
        self.deg = config.deg

    def create_animation(self, pressure_over_time: np.ndarray, times: np.ndarray, output_filename=None):
        fig, ax = plt.subplots(figsize=(8, 6))
        
        # --- DEBUG FIX 2: Let matplotlib choose the color scale for each frame ---
        abs_pressures = np.abs(pressure_over_time.flatten())
        vmax = np.percentile(abs_pressures, 99.5)
        
        print(f"Robust vmax calculated using 99.5th percentile: {vmax:.2e}")
        vmax = 0.1
        vmin = -vmax
        
        p_sol_frame = fem.Function(self.V_sol)

        def update(frame):
            ax.clear()
            self._setup_plot_aesthetics(ax)
            p_sol_frame.x.array[:] = pressure_over_time[frame]
            self.p_vis.interpolate(p_sol_frame)

            ax.tricontourf(self.triang, self.p_vis.x.array, levels=40, cmap='RdBu_r', vmax=vmax, vmin=vmin)
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
