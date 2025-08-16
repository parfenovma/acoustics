import gmsh
import dolfinx
import dolfinx.fem as fem
import src.mesh as msh
import src.problem as problem
import src.visualizer as vis
import src.config as cfg
import matplotlib.pyplot as plt


def main():
    config: cfg.ConfigProtocol = cfg.TimeDomainSimulationConfig(
        # t_end=0.001,       # Stop before the first reflection
        # dt=2e-6,           # Keep dt small for stability
        # source_delay=0.0003  # Make the pulse appear earlier
    )

    h_min = 1.0 / config.n_elem
    cfl = config.c0 * config.dt / h_min
    print(f"Mesh size (h): ~{h_min:.4f} m"); print(f"CFL number (estimated): {cfl:.1f}")
    if cfl > 0.5:
        print("WARNING: Estimated CFL > 0.5. Simulation might be unstable.")

    mesh_generator: msh.IMeshGenerator = msh.GmshChannelMesh(config)
    mesh, _ = mesh_generator.generate()
    
    V_solver_space = fem.functionspace(mesh, ("Lagrange", config.deg))
    source: problem.ISourceTerm = problem.HammingPulseSource(V_solver_space, config)
    
    solver: problem.IProblemSolver = problem.TimeDomainWaveSolver(mesh, source, config)
    
    pressure_history, times_stored = solver.solve(frames_to_store=2) # Store every 2nd frame
    
    visualizer: vis.IVisualizer = vis.MatplotlibTimeVisualizer(mesh, config)
    visualizer.create_animation(pressure_history, times_stored, "animations/initial_wave_hamming_modified.gif")
    
    plt.show()


if __name__ == "__main__":
    main()