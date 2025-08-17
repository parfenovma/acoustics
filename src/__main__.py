import gmsh
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
    
    pressure_history, times_stored = solver.solve(frames_to_store=10) # Store every 2nd frame
    
    visualizer: vis.IVisualizer = vis.MatplotlibTimeVisualizer(mesh, config)
    visualizer.create_animation(pressure_history, times_stored, "animations/initial_wave_hamming_modified_visual.gif")
    
    plt.show()


def main_w_pml():
    # --- Case 1: All boundaries are absorbing ---
    print("\n--- Running Case 1: Full Absorbing Boundaries ---")
    config1 = cfg.TimeDomainSimulationConfig()
    
    mesh_gen1 = msh.GmshBoundaryTaggedMesh(config1)
    mesh1, facet_markers1 = mesh_gen1.generate()
    V_space1 = fem.functionspace(mesh1, ("Lagrange", config1.deg))
    source1 = problem.HammingPulseSource(V_space1, config1)
    
    # Применяем ГУ ко всем 4 границам (теги 1, 2, 3, 4)
    solver1 = problem.ImpedanceBCSolver(mesh1, facet_markers1, source1, config1, bc_tags=[1, 2, 3, 4])
    
    pressure_history1, times1 = solver1.solve(frames_to_store=20)
    visualizer1 = vis.MatplotlibTimeVisualizer(mesh1, config1)
    visualizer1.create_animation(pressure_history1, times1, "animations/impedance_full.gif")
    plt.show()

    # --- Case 2: Waveguide with absorbing ends ---
    print("\n--- Running Case 2: Waveguide with Absorbing Ends ---")
    config2 = cfg.TimeDomainSimulationConfig()
    
    mesh_gen2 = msh.GmshBoundaryTaggedMesh(config2)
    mesh2, facet_markers2 = mesh_gen2.generate()
    V_space2 = fem.functionspace(mesh2, ("Lagrange", config2.deg))
    source2 = problem.HammingPulseSource(V_space2, config2)
    
    # Применяем ГУ только к левой и правой границам (теги 2, 4)
    # Верх и низ (1, 3) останутся жесткими (естественное ГУ)
    solver2 = problem.ImpedanceBCSolver(mesh2, facet_markers2, source2, config2, bc_tags=[2, 4])

    pressure_history2, times2 = solver2.solve(20)
    visualizer2 = vis.MatplotlibTimeVisualizer(mesh2, config2)
    visualizer2.create_animation(pressure_history2, times2, "animations/impedance_waveguide.gif")
    plt.show()

if __name__ == "__main__":
    # main()
    main_w_pml()
