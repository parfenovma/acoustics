# Some scripts for acoustic wave modeling
For now it might be useful for modeling acoustic waves in partially limited environment.
Under the hood it's a [fenicsx project](https://fenicsproject.org/) backends and gmsh for mesh generation.
time domain and frequency domain animations is supported

how to setup env?

```bash
conda init
conda create -n env3-10-complex -c conda-forge python=3.10 fenics-dolfinx petsc=*=complex* mpich
conda activate env3-10-complex
conda install -c conda-forge ipykernel
python -m ipykernel install --user --name=env3-10-complex
pip install gmsh matplotlib
sudo apt-get -y install gmsh
export PYTHONPATH=path/to/your/directory/acoustics
```

how to run?

*Currently I've tried only one-thread mode, and get SEGFAULT in other cases.*
```bash
mpirun -n 1 python3 -m src
```


## notes
1) You can watch my tries to structure this project properly via commit history )
2) To contribute, installing git-lfs is needed


# Concept
## /src
Directory for source code. Supposed to run as a module.
### config.py
Contains configuration for the project as a set of dataclasses. If needed, you can add your configuration here, and mark it as a `typing.Protocol` in `problem.py` for your problem (ex. for a 3D problem). For now, common protocol is defined in `config.py`(`ConfigProtocol`)
### mesh.py
Contains Mesh generators. Mesh generator must implement `IMeshGenerator` ABC with the only public methiod `ImeshGenerator.generate()`. 
### solver.py
Contains solver for the problem. Solver must implement `ISolver` ABC with the only public methiod `ISolver.solve()`. I am currently not sure what return type for `ISolver.solve()` should be, so this behaviour may change in the future.
### visualizer.py
This abstraction is actually very useful, mostly because you want to be able to change the visualization and compare different animation methods. Visualizer must implement `IVisualizer` ABC with the only public methiod `IVisualizer.create_animation()` and `IVisualizer.create_static_plot()`. Good idea is to add more flexibility to an interface.
## /pic
Directory for pictures / plots.
## /animations
Directory for animations. Animations may weight more than 50MiB, for now it's about 70-80 Mib (and we're going to animate in 3D soon!)
In that case, using git-lfs is highly recommended.