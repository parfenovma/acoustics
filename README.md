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

### New version:
```bash
sudo add-apt-repository ppa:fenics-packages/fenics;
sudo apt update;
sudo apt install fenicsx;
```
**For some reasons, versions of fenicsx, distributed via pip and conda, are deprecated now, so let's build them locally!**

*How hard can it be?*


basix
```bash
python3 -m venv venv
source venv/bin/activate
python3 -m pip install git+https://github.com/FEniCS/basix.git@main
```

also, we need to install basicx .so version (to build dolfinx)

```bash
git clone https://github.com/FEniCS/basix.git basicx
cd basicx/cpp/
cmake -DCMAKE_BUILD_TYPE=Release -B build-dir -S .
cmake --build build-dir
sudo cmake --install build-dir
```

ufl
```bash
python3 -m pip install git+https://github.com/FEniCS/ufl.git@main
```

ffcx
```bash
python3 -m pip install git+https://github.com/FEniCS/ffcx.git@main
```

dolfinx
```bash
git clone https://github.com/FEniCS/dolfinx.git dolfinx
cd dolfinx/cpp
mkdir build
cd build
cmake ..
sudo make install
source /usr/local/lib/dolfinx/dolfinx.conf
cd ../../python/
python -m pip install petsc4py mpi4py
pip install .
```


how to run?

**Currently I've tried only one-thread mode, and get SEGFAULT in other cases.**
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
