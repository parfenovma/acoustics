# Some scripts for acoustic wave modeling
For now it might be useful for modeling acoustic waves in partially limited environment.
Under the hood it's a [fenicsx project](https://fenicsproject.org/) backends and gmsh for mesh generation.


how to setup env?

```bash
conda init
conda create -n env3-10-complex -c conda-forge python=3.10 fenics-dolfinx petsc=*=complex* mpich
conda activate env3-10-complex
conda install -c conda-forge ipykernel
python -m ipykernel install --user --name=env3-10-complex
pip install gmsh matplotlib
sudo apt-get -y install gmsh
```

how to run?

```bash
mpirun -n 1 python3 hard_horizontal.py
```

## notes
1) Current version is draft, I will refactor it eventially (at least I hope so). The most useful script is currently in `/exp/general_abstractions.py`
2) To contribute, installing git-lfs is needed