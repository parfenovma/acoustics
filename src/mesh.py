import abc
import numpy as np
import gmsh
import dolfinx
import src.config as cfg
import dolfinx.io as io
import mpi4py.MPI as MPI


class IMeshGenerator(abc.ABC):
    @abc.abstractmethod
    def generate(self) -> tuple[dolfinx.mesh.Mesh, ...]:
        pass


class GmshChannelMesh(IMeshGenerator):
    def __init__(self, config: cfg.ConfigProtocol):
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