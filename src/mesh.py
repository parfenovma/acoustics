import abc
import enum
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


class PMLConfigProtocol(cfg.ConfigProtocol):
    pml_thickness: float
    pml_sigma_max: float

class EPMLBoundaries(enum.Enum):
    BOTTOM = "bottom"
    TOP = "top"
    LEFT = "left"
    RIGHT = "right"


class GmshMeshWithPML(IMeshGenerator):
    def __init__(self, config: PMLConfigProtocol, pml_boundaries: list[EPMLBoundaries]):
        """
        Args:
            pml_boundaries: A list of strings, e.g., ["left", "right", "top", "bottom"]
        """
        self.config = config
        self.pml_boundaries = set(pml_boundaries)

    def generate(self):
        cfg = self.config
        L_pml = cfg.pml_thickness
        gmsh.initialize()
        gmsh.model.add("pml_mesh")
        
        # Tags for physical groups
        DOMAIN_TAG, PML_TAG = 1, 2

        # Main domain
        main_rect = gmsh.model.occ.addRectangle(0, 0, 0, cfg.width, cfg.height)
        
        # Create PML layers
        pml_rects = []
        if "bottom" in self.pml_boundaries:
            pml_rects.append(gmsh.model.occ.addRectangle(0, -L_pml, 0, cfg.width, L_pml))
        if "top" in self.pml_boundaries:
            pml_rects.append(gmsh.model.occ.addRectangle(0, cfg.height, 0, cfg.width, L_pml))
        if "left" in self.pml_boundaries:
            pml_rects.append(gmsh.model.occ.addRectangle(-L_pml, -L_pml, 0, L_pml, cfg.height + 2*L_pml if "top" in self.pml_boundaries and "bottom" in self.pml_boundaries else cfg.height + L_pml if "top" in self.pml_boundaries else cfg.height ))
        if "right" in self.pml_boundaries:
            pml_rects.append(gmsh.model.occ.addRectangle(cfg.width, -L_pml if "bottom" in self.pml_boundaries else 0, 0, L_pml, cfg.height + 2*L_pml if "top" in self.pml_boundaries and "bottom" in self.pml_boundaries else cfg.height + L_pml if "top" in self.pml_boundaries else cfg.height))
        
        #TODO@parfenovma: refactor this part, may be some artifacts in corners
        all_shapes = [(2, main_rect)] + [(2, r) for r in pml_rects]
        gmsh.model.occ.fragment(all_shapes, [])
        gmsh.model.occ.synchronize()

        # Find and tag regions
        domain_center = (cfg.width/2, cfg.height/2, 0)
        vols = gmsh.model.getEntities(dim=2)
        main_domain_tag = -1
        pml_domain_tags = []
        for vol in vols:
            com = gmsh.model.occ.getCenterOfMass(vol[0], vol[1])
            if np.allclose(com, domain_center, atol=cfg.width/2):
                main_domain_tag = vol[1]
            else:
                pml_domain_tags.append(vol[1])

        gmsh.model.addPhysicalGroup(2, [main_domain_tag], DOMAIN_TAG)
        if pml_domain_tags:
            gmsh.model.addPhysicalGroup(2, pml_domain_tags, PML_TAG)

        # Meshing
        lc = 1.0 / cfg.n_elem
        gmsh.option.setNumber("Mesh.CharacteristicLengthMax", lc)
        gmsh.model.mesh.generate(2)
        
        msh, cell_markers, _ = io.gmshio.model_to_mesh(gmsh.model, MPI.COMM_WORLD, 0, gdim=2)
        gmsh.finalize()
        return msh, cell_markers


class GmshBoundaryTaggedMesh(IMeshGenerator):
    def __init__(self, config: PMLConfigProtocol):
        self.config = config

    def generate(self):
        cfg = self.config
        gmsh.initialize()
        gmsh.model.add("boundary_tagged_mesh")
        
        # Создаем геометрию
        p1 = gmsh.model.occ.addPoint(0, 0, 0)
        p2 = gmsh.model.occ.addPoint(cfg.width, 0, 0)
        p3 = gmsh.model.occ.addPoint(cfg.width, cfg.height, 0)
        p4 = gmsh.model.occ.addPoint(0, cfg.height, 0)
        
        l_bottom = gmsh.model.occ.addLine(p1, p2)
        l_right = gmsh.model.occ.addLine(p2, p3)
        l_top = gmsh.model.occ.addLine(p3, p4)
        l_left = gmsh.model.occ.addLine(p4, p1)
        
        curve_loop = gmsh.model.occ.addCurveLoop([l_bottom, l_right, l_top, l_left])
        surface = gmsh.model.occ.addPlaneSurface([curve_loop])
        gmsh.model.occ.synchronize()

        # Создаем физические группы для границ и домена
        # Теги: 1-низ, 2-право, 3-верх, 4-лево, 5-домен
        gmsh.model.addPhysicalGroup(1, [l_bottom], 1)
        gmsh.model.addPhysicalGroup(1, [l_right], 2)
        gmsh.model.addPhysicalGroup(1, [l_top], 3)
        gmsh.model.addPhysicalGroup(1, [l_left], 4)
        gmsh.model.addPhysicalGroup(2, [surface], 5)

        # Меширование
        lc = 1.0 / cfg.n_elem
        gmsh.option.setNumber("Mesh.CharacteristicLengthMax", lc)
        gmsh.model.mesh.generate(2)
        
        # Импорт в dolfinx
        msh, _, facet_markers = io.gmshio.model_to_mesh(gmsh.model, MPI.COMM_WORLD, 0, gdim=2)
        gmsh.finalize()
        return msh, facet_markers