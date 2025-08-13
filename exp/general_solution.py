import numpy as np
import ufl
from dolfinx import geometry, fem
from dolfinx.fem import Function, functionspace, assemble_scalar, form, Constant
from dolfinx.fem.petsc import LinearProblem
from ufl import dx, grad, inner, ds
from mpi4py import MPI
from petsc4py import PETSc
import matplotlib.pyplot as plt
import matplotlib.tri as tri
import matplotlib.animation as animation
import gmsh
from dolfinx.io import gmshio


class WavyChannelMesh:
    """Класс для генерации сетки канала с волнообразными границами."""
    
    def __init__(self, width=1.0, height=1.0, amplitude=0.05, num_waves=3,
                 num_points_boundary=50, n_elem=64):
        self.width = width
        self.height = height
        self.amplitude = amplitude
        self.num_waves = num_waves
        self.num_points_boundary = num_points_boundary
        self.n_elem = n_elem
        
    def generate(self):
        """Генерация сетки с помощью gmsh."""
        gmsh.initialize()
        gmsh.model.add("wavy_channel")
        
        # Генерация точек для нижней и верхней границ
        bottom_points = self._create_boundary_points(
            lambda x: self.amplitude * np.sin(2 * np.pi * self.num_waves * x / self.width)
        )
        top_points = self._create_boundary_points(
            lambda x: self.height + self.amplitude * np.sin(2 * np.pi * self.num_waves * x / self.width)
        )
        
        # Создание геометрии
        bottom_spline = gmsh.model.geo.addSpline(bottom_points)
        top_spline = gmsh.model.geo.addSpline(top_points)
        left_line = gmsh.model.geo.addLine(bottom_points[0], top_points[0])
        right_line = gmsh.model.geo.addLine(bottom_points[-1], top_points[-1])
        
        loop = gmsh.model.geo.addCurveLoop([bottom_spline, right_line, -top_spline, -left_line])
        surface = gmsh.model.geo.addPlaneSurface([loop])
        
        gmsh.model.geo.synchronize()
        
        # Физические группы
        gmsh.model.addPhysicalGroup(1, [bottom_spline], 1)  # Нижняя граница
        gmsh.model.addPhysicalGroup(1, [top_spline], 3)     # Верхняя граница
        gmsh.model.addPhysicalGroup(1, [left_line], 4)      # Левая граница
        gmsh.model.addPhysicalGroup(1, [right_line], 2)     # Правая граница
        gmsh.model.addPhysicalGroup(2, [surface], 1)        # Область
        
        # Параметры сетки
        lc = self.width / self.n_elem
        gmsh.option.setNumber("Mesh.CharacteristicLengthMin", lc)
        gmsh.option.setNumber("Mesh.CharacteristicLengthMax", lc)
        gmsh.model.mesh.generate(2)
        
        # Импорт в dolfinx
        msh, cell_markers, facet_markers = gmshio.model_to_mesh(
            gmsh.model, MPI.COMM_WORLD, 0, gdim=2
        )
        gmsh.finalize()
        
        return msh, facet_markers
    
    def _create_boundary_points(self, y_func):
        """Создание точек вдоль границы с заданной функцией высоты."""
        points = []
        for i in range(self.num_points_boundary + 1):
            x = self.width * i / self.num_points_boundary
            y = y_func(x)
            p = gmsh.model.geo.addPoint(x, y, 0)
            points.append(p)
        return points


class AcousticProblem:
    """Класс для решения акустической задачи в канале."""
    
    def __init__(self, mesh, facet_markers, deg=2, Sx=0.1, Sy=0.5):
        self.mesh = mesh
        self.facet_markers = facet_markers
        self.V = functionspace(mesh, ("CG", deg))
        self.u = ufl.TrialFunction(self.V)
        self.v = ufl.TestFunction(self.V)
        
        # Физические параметры
        self.c0 = 340.0        # Скорость звука [м/с]
        self.rho_0 = 1.225     # Плотность воздуха [кг/м³]
        
        # Параметры источника
        self.Sx, self.Sy = Sx, Sy  # Положение источника
        self.Q = 0.0001             # Амплитуда источника
        
        # Частота и волновое число
        self.omega = Constant(mesh, PETSc.ScalarType(1))
        self.k0 = Constant(mesh, PETSc.ScalarType(1))
        
        # Граничные меры
        self.ds_impedance = ufl.Measure("ds", domain=mesh, subdomain_data=facet_markers)((2, 4))
        
        # Инициализация функций
        self._initialize_functions()
    
    def _initialize_functions(self):
        """Инициализация функций для источника и импеданса."""
        # Аппроксимация дельта-функции
        alfa = 0.015
        delta_tmp = Function(self.V)
        delta_tmp.interpolate(lambda x: 1/(np.abs(alfa)*np.sqrt(np.pi)) * 
                             np.exp(-(((x[0]-self.Sx)**2 + (x[1]-self.Sy)**2)/(alfa**2))))
        int_delta = assemble_scalar(form(delta_tmp * dx))
        self.delta = delta_tmp / int_delta
        
        # Расстояние от источника
        r = Function(self.V)
        r.interpolate(lambda x: np.sqrt((x[0]-self.Sx)**2 + (x[1]-self.Sy)**2) + 1e-8)
        self.Z = self.rho_0 * self.c0 / (1 + 1/(1j * self.k0 * r))
    
    def solve_at_frequency(self, freq):
        """Решение задачи на заданной частоте."""
        print(f"Computing for frequency: {freq} Hz")
        
        # Установка параметров частоты
        self.omega.value = freq * 2 * np.pi
        self.k0.value = 2 * np.pi * freq / self.c0
        
        # Правая часть и коэффициенты
        f = 1j * self.rho_0 * self.omega * self.Q * self.delta
        g = 1j * self.rho_0 * self.omega / self.Z
        
        # Формулировка задачи
        a = inner(grad(self.u), grad(self.v)) * dx - self.k0**2 * inner(self.u, self.v) * dx + \
            g * inner(self.u, self.v) * self.ds_impedance
        L = inner(f, self.v) * dx
        
        # Решение
        uh = Function(self.V)
        problem = LinearProblem(a, L, u=uh, petsc_options={"ksp_type": "preonly", "pc_type": "lu"})
        problem.solve()
        
        return uh


class Visualizer:
    """Класс для визуализации результатов."""
    
    def __init__(self, mesh, solution_function, params):
        self.mesh = mesh
        self.uh = solution_function
        self.params = params
        
        # Извлечение данных для визуализации
        self.vertex_coords = mesh.geometry.x
        self.dof_coords = solution_function.function_space.tabulate_dof_coordinates()
        self.p_complex = self._get_solution_at_vertices()
        
        # Создание триангуляции
        topology = mesh.topology
        topology.create_connectivity(topology.dim, 0)
        triangles = topology.connectivity(topology.dim, 0).array.reshape(-1, 3)
        self.triang = tri.Triangulation(
            self.vertex_coords[:, 0],
            self.vertex_coords[:, 1],
            triangles
        )
        
        # Параметры границ
        self.x_bound = np.linspace(0, 1, 100)
        self.y_bottom = params['A'] * np.sin(2 * np.pi * params['num_waves'] * self.x_bound / params['width'])
        self.y_top = params['height'] + params['A'] * np.sin(2 * np.pi * params['num_waves'] * self.x_bound / params['width'])
        
        y_bottom_min = np.min(self.y_bottom) - 0.05
        y_top_max = np.max(self.y_top) + 0.05
        self.ylim = (y_bottom_min, y_top_max)
    
    def _get_solution_at_vertices(self):
        """Получение значений решения в вершинах сетки."""
        dof_idx_on_vertex = []
        for v in self.vertex_coords:
            distances = np.linalg.norm(self.dof_coords - v, axis=1)
            closest_dof = np.argmin(distances)
            dof_idx_on_vertex.append(closest_dof)
        
        return self.uh.x.array[np.array(dof_idx_on_vertex)]
    
    def create_animation(self, freq, num_frames=30, output_filename=None):
        """Создание анимации распределения давления."""
        phases = np.linspace(0, 2*np.pi, num_frames, endpoint=False)
        abs_max = np.abs(self.p_complex).max()
        
        # Настройка фигуры
        fig, ax = plt.subplots(figsize=(8, 6))
        ax.set_xlim(0, 1)
        ax.set_ylim(*self.ylim)
        ax.set_aspect('equal')
        ax.set_xlabel('X [m]')
        ax.set_ylabel('Y [m]')
        
        # Добавление границ и элементов
        self._add_boundaries_and_elements(ax, freq)
        
        # Начальные данные
        p_real0 = np.real(self.p_complex * np.exp(1j * phases[0]))
        
        # Создаём контур
        contour = ax.tricontourf(self.triang, p_real0, levels=50, cmap='RdBu_r', 
                                vmin=-abs_max, vmax=abs_max)
        cbar = fig.colorbar(contour, ax=ax, shrink=0.8)
        cbar.set_label('Pressure [Pa]')
        
        # Заголовок
        title = ax.set_title(f'Pressure at {freq} Hz, Phase: 0.00π rad\nWavy Hard Boundaries Top/Bottom')
        
        def update(frame):
            phase = phases[frame]
            p_real = np.real(self.p_complex * np.exp(1j * phase))
            contour.set_array(p_real)  # Обновляем данные
            title.set_text(f'Pressure at {freq} Hz, Phase: {phase/np.pi:.2f}π rad\nWavy Hard Boundaries Top/Bottom')
            return [contour, title]
        
        # Создаём анимацию
        ani = animation.FuncAnimation(
            fig, update, frames=num_frames, interval=100, repeat=True, blit=False
        )
        
        # Прогрев: вызываем первые несколько кадров, чтобы заполнить _frames
        for i in range(min(3, num_frames)):
            update(i)
        
        if output_filename:
            ani.save(output_filename, writer='pillow', fps=10, dpi=100)
            print(f"GIF saved as {output_filename}")
        
        return ani, fig
    
    def create_static_plot(self, freq, output_filename=None):
        """Создание статического графика распределения давления."""
        fig, ax = plt.subplots(figsize=(8, 6))
        
        p_real = np.real(self.p_complex)
        contour = ax.tricontourf(self.triang, p_real, levels=50, cmap='RdBu_r')
        plt.colorbar(contour, ax=ax, label='Pressure [Pa]')
        
        # Добавление границ и элементов
        self._add_boundaries_and_elements(ax, freq)
        
        ax.set_title(f'Pressure Distribution at {freq} Hz (Wavy Hard Boundaries)')
        ax.set_xlabel('X [m]')
        ax.set_ylabel('Y [m]')
        ax.set_xlim(0, 1)
        ax.set_ylim(*self.ylim)
        ax.legend()
        
        if output_filename:
            plt.savefig(output_filename, dpi=300, bbox_inches='tight')
            print(f"Static plot saved as {output_filename}")
        
        return fig
    
    def _add_boundaries_and_elements(self, ax, freq):
        """Добавление границ и элементов на график."""
        # Источник и микрофон
        ax.plot(self.params['Sx'], self.params['Sy'], 'ro', markersize=8, label='Source')
        ax.plot(0.9, 0.9, 'go', markersize=8, label='Mic')
        
        # Волнообразные границы
        ax.plot(self.x_bound, self.y_bottom, 'k-', linewidth=2, label='Wavy Hard Boundary')
        ax.plot(self.x_bound, self.y_top, 'k-', linewidth=2)
        
        # Импедансные границы
        ax.plot([0, 0], [self.y_bottom[0], self.y_top[0]], 'b-', linewidth=1, label='Impedance Boundary')
        ax.plot([1, 1], [self.y_bottom[-1], self.y_top[-1]], 'b-', linewidth=1)


def main():
    """Основная функция для запуска симуляции."""
    # Параметры
    params = {
        'width': 1.0,
        'height': 1.0,
        'A': 0.1,           # Амплитуда волны
        'num_waves': 3,      # Количество волн
        'num_points_boundary': 50,
        'n_elem': 64,
        'deg': 2,
        'freq': 800.0,       # Частота в Гц
        'Sx': 0.1,           # Координата X источника
        'Sy': 0.5,           # Координата Y источника
    }
    
    # Генерация сетки
    mesh_generator = WavyChannelMesh(**{k: v for k, v in params.items() if k in [
        'width', 'height', 'amplitude', 'num_waves', 'num_points_boundary', 'n_elem'
    ]})
    mesh, facet_markers = mesh_generator.generate()
    
    # Решение акустической задачи
    problem = AcousticProblem(mesh, facet_markers, deg=params['deg'], Sx=params['Sx'], Sy=params['Sy'])
    solution = problem.solve_at_frequency(params['freq'])
    
    # Визуализация
    visualizer = Visualizer(mesh, solution, params)
    
    # Создание анимации
    gif_filename = f"pressure_field_wavy_boundaries_{params['freq']}Hz.gif"
    ani, anim_fig = visualizer.create_animation(params['freq'], output_filename=gif_filename)
    
    # Создание статического графика
    static_filename = f"pressure_wavy_boundaries_{params['freq']}Hz.png"
    static_fig = visualizer.create_static_plot(params['freq'], output_filename=static_filename)
    
    plt.show()
    
    return {
        'mesh': mesh,
        'solution': solution,
        'visualizer': visualizer,
        'animation': ani,
        'figures': {'animation': anim_fig, 'static': static_fig}
    }


if __name__ == "__main__":
    result = main()
