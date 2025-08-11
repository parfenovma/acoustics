import numpy as np
import time
import matplotlib.pyplot as plt
import matplotlib.tri as tri
import matplotlib.animation as animation
import gmsh

import ufl
from ufl import dx, inner, grad, Measure

import dolfinx
from dolfinx import fem
from dolfinx.fem import (
    Function,
    functionspace,
    assemble_scalar,
    form,
    Constant,
)
from dolfinx.fem.petsc import LinearProblem
from dolfinx.io import gmshio
from dolfinx.mesh import meshtags

from mpi4py import MPI
from petsc4py import PETSc


# ---------------------------
# Геометрия и генерация сетки
# ---------------------------
def build_wavy_channel_mesh(
    width: float,
    height: float,
    A: float,
    num_waves: int,
    n_elem: int,
    num_points_boundary: int,
    comm=MPI.COMM_WORLD,
):
    """
    Создает двумерную сетку канала с волнообразными верхней/нижней границами в Gmsh
    и переносит ее в dolfinx. Возвращает (mesh, cell_markers, facet_markers).
    Метки граней:
      1: нижняя, 3: верхняя, 4: левая, 2: правая. Область: 1
    """
    rank = comm.rank
    if rank == 0:
        print("Generating mesh with Gmsh...")

    gmsh.initialize()
    try:
        gmsh.model.add("wavy_channel")

        # Точки нижней границы
        bottom_points = []
        for i in range(num_points_boundary + 1):
            x = width * i / num_points_boundary
            y = A * np.sin(2 * np.pi * num_waves * x / width)
            p = gmsh.model.geo.addPoint(x, y, 0)
            bottom_points.append(p)

        # Точки верхней границы
        top_points = []
        for i in range(num_points_boundary + 1):
            x = width * i / num_points_boundary
            y = height + A * np.sin(2 * np.pi * num_waves * x / width)
            p = gmsh.model.geo.addPoint(x, y, 0)
            top_points.append(p)

        # Сплайны для границ
        bottom_spline = gmsh.model.geo.addSpline(bottom_points)
        top_spline = gmsh.model.geo.addSpline(top_points)

        # Левые и правые границы
        left_line = gmsh.model.geo.addLine(bottom_points[0], top_points[0])
        right_line = gmsh.model.geo.addLine(bottom_points[-1], top_points[-1])

        # Петля и поверхность
        loop = gmsh.model.geo.addCurveLoop([bottom_spline, right_line, -top_spline, -left_line])
        surface = gmsh.model.geo.addPlaneSurface([loop])

        gmsh.model.geo.synchronize()

        # Физические группы (метки)
        gmsh.model.addPhysicalGroup(1, [bottom_spline], 1)  # Нижняя (1)
        gmsh.model.addPhysicalGroup(1, [top_spline], 3)     # Верхняя (3)
        gmsh.model.addPhysicalGroup(1, [left_line], 4)      # Левая (4)
        gmsh.model.addPhysicalGroup(1, [right_line], 2)     # Правая (2)
        gmsh.model.addPhysicalGroup(2, [surface], 1)        # Область

        # Настройка размера элемента и генерация сетки
        lc = width / n_elem
        gmsh.option.setNumber("Mesh.CharacteristicLengthMin", lc)
        gmsh.option.setNumber("Mesh.CharacteristicLengthMax", lc)
        gmsh.model.mesh.generate(2)

        mesh, cell_markers, facet_markers = gmshio.model_to_mesh(
            gmsh.model, comm, 0, gdim=2
        )
    finally:
        gmsh.finalize()

    if rank == 0:
        print("Mesh generated.")
    return mesh, cell_markers, facet_markers


# ---------------------------
# Подготовка функционального пространства и параметров
# ---------------------------
def setup_function_space(mesh, degree=2):
    V = functionspace(mesh, ("CG", degree))
    u = ufl.TrialFunction(V)
    v = ufl.TestFunction(V)
    return V, u, v


def gaussian_delta_on_mesh(V, Sx, Sy, sigma, comm=MPI.COMM_WORLD):
    """
    Строит нормированную 2D гауссову "дельта"-аппроксимацию на сетке:
    delta(x) ~ N((Sx, Sy), sigma). Нормировка делается численно через интеграл.
    """
    delta_tmp = Function(V, name="delta_tmp")

    def gauss(x):
        return 1.0 / (np.abs(sigma) * np.sqrt(np.pi)) * np.exp(
            -(((x[0] - Sx) ** 2 + (x[1] - Sy) ** 2) / (sigma**2))
        )

    delta_tmp.interpolate(gauss)
    int_delta_tmp = assemble_scalar(form(delta_tmp * dx))
    # На случай параллельного запуска, собрать интеграл со всех процессов
    total_int = comm.allreduce(int_delta_tmp, op=MPI.SUM)

    delta = Function(V, name="delta")
    delta.x.array[:] = delta_tmp.x.array / total_int
    return delta


def impedance_coefficient(V, Sx, Sy, rho0, c0, k0: Constant):
    """
    Формирует UFL-выражение импедансного коэффициента g = i*rho0*omega / Z,
    где Z = rho0*c0 / (1 + 1/(i*k0*r)), r=|x-S|.
    Здесь возвращаем кортеж (r, Z_expr), чтобы затем собрать g, имея omega.
    """
    # Радиальная функция r(x)
    r_fun = Function(V, name="r")

    def r_eval(x):
        return np.sqrt((x[0] - Sx) ** 2 + (x[1] - Sy) ** 2) + 1e-8

    r_fun.interpolate(r_eval)

    # Импеданс Z(x)
    Z_expr = rho0 * c0 / (1 + 1 / (1j * k0 * r_fun))
    return r_fun, Z_expr


# ---------------------------
# Постановка и решение
# ---------------------------
def solve_helholtz_pressure(
    mesh,
    facet_markers: meshtags,
    V,
    u, v,
    freq_hz: float,
    source_amp_Q: float,
    delta_fun: Function,
    rho0: float,
    c0: float,
    Z_expr,
    left_right_markers=(4, 2),
    ksp_options=None,
):
    """
    Решает задачу акустического давления для заданной частоты
    с импедансом на левой/правой границах и жесткими (Неймана) сверху/снизу.
    """
    # Константы
    omega = Constant(mesh, PETSc.ScalarType(2 * np.pi * freq_hz))
    k0 = Constant(mesh, PETSc.ScalarType(2 * np.pi * freq_hz / c0))

    # Источник: f = i * rho0 * omega * Q * delta
    f = (1j) * rho0 * omega * source_amp_Q * delta_fun

    # g = i * rho0 * omega / Z
    g = (1j) * rho0 * omega / Z_expr

    # Мера по границам
    ds = Measure("ds", domain=mesh, subdomain_data=facet_markers)
    ds_impedance = ds(left_right_markers[0]) + ds(left_right_markers[1])

    # Вариационная постановка: 
    # a(u, v) = (grad u, grad v) - k0^2 (u, v) + g (u, v)|_{impedance}
    # L(v) = (f, v)
    a = inner(grad(u), grad(v)) * dx - (k0**2) * inner(u, v) * dx + g * inner(u, v) * ds_impedance
    L_form = inner(f, v) * dx

    # Решение
    uh = Function(V, name="pressure")
    default_opts = {"ksp_type": "preonly", "pc_type": "lu"}
    if ksp_options:
        default_opts.update(ksp_options)

    problem = LinearProblem(a, L_form, u=uh, petsc_options=default_opts)
    problem.solve()
    return uh, omega, k0


# ---------------------------
# Постобработка и визуализация
# ---------------------------
def extract_vertex_values_for_triangulation(mesh, V, uh):
    """
    Возвращает (x, y, triangles, p_vertex), где p_vertex — комплексные значения
    в вершинах, полученные корректно через dofs на 0-мерных сущностях.
    """
    # Координаты вершин
    vertex_coords = mesh.geometry.x
    x = vertex_coords[:, 0].copy()
    y = vertex_coords[:, 1].copy()

    # Треугольная сетка для matplotlib.tri
    tdim = mesh.topology.dim
    mesh.topology.create_connectivity(tdim, 0)
    c2v = mesh.topology.connectivity(tdim, 0).array
    # Предполагаем треугольники
    triangles = c2v.reshape(-1, 3)

    # DOFs на вершинах (в порядке номеров вершин)
    num_local_vertices = mesh.topology.index_map(0).size_local
    vertices = np.arange(num_local_vertices, dtype=np.int32)
    dofs_on_vertices = fem.locate_dofs_topological(V, 0, vertices)

    # Значения решения в DOFs (локальный массив)
    p_dofs = uh.x.array

    # Значения в вершинах (упорядоченные по vertex index)
    p_vertex = p_dofs[dofs_on_vertices]

    return x, y, triangles, p_vertex


def animate_pressure_field(
    x,
    y,
    triangles,
    p_complex_vertex,
    freq_hz,
    A,
    width,
    height,
    num_waves,
    Sx,
    Sy,
    mic_pos=(0.9, 0.9),
    num_frames=30,
    fps=10,
    dpi=100,
    out_gif="pressure_field.gif",
    comm=MPI.COMM_WORLD,
):
    """
    Создает и сохраняет GIF-анимацию распределения давления и статичный PNG.
    """
    rank = comm.rank
    if rank != 0:
        return

    # Параметры сцены
    abs_max = np.abs(p_complex_vertex).max()
    phases = np.linspace(0, 2 * np.pi, num_frames, endpoint=False)

    # Триангуляция
    triang = tri.Triangulation(x, y, triangles)

    # Границы для визуализации
    x_bound = np.linspace(0, width, 100)
    y_bottom = A * np.sin(2 * np.pi * num_waves * x_bound / width)
    y_top = height + A * np.sin(2 * np.pi * num_waves * x_bound / width)
    y_bottom_min = y_bottom.min() - 0.05
    y_top_max = y_top.max() + 0.05

    fig, ax = plt.subplots(figsize=(8, 6))
    ax.set_xlim(0, width)
    ax.set_ylim(y_bottom_min, y_top_max)
    ax.set_aspect("equal")
    ax.set_xlabel("X [m]")
    ax.set_ylabel("Y [m]")
    ax.set_title(f"Pressure Distribution at {freq_hz} Hz (Wavy Hard Boundaries)")

    # Постоянные элементы
    ax.plot(Sx, Sy, "ro", markersize=8, label="Source")
    ax.plot(mic_pos[0], mic_pos[1], "go", markersize=8, label="Mic")
    ax.plot(x_bound, y_bottom, "k-", linewidth=2, label="Wavy Hard Boundary")
    ax.plot(x_bound, y_top, "k-", linewidth=2)
    ax.plot([0, 0], [y_bottom[0], y_top[0]], "b-", linewidth=1, label="Impedance Boundary")
    ax.plot([width, width], [y_bottom[-1], y_top[-1]], "b-", linewidth=1)
    ax.legend(loc="upper right")

    # Первый кадр
    phase0 = phases[0]
    p_real0 = np.real(p_complex_vertex * np.exp(1j * phase0))
    contour = ax.tricontourf(
        triang, p_real0, levels=50, cmap="RdBu_r", vmin=-abs_max, vmax=abs_max
    )
    cbar = fig.colorbar(contour, ax=ax, shrink=0.8)
    cbar.set_label("Pressure [Pa]")

    # Храним ссылки на коллекции контура для аккуратного удаления
    contour_artists = list(contour.collections)

    def update(frame):
        nonlocal contour_artists
        # Удаляем только прежние контуры
        for coll in contour_artists:
            coll.remove()

        phase = phases[frame]
        p_real = np.real(p_complex_vertex * np.exp(1j * phase))

        new_contour = ax.tricontourf(
            triang, p_real, levels=50, cmap="RdBu_r", vmin=-abs_max, vmax=abs_max
        )
        contour_artists = list(new_contour.collections)

        ax.set_title(
            f"Pressure at {freq_hz} Hz, Phase: {phase/np.pi:.2f}π rad\nWavy Hard Boundaries Top/Bottom"
        )
        return contour_artists

    ani = animation.FuncAnimation(
        fig, update, frames=num_frames, interval=100, repeat=True, blit=False
    )

    gif_filename = out_gif or f"pressure_field_wavy_boundaries_{freq_hz}Hz.gif"
    ani.save(gif_filename, writer="pillow", fps=fps, dpi=dpi)
    print(f"GIF saved as {gif_filename}")

    # Статичная картинка
    plt.figure(figsize=(8, 6))
    p_real = np.real(p_complex_vertex)
    contour_static = plt.tricontourf(triang, p_real, levels=50, cmap="RdBu_r")
    plt.colorbar(contour_static, label="Pressure [Pa]")
    plt.plot(Sx, Sy, "ro", markersize=8, label="Source")
    plt.plot(*mic_pos, "go", markersize=8, label="Mic")
    plt.plot(x_bound, y_bottom, "k-", linewidth=2, label="Wavy Hard Boundary")
    plt.plot(x_bound, y_top, "k-", linewidth=2)
    plt.plot([0, 0], [y_bottom[0], y_top[0]], "b-", linewidth=1, label="Impedance Boundary")
    plt.plot([width, width], [y_bottom[-1], y_top[-1]], "b-", linewidth=1)
    plt.title(f"Pressure Distribution at {freq_hz} Hz (Wavy Hard Boundaries)")
    plt.xlabel("X [m]")
    plt.ylabel("Y [m]")
    plt.xlim(0, width)
    plt.ylim(y_bottom_min, y_top_max)
    plt.legend()
    png_filename = f"pressure_wavy_boundaries_{freq_hz}Hz.png"
    plt.savefig(png_filename, dpi=150)
    print(f"Static plot saved as {png_filename}")


# ---------------------------
# Основной сценарий
# ---------------------------
def main():
    comm = MPI.COMM_WORLD
    rank = comm.rank

    # Параметры
    deg = 2
    n_elem = 64
    width = 1.0
    height = 1.0
    A = 0.05
    num_waves = 3
    num_points_boundary = 50

    freq = 800.0  # Гц
    Q = 0.0001
    Sx, Sy = 0.1, 0.5  # координаты источника
    c0 = 340.0
    rho0 = 1.225
    sigma = 0.015  # ширина гауссианы для источника

    if rank == 0:
        print(f"Computing for frequency: {freq} Hz")

    # 1) Сетка
    mesh, cell_markers, facet_markers = build_wavy_channel_mesh(
        width=width,
        height=height,
        A=A,
        num_waves=num_waves,
        n_elem=n_elem,
        num_points_boundary=num_points_boundary,
        comm=comm,
    )

    # 2) Пространство
    V, u, v = setup_function_space(mesh, deg)

    # 3) Источник (гауссова дельта)
    delta_fun = gaussian_delta_on_mesh(V, Sx=Sx, Sy=Sy, sigma=sigma, comm=comm)

    # 4) Импедансный коэффициент (требует k0 для формулы, но через Constant зададим позже)
    # Чтобы корректно сформировать Z(x), заведем фиктивный k0 и потом пересоздадим/переиспользуем выражение.
    tmp_k0 = Constant(mesh, PETSc.ScalarType(1.0))  # временное значение
    r_fun, Z_expr = impedance_coefficient(V, Sx=Sx, Sy=Sy, rho0=rho0, c0=c0, k0=tmp_k0)

    # 5) Решение
    uh, omega, k0 = solve_helholtz_pressure(
        mesh=mesh,
        facet_markers=facet_markers,
        V=V,
        u=u,
        v=v,
        freq_hz=freq,
        source_amp_Q=Q,
        delta_fun=delta_fun,
        rho0=rho0,
        c0=c0,
        Z_expr=Z_expr,  # выражение с k0 будет использовать фактический k0 в форме
        left_right_markers=(4, 2),
        ksp_options={"ksp_type": "preonly", "pc_type": "lu"},
    )

    # 6) Данные для анимации
    x, y, triangles, p_complex_vertex = extract_vertex_values_for_triangulation(mesh, V, uh)

    # 7) Анимация и статичный график (только на rank 0)
    animate_pressure_field(
        x=x,
        y=y,
        triangles=triangles,
        p_complex_vertex=p_complex_vertex,
        freq_hz=freq,
        A=A,
        width=width,
        height=height,
        num_waves=num_waves,
        Sx=Sx,
        Sy=Sy,
        mic_pos=(0.9, 0.9),
        num_frames=30,
        fps=10,
        dpi=100,
        out_gif=f"pressure_field_wavy_boundaries_{freq}Hz.gif",
        comm=comm,
    )


if __name__ == "__main__":
    main()
