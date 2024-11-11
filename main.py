import numpy as np

from utils import VTKReader

from surface_processing import process_vtk_surfaces


def load_well_trajectory(filename: str) -> np.ndarray:
    vtk_data = VTKReader(filename)
    trajectory_points = []
    num_points = vtk_data.num_of_points
    for i in range(num_points):
        x, y, z = vtk_data.GetPoint(i)
        trajectory_points.append([x, y, z])

    return np.array(trajectory_points)


def generate_perforation_points(
    well_trajectory: np.ndarray, perforation_step: float = 5.0
) -> np.ndarray:
    """
    Генерирует точки перфорации на траектории скважины с заданным шагом.

    Args:
        well_trajectory (np.ndarray): Траектория скважины.
        perforation_step (float): Шаг между точками перфорации.

    Returns:
        np.ndarray: Массив точек перфорации.
    """
    perforation_points = []
    accumulated_distance = 0.0
    perforation_points.append(well_trajectory[-2])

    for i in range(len(well_trajectory) - 2, -1, -1):
        dist = np.linalg.norm(well_trajectory[i] - well_trajectory[i - 1])
        accumulated_distance += dist
        if accumulated_distance >= perforation_step:
            perforation_points.append(well_trajectory[i])
            accumulated_distance = 0.0

    return np.array(perforation_points)


def point_in_prism(point, prism_vertices):
    """
    Проверяет, находится ли точка внутри призмы.

    Args:
        point (np.ndarray): Координаты точки (x, y, z).
        prism_vertices (list): Список из 6 вершин призмы.

    Returns:
        bool: True, если точка внутри призмы, иначе False.
    """
    p1, p2, p3 = prism_vertices[0], prism_vertices[1], prism_vertices[2]
    p4, p5, p6 = prism_vertices[3], prism_vertices[4], prism_vertices[5]

    def is_point_in_triangle(triangle, point):
        a, b, c = triangle
        v0, v1, v2 = np.array(c) - a, np.array(b) - a, np.array(point) - a
        d00, d01, d11 = np.dot(v0, v0), np.dot(v0, v1), np.dot(v1, v1)
        d20, d21 = np.dot(v2, v0), np.dot(v2, v1)
        denom = d00 * d11 - d01 * d01
        if denom == 0:
            return False
        v = (d11 * d20 - d01 * d21) / denom
        w = (d00 * d21 - d01 * d20) / denom
        u = 1 - v - w
        return (u >= 0) and (v >= 0) and (w >= 0)

    if not (
        is_point_in_triangle([p1, p2, p3], point)
        or is_point_in_triangle([p4, p5, p6], point)
    ):
        return False

    z_min = min(p1[2], p2[2], p3[2])
    z_max = max(p4[2], p5[2], p6[2])
    return z_min <= point[2] <= z_max


def save_mesh_with_tags(
    prism_points,
    prism_cells,
    cell_types,
    perforation_points,
    filename,
):
    """
    Сохраняет сетку с тегами перфорации в VTK файл.

    Args:
        prism_points (list): Список координат точек.
        prism_cells (list): Список ячеек-призм.
        cell_types (list): Список типов ячеек.
        perforation_points (np.ndarray): Массив точек перфорации.
        filename (str): Имя выходного VTK файла.
    """
    num_points = len(prism_points)
    num_cells = len(prism_cells)

    cell_tags = np.zeros(num_cells, dtype=int)
    intersected_cell_ids = []

    prism_points_array = np.array(prism_points)

    for i, cell in enumerate(prism_cells):
        cell_points = prism_points_array[cell]
        tag_value = 0
        for perforation in perforation_points:
            if point_in_prism(perforation, cell_points):
                tag_value = 1
                intersected_cell_ids.append(i)
                break
        cell_tags[i] = tag_value

    with open(filename, "w") as f:
        f.write("# vtk DataFile Version 4.0\n")
        f.write("3D mesh with perforation tags\n")
        f.write("ASCII\n")
        f.write("DATASET UNSTRUCTURED_GRID\n")

        f.write(f"POINTS {num_points} float\n")
        for point in prism_points:
            f.write(f"{point[0]} {point[1]} {point[2]}\n")

        total_num_cell_indices = sum(len(cell) + 1 for cell in prism_cells)

        f.write(f"CELLS {num_cells} {total_num_cell_indices}\n")
        for cell in prism_cells:
            f.write(f"{len(cell)} " + " ".join(map(str, cell)) + "\n")

        f.write(f"CELL_TYPES {num_cells}\n")
        for cell_type in cell_types:
            f.write(f"{cell_type}\n")

        f.write(f"CELL_DATA {num_cells}\n")
        f.write("SCALARS PerforationTag int 1\n")
        f.write("LOOKUP_TABLE default\n")
        for tag in cell_tags:
            f.write(f"{tag}\n")

    print(f"Сетка сохранена в {filename}.")
    print("ID пересекающихся ячеек:", intersected_cell_ids)
    print("Количество пересекающихся ячеек:", len(intersected_cell_ids))


input_pattern = "surfaces_vtk/*.vtk"
output_directory = "vtk"

process_vtk_surfaces(input_pattern, output_directory)

perforation_step = 200.0

surfaces = []
filenames = [
    "vtk/surf_1.vtk",
    "vtk/surf_2.vtk",
    "vtk/surf_3.vtk",
    "vtk/surf_base.vtk",
    "vtk/surf_top.vtk",
]

for filename in filenames:
    vtk_data = VTKReader(filename)
    surfaces.append(vtk_data)

surface_base = surfaces[3]
z_min = min(surface_base.GetPoint(i)[2] for i in range(surface_base.num_of_points))

surface_top = surfaces[4]
top_points = {}
for i in range(surface_top.num_of_points):
    x, y, z = surface_top.GetPoint(i)
    key = (round(x, 6), round(y, 6))
    top_points[key] = z

unique_x = sorted(
    set(point.x for surface in surfaces[:-2] for point in surface.points_array)
)
unique_y = sorted(
    set(point.y for surface in surfaces[:-2] for point in surface.points_array)
)

grid_points = []
point_map = {}

for y in unique_y:
    for x in unique_x:
        key = (round(x, 6), round(y, 6))
        if key not in top_points:
            continue

        grid_points.append((x, y, z_min))
        bottom_idx = len(grid_points) - 1
        z_top = top_points[key]
        grid_points.append((x, y, z_top))
        top_idx = len(grid_points) - 1
        point_map[key] = (bottom_idx, top_idx)

prism_cells = []
cell_types = []

dx = unique_x[1] - unique_x[0] if len(unique_x) > 1 else 1.0
dy = unique_y[1] - unique_y[0] if len(unique_y) > 1 else 1.0

for i in range(len(unique_x) - 1):
    for j in range(len(unique_y) - 1):
        x1, x2 = unique_x[i], unique_x[i + 1]
        y1, y2 = unique_y[j], unique_y[j + 1]

        p1 = (round(x1, 6), round(y1, 6))
        p2 = (round(x2, 6), round(y1, 6))
        p3 = (round(x2, 6), round(y2, 6))
        p4 = (round(x1, 6), round(y2, 6))

        if (
            p1 not in point_map
            or p2 not in point_map
            or p3 not in point_map
            or p4 not in point_map
        ):
            continue

        b1, t1 = point_map[p1]
        b2, t2 = point_map[p2]
        b3, t3 = point_map[p3]
        b4, t4 = point_map[p4]

        cell1 = [b1, b2, b3, t1, t2, t3]
        prism_cells.append(cell1)
        cell_types.append(13)

        cell2 = [b1, b3, b4, t1, t3, t4]
        prism_cells.append(cell2)
        cell_types.append(13)

well_trajectory_file = "trajectory_curve.vtk"
well_trajectory = load_well_trajectory(well_trajectory_file)

perforation_points = generate_perforation_points(
    well_trajectory, perforation_step=perforation_step
)

save_mesh_with_tags(
    grid_points,
    prism_cells,
    cell_types,
    perforation_points,
    filename="result.vtk",
)
