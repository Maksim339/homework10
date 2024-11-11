import os
from typing import Any

import numpy as np
import pandas as pd
from geomdl import fitting
from scipy.interpolate import LinearNDInterpolator
from dataclasses import dataclass

from scipy.spatial import Delaunay


@dataclass
class Point:
    """
    Координаты точки.

    Attributes:
        x (float): Координата X.
        y (float): Координата Y.
        z (float): Координата Z.
    """

    x: float
    y: float
    z: float


@dataclass
class Polygon:
    """
    Узлы поверхности.

    Attributes:
        node1 (int): Узел 1.
        node2 (int): Узел 2.
        node3 (int): Узел 3.
    """

    node1: int
    node2: int
    node3: int


@dataclass
class Scalar:
    """
    Данные скаляра.

    Attributes:
        value (float): Значение скаляра.
    """

    value: float


class VTKReader:
    def __init__(self, filename: str):
        if not filename.lower().endswith(".vtk"):
            raise ValueError("Файл должен иметь расширение '.vtk'")
        if not os.path.exists(filename):
            raise FileNotFoundError(f"Файл не найден: {filename}")

        with open(filename, "r") as file:
            lines = file.readlines()
            self._points_array = list(self._load_points(lines, "POINTS"))
            self._polygons_array = list(self._load_polygons(lines, "POLYGONS"))
            self._attribute_array = list(self._load_attribute(lines, "CELL_DATA"))
            self._filename_body = self._filename_body_load(filename)
            self._num_of_points = len(self._points_array)

    @property
    def num_of_points(self):
        """Количество точек."""
        return self._num_of_points

    @property
    def points_array(self) -> list[Point]:
        """Массив с точками."""
        return self._points_array

    @property
    def polygons_array(self) -> list[Polygon]:
        """Массив с поверхностями."""
        return self._polygons_array

    @property
    def attributes_array(self) -> list[Scalar]:
        """Массив с аттрибутами."""
        return self._attribute_array

    @property
    def filename_body(self) -> str:
        """Имя файла без расширения."""
        return self._filename_body

    def GetPoint(self, i):
        """Возвращает координаты точки по индексу."""
        point = self._points_array[i]
        return point.x, point.y, point.z

    def GetNumberOfPolygons(self):
        """Возвращает количество полигонов."""
        return len(self._polygons_array)

    def GetPolygon(self, i):
        """Возвращает индексы узлов полигона по индексу."""
        polygon = self._polygons_array[i]
        return polygon.node1, polygon.node2, polygon.node3

    @staticmethod
    def _load_points(lines: list[str], keyword: str) -> Point:
        reading_points = False
        num_points = 0

        for line in lines:
            if reading_points:
                parts = line.strip().split()
                if len(parts) < 3:
                    continue
                x, y, z = map(float, parts[:3])
                yield Point(x, y, z)
                num_points -= 1
                if num_points == 0:
                    break

            elif keyword in line:
                parts = line.strip().split()
                if len(parts) < 2:
                    continue
                num_points = int(parts[1])
                reading_points = True

    @staticmethod
    def _load_polygons(lines: list[str], keyword: str) -> Polygon:
        reading_polygons = False
        num_polygons = 0

        for line in lines:
            if reading_polygons:
                parts = line.strip().split()
                if len(parts) < 4:
                    continue
                _, node1, node2, node3 = map(int, parts[:4])
                yield Polygon(node1, node2, node3)
                num_polygons -= 1
                if num_polygons == 0:
                    break

            elif keyword in line:
                parts = line.strip().split()
                if len(parts) < 2:
                    continue
                num_polygons = int(parts[1])
                reading_polygons = True

    @staticmethod
    def _load_attribute(lines: list[str], keyword: str) -> Scalar:
        reading_attributes = False
        num_attributes = 0

        for line in lines:
            if reading_attributes:
                if num_attributes > 0:
                    try:
                        value = float(line.strip())
                        yield Scalar(value)
                        num_attributes -= 1
                    except ValueError:
                        continue
                    if num_attributes == 0:
                        break

            elif keyword in line:
                parts = line.strip().split()
                if len(parts) < 2:
                    continue
                num_attributes = int(parts[1])
                reading_attributes = True

    @staticmethod
    def _filename_body_load(filename: str) -> str:
        return os.path.splitext(os.path.basename(filename))[0]

    def vtk2pandas(self) -> pd.DataFrame:
        """
        Преобразование точек VTK в pandas DataFrame.
        """
        data = pd.DataFrame(
            {
                "x": [point.x for point in self.points_array],
                "y": [point.y for point in self.points_array],
                "z": [point.z for point in self.points_array],
            }
        )
        return data

    def save_curve_to_vtk(self, curve_points: list[tuple], filename: str):

        with open(filename, "w") as file:
            file.write("# vtk DataFile Version 4.0\n")
            file.write("Curve data\n")
            file.write("ASCII\n")
            file.write("DATASET POLYDATA\n")

            num_points = len(curve_points)
            file.write(f"POINTS {num_points} float\n")
            for point in curve_points:
                file.write(f"{point[0]} {point[1]} {point[2]}\n")

            num_lines = num_points - 1
            file.write(f"LINES {num_lines} {num_lines * 3}\n")
            for i in range(num_lines):
                file.write(f"2 {i} {i + 1}\n")
