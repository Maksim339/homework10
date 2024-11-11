import glob
import pandas as pd
import numpy as np
from tqdm import tqdm
from functools import reduce


def read_vtk_files(file_pattern):
    vtk_files = glob.glob(file_pattern)
    print(vtk_files)
    if len(vtk_files) != 5:
        raise ValueError(f"Файлов должно быть 5, найдено {len(vtk_files)}")

    vtk_contents = {}
    for filepath in tqdm(vtk_files, desc="Загрузка vtk файлов"):
        with open(filepath, "r") as file:
            vtk_contents[filepath] = file.read()
    return vtk_contents


def extract_coordinates(vtk_text):
    lines = vtk_text.splitlines()
    coords = []
    start_parsing = False
    for line in lines:
        if line.startswith("POINTS"):
            start_parsing = True
            continue
        if start_parsing:
            if not line.strip():
                break
            parts = line.strip().split()
            if len(parts) == 3:
                coords.append([float(part) for part in parts])
    return pd.DataFrame(coords, columns=["x", "y", "z"])


def combine_surface_dataframes(surface_dfs):
    """Cоединяем таблицы с данными поверхностей внешним джойном по x, y"""
    # Rename columns to include surface names
    renamed_dfs = {}
    for name, df in surface_dfs.items():
        df_renamed = df.rename(columns={"z": f"z_{name}", "exist": f"exist_{name}"})
        renamed_dfs[name] = df_renamed
    merged_df = reduce(
        lambda left, right: pd.merge(left, right, on=["x", "y"], how="outer"),
        renamed_dfs.values(),
    )
    return merged_df


def generate_trimmed_surfaces(merged_df):
    """Создание обрезанных поверхностей"""

    surfaces = {}

    surface_top = merged_df[merged_df["exist_surf_top"] == True]
    surfaces["surf_top"] = surface_top[["x", "y", "z_surf_top"]].rename(
        columns={"z_surf_top": "z"}
    )

    condition1 = (merged_df["exist_surf_1"] == True) & (
        merged_df["z_surf_base"] <= merged_df["z_surf_1"]
    )
    condition2 = (merged_df["exist_surf_base"] == True) & (
        merged_df["z_surf_base"] >= merged_df["z_surf_1"]
    )
    surface1 = merged_df[condition1][["x", "y", "z_surf_1"]].rename(
        columns={"z_surf_1": "z"}
    )
    surface1_alt = merged_df[condition2][["x", "y"]].copy()
    surface1_alt["z"] = merged_df.loc[condition2, "z_surf_base"]
    surfaces["surf_1"] = pd.concat([surface1, surface1_alt])

    condition1 = (merged_df["exist_surf_2"] == True) & (
        merged_df["z_surf_base"] <= merged_df["z_surf_2"]
    )
    condition2 = (merged_df["exist_surf_base"] == True) & (
        (merged_df["z_surf_base"] >= merged_df["z_surf_2"])
        | (merged_df["z_surf_2"].isna())
    )
    surface2 = merged_df[condition1][["x", "y", "z_surf_2"]].rename(
        columns={"z_surf_2": "z"}
    )
    surface2_alt = merged_df[condition2][["x", "y"]].copy()
    surface2_alt["z"] = merged_df.loc[condition2, "z_surf_base"]
    surfaces["surf_2"] = pd.concat([surface2, surface2_alt])

    condition1 = merged_df["exist_surf_3"] == True
    condition2 = (merged_df["exist_surf_base"] is True) & (
        merged_df["exist_surf_3"] != True
    )
    surface3 = merged_df[condition1][["x", "y", "z_surf_3"]].rename(
        columns={"z_surf_3": "z"}
    )
    surface3_alt = merged_df[condition2][["x", "y"]].copy()
    surface3_alt["z"] = merged_df.loc[condition2, "z_surf_base"]
    surfaces["surf_3"] = pd.concat([surface3, surface3_alt])

    surfaces["surf_base"] = merged_df[merged_df["exist_surf_base"] == True][
        ["x", "y", "z_surf_base"]
    ].rename(columns={"z_surf_base": "z"})

    surf_1_coords = surfaces["surf_1"][["x", "y"]]
    for key in ["surf_top", "surf_2", "surf_3", "surf_base"]:
        surfaces[key] = pd.merge(
            surfaces[key], surf_1_coords, on=["x", "y"], how="inner"
        )

    for key in surfaces:
        surfaces[key] = surfaces[key].sort_values(by=["y", "x"])

    return surfaces


def save_vtk_files(surfaces, output_directory):
    """Сохранение обрезанных поверхностей в vtk"""
    arr = np.arange(0, 5001, 50)
    xx, yy = np.meshgrid(arr, arr)

    for name, df in tqdm(surfaces.items(), desc="Сохранение vtk файлов"):
        grid_df = df.pivot(index="y", columns="x", values="z").reindex(
            index=arr, columns=arr
        )
        zz = grid_df.values

        flags = np.full(zz.shape, -1, dtype=int)
        counter = 0
        for j in range(zz.shape[0]):
            for i in range(zz.shape[1]):
                if not np.isnan(zz[j, i]):
                    flags[j, i] = counter
                    counter += 1

        cells = []
        cell_values = []
        ny, nx = zz.shape
        for j in range(ny - 1):
            for i in range(nx - 1):
                indices = [
                    flags[j, i],
                    flags[j + 1, i],
                    flags[j + 1, i + 1],
                    flags[j, i + 1],
                ]
                if all(idx != -1 for idx in indices):
                    cells.append(f"4 {' '.join(map(str, indices))}")
                    cell_values.append(
                        np.nanmean(
                            [zz[j, i], zz[j + 1, i], zz[j + 1, i + 1], zz[j, i + 1]]
                        )
                    )

        output_path = f"{output_directory}/{name}.vtk"
        with open(output_path, "w") as vtk_file:
            vtk_file.write("# vtk DataFile Version 3.0\n")
            vtk_file.write("Generated by vtk_surface_processor\n")
            vtk_file.write("ASCII\n")
            vtk_file.write("DATASET POLYDATA\n")
            vtk_file.write(f"POINTS {counter} double\n")
            for j in range(zz.shape[0]):
                for i in range(zz.shape[1]):
                    if flags[j, i] != -1:
                        vtk_file.write(f"{arr[i]} {arr[j]} {zz[j, i]}\n")
            vtk_file.write(f"\nPOLYGONS {len(cells)} {len(cells) * 5}\n")
            for cell in cells:
                vtk_file.write(f"{cell}\n")
            vtk_file.write(f"\nCELL_DATA {len(cells)}\n")
            vtk_file.write("SCALARS Z double 1\n")
            vtk_file.write("LOOKUP_TABLE default\n")
            for value in cell_values:
                vtk_file.write(f"{value}\n")


def process_vtk_surfaces(input_pattern, output_directory):
    """Основная функция для того, чтобы обрезать поверхности из vtk и сохранить их в новые vtk"""
    vtk_texts = read_vtk_files(input_pattern)
    surface_dfs = {}
    for filepath, text in vtk_texts.items():
        name = filepath.split("/")[-1].split(".")[0]
        df = extract_coordinates(text)
        df["exist"] = True
        surface_dfs[name] = df.sort_values(by=["y", "x"])
    merged_df = combine_surface_dataframes(surface_dfs)
    trimmed_surfaces = generate_trimmed_surfaces(merged_df)
    save_vtk_files(trimmed_surfaces, output_directory)
    return trimmed_surfaces
