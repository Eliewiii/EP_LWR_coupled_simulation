"""Module for programmatically orchestrating baseline geometries, compiling IDFs,

and instantiating analytical View Factor sparse matrices for LWR test execution loops.
"""

from pathlib import Path

import numpy as np

from ..._schemas import BuildingInput, SimulationInputs
from ..ep_models.default import (
    get_default_base_registry,
    get_single_zone_box_geometry,
    get_single_zone_u_geometry,
    get_three_box_buildings_geometry,
)
from ..ep_models.generate_model import compile_energyplus_model
from ..vf_computation.analytical_vf import (
    view_factor_parallel_plates,
    view_factor_perpendicular_plates,
)
from .geometry_parsers import derive_surface_radiative_properties, extract_outdoor_surfaces
from .io_handlers import save_and_convert_to_idf, serialize_numpy_matrices


# =====================================================================
# Case Study 1: Solo Box Building
# =====================================================================
def generate_solo_box_case_study(
    output_workspace: Path,
    epw_file_path: Path,
    width: float = 10.0,
    length: float = 10.0,
    height: float = 3.0,
) -> SimulationInputs:
    """Generates a single standalone box building case study workspace.

    Extracts properties and surfaces natively from raw geometric arrays, compiles the target
    IDF via the core compiler, and sets an empty analytical view factor matrix.
    """
    output_workspace.mkdir(exist_ok=True, parents=True)

    base_registry = get_default_base_registry()

    # 1. Fetch raw footprint geometry maps
    geo_data = get_single_zone_box_geometry(width=width, length=length, height=height)
    surfaces_pool = geo_data["BuildingSurfaceDetailed"]

    # 2. Extract outward-facing envelope coordinates and material properties immediately from
    # the pool
    outdoor_surfaces = extract_outdoor_surfaces(surfaces_pool)
    eps_v, rho_v, tau_v = derive_surface_radiative_properties(
        outdoor_surfaces, surfaces_pool, base_registry
    )

    # 3. Pass registries to core builder and convert model to physical IDF format
    ep_model = compile_energyplus_model(
        base_registry=base_registry,
        building_zone_registry=geo_data["Zone"],
        building_surface_registry=surfaces_pool,
    )
    idf_path = save_and_convert_to_idf(ep_model, output_workspace, "solo_box_building")

    # 4. Formulate the view factor matrix (Strictly 0 for a single convex box enclosure)
    n = len(outdoor_surfaces)
    vf_matrix = np.zeros((n, n))

    # 5. Commit matrix tracking assets to storage files
    matrices = serialize_numpy_matrices(
        output_workspace, "solo_box", vf_matrix, eps_v, rho_v, tau_v
    )

    # 6. Pack data structures into Pydantic container states
    building_input = BuildingInput(
        building_id="bldg_solo_box",
        idf_path=idf_path,
        outdoor_surface_names=outdoor_surfaces,
    )

    num_ts_per_h = list(base_registry.timestep.values())[0].number_of_timesteps_per_hour
    if num_ts_per_h is None:
        raise ValueError("Timestep registry entry must have a defined number_of_timesteps_per_hour")

    return SimulationInputs(
        workspace_dir=output_workspace,
        epw_path=epw_file_path,
        num_ts_per_h=num_ts_per_h,
        vf_matrix_path=matrices["vf"],
        eps_matrix_path=matrices["eps"],
        rho_matrix_path=matrices["rho"],
        tau_matrix_path=matrices["tau"],
        buildings=[building_input],
    )


# =====================================================================
# Case Study 2: Solo U-Shape Building
# =====================================================================
def generate_solo_u_case_study(
    output_workspace: Path,
    epw_file_path: Path,
    w_total: float = 20.0,
    l_total: float = 20.0,
    w_wing: float = 6.0,
    l_courtyard: float = 12.0,
    height: float = 3.5,
) -> SimulationInputs:
    """Generates a single U-shaped courtyard building case study workspace.

    Analytically computes internal reflection factors between opposing courtyard walls,
    extracts the outer material radiative properties directly from the shape pool, and compiles
    the IDF.
    """
    output_workspace.mkdir(exist_ok=True, parents=True)

    base_registry = get_default_base_registry()

    # 1. Instantiate the target courtyard shell footprint geometry
    geo_data = get_single_zone_u_geometry(
        w_total=w_total, l_total=l_total, w_wing=w_wing, l_courtyard=l_courtyard, height=height
    )
    surfaces_pool = geo_data["BuildingSurfaceDetailed"]

    # 2. Extract envelope surfaces and material metrics from the clean dictionary structure
    outdoor_surfaces = extract_outdoor_surfaces(surfaces_pool)
    eps_v, rho_v, tau_v = derive_surface_radiative_properties(
        outdoor_surfaces, surfaces_pool, base_registry
    )

    # 3. Compile the actual simulation model and export to an IDF asset script
    ep_model = compile_energyplus_model(
        base_registry=base_registry,
        building_zone_registry=geo_data["Zone"],
        building_surface_registry=surfaces_pool,
    )
    idf_path = save_and_convert_to_idf(ep_model, output_workspace, "solo_u_building")

    # 3. Construct the analytical internal view factor matrix
    n = len(outdoor_surfaces)
    vf_matrix = np.zeros((n, n))

    # Calculate the opening width of the courtyard aperture gap
    w_open = w_total - (2.0 * w_wing)

    # A. Perpendicular Corner interactions (Side walls to Back wall)
    # Plate 1 width = back wall (w_open), Plate 2 width = side wall (l_courtyard)
    vf_perp = view_factor_perpendicular_plates(w1=w_open, w2=l_courtyard, h=height)

    # By reciprocity: F_(side->back) = F_(back->side) * Area_(back) / Area_(side)
    # Area_back = w_open * height, Area_side = l_courtyard * height
    vf_perp_reciprocal = vf_perp * (w_open / l_courtyard)

    # B. Parallel Opposing interactions (Wing face to Wing face)
    vf_parallel = view_factor_parallel_plates(w=l_courtyard, h=height, d=w_open)

    # Map the analytical components into the matrix based on surface name strings
    for i, s_i in enumerate(outdoor_surfaces):
        for j, s_j in enumerate(outdoor_surfaces):
            # --- 1. Parallel Wing-to-Wing Exchange ---
            if "wall_3" in s_i and "wall_5" in s_j:
                vf_matrix[i, j] = vf_parallel
            elif "wall_5" in s_i and "wall_3" in s_j:
                vf_matrix[i, j] = vf_parallel

            # --- 2. Perpendicular Side-to-Back Exchange ---
            # From side walls looking at the back wall
            elif ("wall_3" in s_i or "wall_5" in s_i) and "wall_4" in s_j:
                vf_matrix[i, j] = vf_perp_reciprocal

            # From back wall looking out at the side walls
            elif "wall_4" in s_i and ("wall_3" in s_j or "wall_5" in s_j):
                vf_matrix[i, j] = vf_perp

    # 5. Commit matrices and vectors to disk space storage
    matrices = serialize_numpy_matrices(output_workspace, "solo_u", vf_matrix, eps_v, rho_v, tau_v)

    # 6. Pack parameters into execution state containers
    building_input = BuildingInput(
        building_id="bldg_solo_u",
        idf_path=idf_path,
        outdoor_surface_names=outdoor_surfaces,
    )

    num_ts_per_h = list(base_registry.timestep.values())[0].number_of_timesteps_per_hour
    if num_ts_per_h is None:
        raise ValueError("Timestep registry entry must have a defined number_of_timesteps_per_hour")

    return SimulationInputs(
        workspace_dir=output_workspace,
        epw_path=epw_file_path,
        num_ts_per_h=num_ts_per_h,
        vf_matrix_path=matrices["vf"],
        eps_matrix_path=matrices["eps"],
        rho_matrix_path=matrices["rho"],
        tau_matrix_path=matrices["tau"],
        buildings=[building_input],
    )


# =====================================================================
# Case Study 3: Three Box Buildings (Parallel & Identical)
# =====================================================================
def generate_three_box_case_study(
    output_workspace: Path,
    epw_file_path: Path,
    width: float = 10.0,
    length: float = 10.0,
    height: float = 3.0,
    separation_distance: float = 6.0,
) -> SimulationInputs:
    """Generates three identical, parallel box building models aligned cleanly along the X-axis.

    Utilizes the default base registry and the macro factory function for decoupled geometries,
    compiles independent IDFs, and builds the combined analytical cross-building view factor map.
    """
    output_workspace.mkdir(exist_ok=True, parents=True)
    building_inputs_list = []
    all_models_surfaces = []

    base_registry = get_default_base_registry()

    # Plain Python dictionary to aggregate decoupled surface data safe from model property
    # constraints
    raw_surfaces_pool = {}

    # 1. Fetch decoupled absolute geometry configurations via standard sub-package function
    geometries = get_three_box_buildings_geometry(
        west_zone_name="west_building",
        middle_zone_name="middle_building",
        east_zone_name="east_building",
        width=width,
        length=length,
        height=height,
        separation_distance=separation_distance,
    )

    # 2. Extract surfaces natively and compile individual IDF files sequentially
    for orientation in ["west", "middle", "east"]:
        geo_data = geometries[orientation]
        bldg_id = f"{orientation}_building"
        surfaces_pool = geo_data["BuildingSurfaceDetailed"]

        # Parse outdoor targets natively from the raw shape block dictionary before model
        # encapsulation
        outdoor_surfaces = extract_outdoor_surfaces(surfaces_pool)

        ep_model = compile_energyplus_model(
            base_registry=base_registry,
            building_zone_registry=geo_data["Zone"],
            building_surface_registry=surfaces_pool,
        )
        idf_path = save_and_convert_to_idf(ep_model, output_workspace, f"multi_{bldg_id}")

        building_inputs_list.append(
            BuildingInput(
                building_id=bldg_id, idf_path=idf_path, outdoor_surface_names=outdoor_surfaces
            )
        )

        # Track master order structures and pool data records for unified cross-building arrays
        all_models_surfaces.extend(outdoor_surfaces)
        raw_surfaces_pool.update(surfaces_pool)

    # 3. Compute exact parallel-plate interaction factors across building gaps
    total_n = len(all_models_surfaces)
    master_vf_matrix = np.zeros((total_n, total_n))

    # Spacing variables for immediate neighbor structures vs far field jumps
    vf_adjacent = view_factor_parallel_plates(w=length, h=height, d=separation_distance)
    vf_distant = view_factor_parallel_plates(
        w=length, h=height, d=(2.0 * width) + (2.0 * separation_distance)
    )

    # 4. Construct cross-building view factor matrix bindings matching specific zone surface strings
    for i, s_i in enumerate(all_models_surfaces):
        for j, s_j in enumerate(all_models_surfaces):
            # Interactions between West and Middle buildings
            if "west_building_wall_e" in s_i and "middle_building_wall_w" in s_j:
                master_vf_matrix[i, j] = vf_adjacent
            elif "middle_building_wall_w" in s_i and "west_building_wall_e" in s_j:
                master_vf_matrix[i, j] = vf_adjacent

            # Interactions between Middle and East buildings
            elif "middle_building_wall_e" in s_i and "east_building_wall_w" in s_j:
                master_vf_matrix[i, j] = vf_adjacent
            elif "east_building_wall_w" in s_i and "middle_building_wall_e" in s_j:
                master_vf_matrix[i, j] = vf_adjacent

            # Far field long distance radiation jump interaction between West and East buildings
            elif "west_building_wall_e" in s_i and "east_building_wall_w" in s_j:
                master_vf_matrix[i, j] = vf_distant
            elif "east_building_wall_w" in s_i and "west_building_wall_e" in s_j:
                master_vf_matrix[i, j] = vf_distant

    # 5. Extract material parameters directly from the clean aggregated surface pool dictionary
    eps_v, rho_v, tau_v = derive_surface_radiative_properties(
        all_models_surfaces, raw_surfaces_pool, base_registry
    )
    matrices = serialize_numpy_matrices(
        output_workspace, "three_box", master_vf_matrix, eps_v, rho_v, tau_v
    )

    # 6. Return compiled validation configuration states
    num_ts_per_h = list(base_registry.timestep.values())[0].number_of_timesteps_per_hour
    if num_ts_per_h is None:
        raise ValueError("Timestep registry entry must have a defined number_of_timesteps_per_hour")

    return SimulationInputs(
        workspace_dir=output_workspace,
        epw_path=epw_file_path,
        num_ts_per_h=num_ts_per_h,
        vf_matrix_path=matrices["vf"],
        eps_matrix_path=matrices["eps"],
        rho_matrix_path=matrices["rho"],
        tau_matrix_path=matrices["tau"],
        buildings=building_inputs_list,
    )
