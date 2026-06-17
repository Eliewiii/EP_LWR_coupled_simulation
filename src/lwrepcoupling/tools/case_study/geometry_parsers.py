"""Module for inspecting and parsing physical/thermal parameters from EnergyPlus models."""

from typing import Any

import numpy as np

from ..schemas import SimulationBaseRegistry


def extract_outdoor_surfaces(surfaces_pool: dict[str, Any]) -> list[str]:
    """
    Inspects a raw geometry surface pool dictionary and extracts outdoor-exposed surface names.
    """
    outdoor_surfaces = []
    for surface_name, surface_obj in surfaces_pool.items():
        # Safeguard: Extract the string value if it's a Pydantic/Python Enum instance
        bc = surface_obj.outside_boundary_condition
        bc_str = bc.value if hasattr(bc, "value") else str(bc)

        if bc_str.lower() == "outdoors":
            outdoor_surfaces.append(surface_name)

    return sorted(outdoor_surfaces)


def derive_surface_radiative_properties(
    outdoor_surfaces: list[str],
    surfaces_pool: dict[str, Any],
    base_registry: SimulationBaseRegistry,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Looks up the outermost material layers to derive thermal emissivity and reflectivity."""
    n = len(outdoor_surfaces)
    eps_vector = np.zeros(n)
    rho_vector = np.zeros(n)
    tau_vector = np.zeros(n)

    for idx, surface_name in enumerate(outdoor_surfaces):
        surface_obj = surfaces_pool[surface_name]
        const_name = surface_obj.construction_name

        construction_obj = base_registry.construction[const_name]
        outside_material_name = construction_obj.outside_layer

        if outside_material_name in base_registry.material:
            material_obj = base_registry.material[outside_material_name]
            eps = getattr(material_obj, "thermal_absorptance", 0.9)
        else:
            eps = 0.84

        eps_vector[idx] = eps
        rho_vector[idx] = 1.0 - eps

    return eps_vector, rho_vector, tau_vector
