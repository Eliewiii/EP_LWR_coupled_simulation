"""Module defining structured, type-safe data containers for the generator suite."""

from pydantic import BaseModel
from pyenergyplus.model.model import (
    Building,
    Construction,
    GlobalGeometryRules,
    Material,
    OutputSqLite,
    OutputVariable,
    RunPeriod,
    SimulationControl,
    Timestep,
    WindowMaterialSimpleGlazingSystem,
)


class SimulationBaseRegistry(BaseModel):
    """Type-safe data class mapping baseline profiles, eliminating string lookup typos."""

    building: dict[str, Building]
    global_geometry_rules: dict[str, GlobalGeometryRules]
    timestep: dict[str, Timestep]
    simulation_control: dict[str, SimulationControl]
    output_variable: dict[str, OutputVariable]
    output_sq_lite: dict[str, OutputSqLite]
    run_period: dict[str, RunPeriod]
    material: dict[str, Material]
    window_material_simple_glazing_system: dict[str, WindowMaterialSimpleGlazingSystem]
    construction: dict[str, Construction]

    class Config:
        """Configuring arbitrary type handling to play nice with direct pyenergyplus instances."""

        arbitrary_types_allowed = True
