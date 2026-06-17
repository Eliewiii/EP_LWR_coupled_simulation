"""Module providing production-grade default presets for EnergyPlus simulation initialization."""

from enum import Enum
from typing import Any

from pyenergyplus.model.model import (
    Building,
    Construction,
    CoordinateSystem,
    GlobalGeometryRules,
    Material,
    OptionType1,
    OutputSqLite,
    OutputVariable,
    ReportingFrequency,
    Roughness,
    RunPeriod,
    SimulationControl,
    SolarDistribution,
    StartingVertexPosition,
    Terrain,
    Timestep,
    UnitConversionForTabularData,
    VertexEntryDirection,
    WindowMaterialSimpleGlazingSystem,
)

from ..schemas import SimulationBaseRegistry
from .geometries import translate_zone_geometry
from .shapes import generate_cube_zone, generate_u_shape_zone
from .utils_epjson import to_epboolean


# =====================================================================
# Static Construction Registry Keys (Single Source of Truth)
# =====================================================================
class DefaultConstructionNames(str, Enum):
    """Provides explicit, string-backed accessors for baseline construction names."""

    WALL = "Default_Exterior_Wall_Construction"
    ROOF = "Default_Roof_Construction"
    FLOOR = "Default_Ground_Floor_Construction"
    WINDOW = "Default_Double_Glazing_Construction"


# =====================================================================
# Default global simulation parameters
# =====================================================================


def get_default_building_registry(name: str = "Main_Building") -> dict[str, Building]:
    """Returns a standard baseline building configuration map."""
    return {
        name: Building(
            terrain=Terrain.urban,
            solar_distribution=SolarDistribution.full_exterior_with_reflections,
            maximum_number_of_warmup_days=25,
        )
    }


def get_default_geometry_rules_registry() -> dict[str, GlobalGeometryRules]:
    """Returns the required relative coordinate tracking specifications matrix."""
    return {
        "GlobalGeometryRules 1": GlobalGeometryRules(
            starting_vertex_position=StartingVertexPosition.upper_left_corner,
            vertex_entry_direction=VertexEntryDirection.counterclockwise,
            coordinate_system=CoordinateSystem.relative,
        )
    }


def get_default_timestep_registry(timesteps_per_hour: int = 10) -> dict[str, Timestep]:
    """Returns a standard discrete execution step frequency map."""
    return {"Timestep 1": Timestep(number_of_timesteps_per_hour=timesteps_per_hour)}


def get_SimulationControl(
    name: str = "SimulationControl 1",
    *,
    do_zone_sizing_calculation: bool = True,  # REQUIRED for autosizing/ideal loads
    do_system_sizing_calculation: bool = False,  # Skip central air system sizing
    do_plant_sizing_calculation: bool = False,  # Skip central chiller/boiler sizing
    run_simulation_for_sizing_periods: bool = False,  # Do not simulate the design days
    run_simulation_for_weather_file_run_periods: bool = True,  # Run the actual EPW weather file
) -> dict[str, SimulationControl]:
    return {
        name: SimulationControl(
            do_zone_sizing_calculation=to_epboolean(do_zone_sizing_calculation),
            do_system_sizing_calculation=to_epboolean(do_system_sizing_calculation),
            do_plant_sizing_calculation=to_epboolean(do_plant_sizing_calculation),
            run_simulation_for_sizing_periods=to_epboolean(run_simulation_for_sizing_periods),
            run_simulation_for_weather_file_run_periods=to_epboolean(
                run_simulation_for_weather_file_run_periods
            ),
        )
    }


# =====================================================================
# Default output parameters
# =====================================================================


def get_default_zone_heating_and_cooling_output_registry(
    frequency: ReportingFrequency = ReportingFrequency.annual,
) -> dict[str, OutputVariable]:
    """Returns zone heating and cooling output variable requests."""
    return {
        "Output:Variable 1": OutputVariable(
            key_value="*",
            variable_name="Zone Total Heating Energy",
            reporting_frequency=frequency,
        ),
        "Output:Variable 2": OutputVariable(
            key_value="*",
            variable_name="Zone Total Cooling Energy",
            reporting_frequency=frequency,
        ),
    }


def get_default_sqlite_output_registry(
    name: str = "Output:SQLite 1",
    option_type: OptionType1 = OptionType1.simple,
) -> dict[str, OutputSqLite]:
    """Returns the default SQLite output registry configuration."""
    return {
        name: OutputSqLite(
            option_type=option_type,
            unit_conversion_for_tabular_data=UnitConversionForTabularData.use_output_control_table_style,
            format_numeric_values_for_tabular_data=to_epboolean(False),
        )
    }


def get_default_one_year_run_period_registry(
    name: str = "RunPeriod 1",
    *,
    begin_month: int = 1,
    begin_day_of_month: int = 1,
    end_month: int = 12,
    end_day_of_month: int = 31,
) -> dict[str, RunPeriod]:
    """Returns a one-year simulation run period registry."""
    return {
        name: RunPeriod(
            begin_month=begin_month,
            begin_day_of_month=begin_day_of_month,
            end_month=end_month,
            end_day_of_month=end_day_of_month,
            use_weather_file_holidays_and_special_days=to_epboolean(False),
            use_weather_file_daylight_saving_period=to_epboolean(False),
            use_weather_file_rain_indicators=to_epboolean(False),
            use_weather_file_snow_indicators=to_epboolean(False),
        )
    }


# =====================================================================
# Default material presets (Standard Building Physics Values)
# =====================================================================


def get_default_materials_registry() -> dict[str, Material]:
    """Returns a dictionary containing a comprehensive baseline suite of standard materials."""
    return {
        # Heavy Structural Concrete (Thermal Mass)
        "Heavy_Concrete": Material(
            roughness=Roughness.medium_rough,
            thickness=0.10,  # 10 cm
            conductivity=1.4,
            density=2200,
            specific_heat=880,
        ),
        # Common Structural Brick / Outer Layer
        "Common_Brick": Material(
            roughness=Roughness.rough,
            thickness=0.10,  # 10 cm
            conductivity=0.89,
            density=1920,
            specific_heat=790,
        ),
        # High-Performance Rigid Insulation (Polyurethane/EPS style)
        "Wall_Roof_Insulation": Material(
            roughness=Roughness.smooth,
            thickness=0.08,  # 8 cm
            conductivity=0.03,
            density=32,
            specific_heat=1200,
        ),
        # Ground Floor Slab Foam Insulation
        "Floor_Insulation": Material(
            roughness=Roughness.smooth,
            thickness=0.05,  # 5 cm
            conductivity=0.035,
            density=35,
            specific_heat=1400,
        ),
        # Interior Plaster/Gypsum Board Finish
        "Gypsum_Board": Material(
            roughness=Roughness.smooth,
            thickness=0.012,  # 1.2 cm Standard dry wall
            conductivity=0.16,
            density=800,
            specific_heat=1090,
        ),
    }


def get_default_double_glazing_window_material_registry(
    name: str = "Default_Double_Glazing_Window",
) -> dict[str, WindowMaterialSimpleGlazingSystem]:
    """Returns a standard double glazing window material configuration map."""
    return {
        name: WindowMaterialSimpleGlazingSystem(
            u_factor=1.8,
            solar_heat_gain_coefficient=0.55,
            visible_transmittance=0.65,
        )
    }


# =====================================================================
# Default standard construction layer assemblies (Outside -> Inside)
# =====================================================================


def get_default_external_wall_construction_registry(
    material_pool: dict[str, Material],
    name: str = DefaultConstructionNames.WALL,
    *,
    layer_outside: str = "Common_Brick",
    layer_insulation: str = "Wall_Roof_Insulation",
    layer_structure: str = "Heavy_Concrete",
    layer_inside: str = "Gypsum_Board",
) -> dict[str, Construction]:
    """Returns a multi-layer external wall construction registry."""
    required_layers = [layer_outside, layer_insulation, layer_structure, layer_inside]
    for layer in required_layers:
        if layer not in material_pool:
            raise ValueError(f"Material '{layer}' assigned to wall construction does not exist.")

    return {
        name: Construction(
            outside_layer=layer_outside,
            layer_2=layer_insulation,
            layer_3=layer_structure,
            layer_4=layer_inside,
        )
    }


def get_default_roof_construction_registry(
    material_pool: dict[str, Material],
    name: str = DefaultConstructionNames.ROOF,
    *,
    layer_outside: str = "Wall_Roof_Insulation",
    layer_structure: str = "Heavy_Concrete",
    layer_inside: str = "Gypsum_Board",
) -> dict[str, Construction]:
    """Returns a multi-layer roof insulation block assembly."""
    required_layers = [layer_outside, layer_structure, layer_inside]
    for layer in required_layers:
        if layer not in material_pool:
            raise ValueError(f"Material '{layer}' assigned to roof construction does not exist.")

    return {
        name: Construction(
            outside_layer=layer_outside, layer_2=layer_structure, layer_3=layer_inside
        )
    }


def get_default_ground_floor_construction_registry(
    material_pool: dict[str, Material],
    name: str = DefaultConstructionNames.FLOOR,
    *,
    layer_outside_ground: str = "Floor_Insulation",
    layer_structure: str = "Heavy_Concrete",
) -> dict[str, Construction]:
    """Returns an insulated ground slab floor assembly."""
    required_layers = [layer_outside_ground, layer_structure]
    for layer in required_layers:
        if layer not in material_pool:
            raise ValueError(
                f"Material '{layer}' assigned to ground floor construction does not exist."
            )

    return {name: Construction(outside_layer=layer_outside_ground, layer_2=layer_structure)}


def get_default_double_glazing_construction_registry(
    name: str = DefaultConstructionNames.WINDOW,
    *,
    window_material_name: str = "Default_Double_Glazing_Window",
) -> dict[str, Construction]:
    """Returns a default construction registry for a double glazing window."""
    return {name: Construction(outside_layer=window_material_name)}


def get_default_base_registry() -> SimulationBaseRegistry:
    """Gathers all standard initialization profiles into the strict Pydantic container."""
    materials_pool = get_default_materials_registry()

    return SimulationBaseRegistry(
        building=get_default_building_registry(),
        global_geometry_rules=get_default_geometry_rules_registry(),
        timestep=get_default_timestep_registry(),
        simulation_control=get_SimulationControl(),
        output_variable=get_default_zone_heating_and_cooling_output_registry(),
        output_sq_lite=get_default_sqlite_output_registry(),
        run_period=get_default_one_year_run_period_registry(),
        material=materials_pool,
        window_material_simple_glazing_system=get_default_double_glazing_window_material_registry(),
        construction={
            **get_default_external_wall_construction_registry(materials_pool),
            **get_default_roof_construction_registry(materials_pool),
            **get_default_ground_floor_construction_registry(materials_pool),
        },
    )


# =====================================================================
# Generate Single zone box geometry
# =====================================================================
def get_single_zone_box_geometry(
    zone_name: str = "box_zone",
    width: float = 10.0,
    length: float = 10.0,
    height: float = 3.0,
    *,
    wall_construction: str = DefaultConstructionNames.WALL,
    roof_construction: str = DefaultConstructionNames.ROOF,
    floor_construction: str = DefaultConstructionNames.FLOOR,
    origin: list[float] | None = None,
) -> dict[str, dict[str, Any]]:
    """Procedural wrapper calling shapes.py to generate a standard single zone box."""
    return generate_cube_zone(
        zone_name=zone_name,
        width=width,
        length=length,
        height=height,
        wall_construction=wall_construction,
        roof_construction=roof_construction,
        floor_construction=floor_construction,
        origin=origin,
    )


# =====================================================================
# Generate Single zone U  Geometry
# =====================================================================
def get_single_zone_u_geometry(
    zone_name: str = "u_zone",
    w_total: float = 20.0,
    l_total: float = 20.0,
    w_wing: float = 6.0,
    l_courtyard: float = 12.0,
    height: float = 3.5,
    *,
    wall_construction: str = DefaultConstructionNames.WALL,
    roof_construction: str = DefaultConstructionNames.ROOF,
    floor_construction: str = DefaultConstructionNames.FLOOR,
    origin: list[float] | None = None,
) -> dict[str, dict[str, Any]]:
    """Procedural wrapper calling shapes.py to generate a single U-shaped zone envelope."""
    return generate_u_shape_zone(
        zone_name=zone_name,
        w_total=w_total,
        l_total=l_total,
        w_wing=w_wing,
        l_courtyard=l_courtyard,
        height=height,
        wall_construction=wall_construction,
        roof_construction=roof_construction,
        floor_construction=floor_construction,
        origin=origin,
    )


# =====================================================================
# Generate 2 box buildings, east and west facing (Separate Outputs)
# =====================================================================
# =====================================================================
# Generate 3 box buildings, west, middle, and east aligned (Separate Outputs)
# =====================================================================
def get_three_box_buildings_geometry(
    west_zone_name: str = "west_building",
    middle_zone_name: str = "middle_building",
    east_zone_name: str = "east_building",
    width: float = 10.0,
    length: float = 10.0,
    height: float = 3.0,
    separation_distance: float = 5.0,
    *,
    wall_construction: str = DefaultConstructionNames.WALL,
    roof_construction: str = DefaultConstructionNames.ROOF,
    floor_construction: str = DefaultConstructionNames.FLOOR,
) -> dict[str, dict[str, dict[str, Any]]]:
    """Generates three independent box building models sharing a global coordinate system.

    Returns a dictionary containing three separate, fully-formed epJSON geometry structures
    under 'west', 'middle', and 'east' keys so they can be compiled into separate IDF files.
    """
    # 1. Build West building at global baseline [0, 0, 0]
    west_bldg = generate_cube_zone(
        zone_name=west_zone_name,
        width=width,
        length=length,
        height=height,
        wall_construction=wall_construction,
        roof_construction=roof_construction,
        floor_construction=floor_construction,
        origin=[0.0, 0.0, 0.0],
    )

    # 2. Build Middle building at global [0, 0, 0] initially
    middle_bldg = generate_cube_zone(
        zone_name=middle_zone_name,
        width=width,
        length=length,
        height=height,
        wall_construction=wall_construction,
        roof_construction=roof_construction,
        floor_construction=floor_construction,
        origin=[0.0, 0.0, 0.0],
    )

    # 3. Build East building at global [0, 0, 0] initially
    east_bldg = generate_cube_zone(
        zone_name=east_zone_name,
        width=width,
        length=length,
        height=height,
        wall_construction=wall_construction,
        roof_construction=roof_construction,
        floor_construction=floor_construction,
        origin=[0.0, 0.0, 0.0],
    )

    # 4. Compute the translation shifts along the X-axis
    # Middle is shifted by one unit of width + separation distance
    middle_shift = width + separation_distance

    # East is shifted by two units of width + two units of separation distance
    east_shift = 2.0 * (width + separation_distance)

    # 5. Translate geometries in place (keeping Zone Origins at 0,0,0)
    translate_zone_geometry(
        middle_bldg, zone_name=middle_zone_name, dx=middle_shift, dy=0.0, dz=0.0
    )
    translate_zone_geometry(east_bldg, zone_name=east_zone_name, dx=east_shift, dy=0.0, dz=0.0)

    # 6. Return them decoupled under semantic keys for isolated serialization
    return {"west": west_bldg, "middle": middle_bldg, "east": east_bldg}
