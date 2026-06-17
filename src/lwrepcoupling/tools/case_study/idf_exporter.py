"""
Module for dynamically exporting Pydantic EnergyPlusModel objects to legacy IDF text format using
"""

from pathlib import Path
from typing import Any

from pyenergyplus.model import EnergyPlusModel


def _format_value(val: Any) -> str:
    """Translates primitives and Enums to safe EnergyPlus text literals."""
    if val is None:
        return ""
    if hasattr(val, "value"):  # Handles explicit Enum parameters cleanly
        return str(val.value)
    if isinstance(val, bool):
        return "Yes" if val else "No"
    return str(val)


def export_model_to_idf_string(model: Any) -> str:
    """Converts a compiled EnergyPlusModel straight into an index-compliant IDF text sequence.

    Preserves strict index alignment for component metadata while dynamically trimming empty
    trailing material layers within Construction blocks to ensure execution safety.
    """
    sections = []
    model_dict = model.model_dump(exclude_none=False, by_alias=True)

    # Class categories that strictly DO NOT accept an identifying object name string field in
    # position #1
    NO_NAME_CLASSES = {
        "Version",
        "SimulationControl",
        "GlobalGeometryRules",
        "Timestep",
        "Output:SQLite",
        "Output:Variable",
    }

    for class_name, instances in model_dict.items():
        if not instances or not isinstance(instances, dict):
            continue

        # Map programmatic class names back to official legacy syntax conventions
        idf_class_name = class_name
        if class_name == "BuildingSurfaceDetailed":
            idf_class_name = "BuildingSurface:Detailed"
        elif class_name == "OutputSqLite":
            idf_class_name = "Output:SQLite"
        elif class_name == "OutputVariable":
            idf_class_name = "Output:Variable"

        for obj_name, obj_data in instances.items():
            if not isinstance(obj_data, dict):
                continue

            block_lines = [f"{idf_class_name},"]
            field_items = []

            # Inject the object name identifier key if the class belongs to a standard component
            # group
            if idf_class_name not in NO_NAME_CLASSES:
                field_items.append(("Name", obj_name))

            # Process properties ensuring index positions match official schema layouts
            for f_key, f_val in obj_data.items():
                if f_key.lower() == "name":
                    continue

                # --- CONSTRUCTION DYNAMIC TRIMMING LAYER ---
                # If we are in a Construction block and hit an empty layer field, skip it entirely
                if (
                    idf_class_name == "Construction"
                    and f_key.lower().startswith("layer_")
                    and f_val is None
                ):
                    continue

                # Compress vertices into combined X, Y, Z lines on a single row
                if f_key == "vertices" and isinstance(f_val, list):
                    for v_idx, vertex in enumerate(f_val):
                        v_num = v_idx + 1
                        x = _format_value(vertex.get("vertex_x_coordinate"))
                        y = _format_value(vertex.get("vertex_y_coordinate"))
                        z = _format_value(vertex.get("vertex_z_coordinate"))

                        coord_str = f"{x}, {y}, {z}"
                        field_items.append((f"X,Y,Z Vertex {v_num} [m]", coord_str))
                else:
                    field_items.append((f_key, f_val))

            # Assemble block rows applying correct terminal commas and semicolons
            for idx, (f_name, f_val) in enumerate(field_items):
                is_last = idx == len(field_items) - 1
                symbol = ";" if is_last else ","

                # Format the value property cleanly
                fmt_val = "" if f_val is None else _format_value(f_val)

                # Join the delimiter symbol directly to the end of the text string value
                val_with_symbol = f"{fmt_val}{symbol}"

                # Apply the 40-character column padding to the combined token string block
                block_lines.append(f"  {val_with_symbol:<40} !- {f_name}")

            sections.append("\n".join(block_lines))

    return "\n\n".join(sections) + "\n"


def write_native_idf(model: EnergyPlusModel, target_path: Path, additional_text: str = "") -> Path:
    """
    Writes a compiled EnergyPlusModel straight to an IDF file on disk, appending coupling
    modifications.
    """
    target_path.parent.mkdir(parents=True, exist_ok=True)

    idf_content = export_model_to_idf_string(model)

    if additional_text:
        idf_content += "\n" + additional_text.strip() + "\n"

    target_path.write_text(idf_content, encoding="utf-8")
    return target_path
