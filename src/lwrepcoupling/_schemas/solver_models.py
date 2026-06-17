from typing import Annotated

from pydantic import BaseModel, Field, model_validator


class SurfaceAddStringConfig(BaseModel):
    surface_name: Annotated[str, Field(min_length=1)]
    cumulated_ext_surf_view_factor: Annotated[float, Field(ge=0, le=1)]
    sky_view_factor: Annotated[float | None, Field(ge=0, le=1)] = None
    ground_view_factor: Annotated[float | None, Field(ge=0, le=1)] = None
    ground_temperature_schedule: str = ""

    @model_validator(mode="after")
    def check_total_view_factor(self) -> "SurfaceAddStringConfig":
        # Treat None as 0.0 for the summation logic
        sky = self.sky_view_factor or 0.0
        ground = self.ground_view_factor or 0.0
        ext = self.cumulated_ext_surf_view_factor

        total_vf = sky + ground + ext

        if total_vf > 1.0:
            raise ValueError(
                f"Total view factor for {self.surface_name} exceeds 1.0: "
                f"Sum is {total_vf:.4f} (Sky: {sky}, Ground: {ground}, Ext: {ext})"
            )

        return self
