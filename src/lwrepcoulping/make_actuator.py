"""
Functions to create actuator that will update the temperature schedule of context surfaces.
"""

from src.pyenergyplus.api import EnergyPlusAPI


def make_constant_schedule_actuator(api: EnergyPlusAPI, state, schedule_name: str) -> str:
    """
    Create an actuator handle for a constant schedule in EnergyPlus.

    :param api: The EnergyPlus API instance
    :param state: The EnergyPlus simulation state
    :param schedule_name: The name of the schedule to actuate
    :return: The handle for the schedule's value
    """

    # Retrieve the actuator handle for the Schedule:Constant object
    handle = api.exchange.get_actuator_handle(state, "Schedule:Constant", schedule_name, "Schedule Value")
    return handle


def update_actuator_value(api: EnergyPlusAPI, state, handle: str, value: float):
    """
    Update the value of an actuator in EnergyPlus.

    :param api: The EnergyPlus API instance
    :param state: The EnergyPlus simulation state
    :param handle: The handle of the actuator to update
    :param value: The new value to set
    """

    # Set the value of the actuator
    api.exchange.set_actuator_value(state, handle, value)
