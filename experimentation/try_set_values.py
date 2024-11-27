"""
Simulation with one building. The surface temperature of one surface is monitored and stored in a list.
"""

import os

from random import random

import matplotlib.pyplot as plt
from src.lwrepcoupling.pyenergyplus.api import EnergyPlusAPI

path_user = r"C:\Users\elie-medioni"  # Technion
# path_user = r"C:\Users\eliem"  # Home


path_one_drive_dir = os.path.join(path_user, r"OneDrive\OneDrive - Technion")  # Technion
# path_one_drive_dir = os.path.join(path_user,r"C:\Users\eliem\OneDrive - Technion")  # Home


# Set this to your actual EnergyPlus installation directory
energyplus_dir = r"C:\EnergyPlusV23-2-0"

# Path to your EnergyPlus directory and IDF file
idf_file = os.path.join(path_one_drive_dir,
                        r"BUA\LWR\pyenergyplus\sample_cube_buildings\IDFs\cube_1_with_surrounding_surface\openstudio\run\in.idf")
epw_file = os.path.join(path_user,
                        r"AppData\Local\Building_urban_analysis\Libraries\EPW\IS_5280_A_Tel_Aviv.epw")  # Replace with your weather file path
output_dir = os.path.join(path_one_drive_dir,
                          r"BUA\LWR\pyenergyplus\sample_cube_buildings\IDFs\cube_1_with_surrounding_surface\pyenergyplus")  # Replace with your output directory

api = EnergyPlusAPI(running_as_python_plugin=True, path_to_ep_folder=energyplus_dir)

state = api.state_manager.new_state()

time_steps = []
schedule_values = []
schedule_name = "TEST_CONSTANT_SCHEDULE"
schedule_actuator_handle = None

def initialize_actuator(state_argument):
    global schedule_actuator_handle
    schedule_actuator_handle = api.exchange.get_actuator_handle(state_argument, "Schedule:Constant",
                                                                "Schedule Value", schedule_name)
    print (f"schedule_actuator_handle = {schedule_actuator_handle}")
    if schedule_actuator_handle == -1:
        raise ValueError(f"Failed to create actuator for schedule {schedule_name}")

def modify_and_record_schedule_value(state_argument):
    # Get current simulation time (in hours)
    current_time = api.exchange.current_sim_time(state)
    time_steps.append(current_time)

    new_value = 22.0+ random()



    api.exchange.set_actuator_value(state_argument, schedule_actuator_handle, new_value)
    handle = api.exchange.get_variable_handle(state_argument, "Schedule Value", schedule_name)
    current_value = api.exchange.get_variable_value(state_argument, handle)
    schedule_values.append(current_value)
    print(f"Timestep {current_time}: Set value = {new_value}, Recorded value = {current_value}")

# Request the variable to access schedule values during the simulation
api.exchange.request_variable(state, "Schedule Value", schedule_name)

api.runtime.callback_after_new_environment_warmup_complete(state, initialize_actuator)
# api.runtime.callback_after_predictor_after_hvac_managers(state, modify_and_record_schedule_value)
api.runtime.callback_after_predictor_before_hvac_managers(state, modify_and_record_schedule_value)

# Run the EnergyPlus simulation
api.runtime.run_energyplus(state,
                           ['-r',  # Run annual simulation
                            '-w', epw_file,  # Weather file
                            '-d', output_dir,  # Output directory
                            idf_file]  # Input IDF file
                           )

# Plotting the collected surface temperature after the simulation
plt.figure(figsize=(10, 6))
plt.plot(time_steps, schedule_values, label=schedule_name)

plt.xlabel('Time Steps')
plt.ylabel('Surface Temperature (°C)')
plt.title(f'Surface Temperature of {schedule_name} Over Time')
plt.legend()
plt.grid()
plt.show()



print("ok")
