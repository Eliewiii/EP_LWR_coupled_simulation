"""
Simulation with one building. The surface temperature of one surface is monitored and stored in a list.
"""

import os

import matplotlib.pyplot as plt
from src.pyenergyplus.api import EnergyPlusAPI

path_user = r"C:\Users\elie-medioni"  # Technion
# path_user = r"C:\Users\eliem"  # Home


path_one_drive_dir = os.path.join(path_user, r"OneDrive\OneDrive - Technion")  # Technion
# path_one_drive_dir = os.path.join(path_user,r"C:\Users\eliem\OneDrive - Technion")  # Home


# Set this to your actual EnergyPlus installation directory
energyplus_dir = r"C:\EnergyPlusV23-2-0"

# Path to your EnergyPlus directory and IDF file
idf_file = os.path.join(path_one_drive_dir,
                        r"BUA\LWR\pyenergyplus\sample_cube_buildings\IDFs\cube_1\openstudio\run\in.idf")
epw_file = os.path.join(path_user,
                        r"AppData\Local\Building_urban_analysis\Libraries\EPW\IS_5280_A_Tel_Aviv.epw")  # Replace with your weather file path
output_dir = os.path.join(path_one_drive_dir,
                          r"BUA\LWR\pyenergyplus\sample_cube_buildings\IDFs\cube_1\pyenergyplus")  # Replace with your output directory

# Initialize the EnergyPlus API
api = EnergyPlusAPI(running_as_python_plugin=True, path_to_ep_folder=energyplus_dir)

# Set the surface name you want to monitor
surface_name = "Face: room_1_bf4c06dc..Face0"

# Initialize a list to store temperature data
surface_temps = []
outdoor_temps = []
time_steps = []


# Define a callback function to interact with EnergyPlus
def my_callback(state):
    # Get current simulation time (in hours)
    current_time = api.exchange.current_sim_time(state)
    time_steps.append(current_time)
    api.exchange.get_api_data(state)

    # Get the handle for the surface temperature output variable
    surface_temp_handle = api.exchange.get_variable_handle(state, "SURFACE OUTSIDE FACE TEMPERATURE",
                                                           surface_name)
    outdoor_temp_handle = api.exchange.get_variable_handle(
        state, "SITE OUTDOOR AIR DRYBULB TEMPERATURE", "ENVIRONMENT"
    )

    # # Check if the handle is valid
    # if surface_temp_handle > 0:
    # Get the current surface temperature
    surface_temp = api.exchange.get_variable_value(state, surface_temp_handle)
    outdoor_temp = api.exchange.get_variable_value(state, outdoor_temp_handle)

    # Store the surface temperature in the list
    surface_temps.append(surface_temp)
    outdoor_temps.append(outdoor_temp)


# Register the callback for the 'end of zone timestep' event
state = api.state_manager.new_state()
api.runtime._set_energyplus_root_directory(state, path := energyplus_dir)
api.runtime.callback_end_zone_timestep_after_zone_reporting(state=state, f=my_callback)
# set the variables to be requested, otherwise they won't be available in the callback
api.exchange.request_variable(state, "SITE OUTDOOR AIR DRYBULB TEMPERATURE", "ENVIRONMENT")
api.exchange.request_variable(state, "SURFACE OUTSIDE FACE TEMPERATURE", surface_name)
# Run the EnergyPlus simulation
api.runtime.run_energyplus(state,
                           ['-r',  # Run annual simulation
                            '-w', epw_file,  # Weather file
                            '-d', output_dir,  # Output directory
                            idf_file]  # Input IDF file
                           )

# Plotting the collected surface temperature after the simulation
plt.figure(figsize=(10, 6))
times = list(range(len(surface_temps)))  # Assuming each timestep corresponds to an index in the list
plt.plot(times, surface_temps, label=surface_name)

plt.xlabel('Time Steps')
plt.ylabel('Surface Temperature (°C)')
plt.title(f'Surface Temperature of {surface_name} Over Time')
plt.legend()
plt.grid()
plt.show()

# Plotting the collected surface temperature after the simulation
plt.figure(figsize=(10, 6))
plt.plot(time_steps, outdoor_temps, label="Outdoor")
plt.xlabel('Time Steps')
plt.ylabel('Temperature (°C)')
plt.title(f'Temperature of {surface_name} Over Time')

plt.legend()
plt.grid()
plt.show()

print("ok")
