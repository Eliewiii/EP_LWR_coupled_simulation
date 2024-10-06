import os

import matplotlib.pyplot as plt
from src.pyenergyplus.api import EnergyPlusAPI


# Set this to your actual EnergyPlus installation directory
energyplus_dir = r"C:\EnergyPlusV23-2-0"

# Path to your EnergyPlus directory and IDF file
idf_file = r"C:\path\to\your\file.idf"  # Replace with your actual IDF file path
epw_file = r"C:\path\to\your\weather.epw"  # Replace with your weather file path
output_dir = r"C:\path\to\output\directory"  # Replace with your output directory

# Initialize the EnergyPlus API
api = EnergyPlusAPI()

# Set the surface name you want to monitor
surface_name = "Surface-Name"  # Replace with the actual surface name from your IDF

# Initialize a list to store temperature data
surface_temps = []


# Define a callback function to interact with EnergyPlus
def my_callback(state):
    # Get current simulation time (in hours)
    current_time = api.exchange.current_time(state)

    # Get the handle for the surface temperature output variable
    surface_temp_handle = api.exchange.get_variable_handle(state, "Surface Inside Temperature", surface_name)

    # Check if the handle is valid
    if surface_temp_handle > 0:
        # Get the current surface temperature
        surface_temp = api.exchange.get_variable_value(state, surface_temp_handle)

        # Store the surface temperature in the list
        surface_temps.append(surface_temp)


# Register the callback for the 'end of zone timestep' event
api.runtime.callback_end_of_zone_timestep_after_zone_reporting(state=my_callback)

# Run the EnergyPlus simulation
api.runtime.run_energyplus(
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