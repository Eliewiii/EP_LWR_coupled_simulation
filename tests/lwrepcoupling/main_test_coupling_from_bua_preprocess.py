"""

"""

import os

from lwrepcoupling import EpLwrSimulationManager

if __name__ == "__main__":
    path_ep_lwr_simulation_manager_pkl = r"C:\Users\elie-medioni\AppData\Local\BUA\Simulation_temp\LWR\EP_sim\ep_lwr_simulation_manager.pkl"

    ep_lwr_sim_man = EpLwrSimulationManager.from_pkl(path_ep_lwr_simulation_manager_pkl)

    ep_lwr_sim_man.run_lwr_coupled_simulation()
