"""

"""

import sys

from lwrepcoupling import EpLwrSimulationManager


if __name__ == "__main__":
    #
    path_pkl_file = sys.argv[1]
    ep_lwr_sim_man = EpLwrSimulationManager.load_from_pkl(path_pkl_file)
    #
    ep_lwr_sim_man.run_lwr_simulation()