# LWR-EPCoupling: Long-Wave Radiation EnergyPlus Coupling

![Python](https://img.shields.io/badge/python-3670A0?style=for-the-badge&logo=python&logoColor=ffdd54)
![HPC](https://img.shields.io/badge/HPC-Parallel%20Computing-blueviolet?style=for-the-badge)
![EnergyPlus](https://img.shields.io/badge/EnergyPlus-V23.2.0-005587?style=for-the-badge&logo=energyplus&logoColor=white)
![License](https://img.shields.io/badge/license-MIT-green?style=for-the-badge)

**LWR-EPCoupling** is a high-performance Python framework designed to bridge the gap between individual building energy models and district-scale radiative reality. By leveraging the **EnergyPlus API** through `pyenergyplus`, it enables synchronized, multi-building simulations that account for inter-building long-wave radiation (LWR) exchanges in real-time.

---
> **⚠️ Development Note**
> 
> This package contains a modified implementation of `pyenergyplus` to enable seamless execution within standard Python virtual environments. It is currently optimized for **EnergyPlus V23-2-0**.
> Future update will use `pyenergyplus` as a package.
---

## 🚀 Key Features

### 1. High-Performance Computing (HPC)
*   **Parallel Execution:** Orchestrates multiple EnergyPlus instances simultaneously using advanced concurrency patterns.
*   **Shared Memory Management:** Efficiently shares surface temperature data across processes to minimize I/O overhead during radiative balance calculations.
*   **Synchronization Barriers:** Implemented strict timestep synchronization to ensure physical consistency across the building ensemble at every simulation interval.

### 2. Scientific Modeling & Boundary Forcing
*   **Radiosity Formulation:** Resolves the global radiation balance for all outdoor surfaces by calculating the net long-wave flux between building instances in the urban canopy.
*   **Dynamic Boundary Forcing:** Implements a novel coupling logic to bypass the limitations of native building engines. The framework calculates the net radiative exchange and converts it into an **Equivalent Surrounding Surface Temperature ($T_{eq}$)**.
*   **API Injection:** This $T_{eq}$ value is "forced" into the *EnergyPlus* engine via the API at every timestep, overriding default boundary conditions to reflect the real-time thermal state of neighboring facades.
*   **UHI Research:** Enables high-fidelity modeling of the **Urban Heat Island (UHI)** effect by accounting for active heat exchange rather than static shading.

### 3. Integration & Extensibility
*   **EnergyPlus API (pyenergyplus):** Direct integration with the EnergyPlus engine for high-fidelity thermal results.
*   **Flexible Data Ingestion:** Compatible with pre-computed view factors from any source, with native support for the **[View_Factor_Computation_With_Radiance](https://github.com/Eliewiii/View_Factor_Computation_With_Radiance)** package.


### 4. Current Limitations
*   The framework currently focuses on forcing radiative flux from **building-to-building interactions** only. 
*   Ground and sky exposures are treated using standard *EnergyPlus* algorithms; consequently, inter-reflections from the ground and sky are not yet accounted for in the coupled balance.
*   **Roadmap:** Future development includes extending the coupling logic to include **ground surfaces**, allowing for a full-district radiative closure.
---

## 🛠️ Prerequisites

*   **EnergyPlus™:** (Tested on V23-2-0)..
*   **Simulation Inputs:** Requires EnergyPlus model files (`.idf`), weather files (`.epw`), and a pre-computed View Factor matrix.

---

## 🏗️ Architecture & Logic

This package functions as the "Master Controller" for a distributed simulation.

1.  **Initialization:** Spawns individual building simulation processes.
2.  **Timestep Interception:** At each timestep, the simulation pauses to collect surface temperatures.
3.  **Radiation Balance:** A global radiosity matrix is solved to find the net heat flux for every building surface.
4.  **Feedback Loop:** Updated boundary conditions are injected back into EnergyPlus before the next timestep proceeds.

## 📂 Project Structure

* `src/`: Core coupling logic and modified `pyenergyplus` API wrappers.
* `scripts/`: Implementation of the LWR mathematical solver and radiosity balance logic.
* `tests/`: Validation cases for coupled thermal balance and synchronization tests.
* `docs/`: Technical documentation and theoretical background on inter-building long-wave radiation (to be updated).

---

## 🎓 Context & Credits

**Author:** Elie Medioni, Ph.D.
**Institution:** Technion - Israel Institute of Technology  

This package was developed as part of a Ph.D. thesis focusing on **Urban Building Energy Modeling (UBEM)** and the **Urban Heat Island (UHI)** effect. 

**Technical Verification:** Special thanks to Ivan Girault for his help verifying the theoretical modeling and the long-wave radiation equation systems.

---

## 📄 License

This project is licensed under the **MIT License**. It includes modified source code from `pyenergyplus` (EnergyPlus™), subject to the original EnergyPlus™ license agreement.
