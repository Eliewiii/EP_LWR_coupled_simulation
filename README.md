# LWR-EPCoupling: Long-Wave Radiation EnergyPlus Coupling

![Python](https://img.shields.io/badge/python-3.12+-3670A0?style=for-the-badge&logo=python&logoColor=ffdd54)
![EnergyPlus](https://img.shields.io/badge/EnergyPlus-V26.1.0-005587?style=for-the-badge&logo=energyplus&logoColor=white)
![HPC](https://img.shields.io/badge/HPC-Parallel%20Computing-blueviolet?style=for-the-badge)
![pyenergyplus](https://img.shields.io/badge/pyenergyplus-API-005587?style=for-the-badge)
![Pydantic](https://img.shields.io/badge/Pydantic-Validation-e92063?style=for-the-badge)
![Pytest](https://img.shields.io/badge/Pytest-Testing-0A9EDC?style=for-the-badge)
![GitHub Actions](https://img.shields.io/badge/CI-GitHub_Actions-2088FF?style=for-the-badge&logo=github-actions&logoColor=white)
![Ruff](https://img.shields.io/badge/Linter-Ruff-FCC21B?style=for-the-badge)
![Pyright](https://img.shields.io/badge/Types-Pyright-3178C6?style=for-the-badge)
![License](https://img.shields.io/badge/license-MIT-green?style=for-the-badge)

**LWR-EPCoupling** is a high-performance Python framework designed to bridge the gap between individual building energy models and district-scale radiative reality. By leveraging the **EnergyPlus API** through the standard `pyenergyplus` package, it enables synchronized, multi-building simulations that account for inter-building long-wave radiation (LWR) exchanges in real-time.

---

## 🚀 Key Features

### 1. High-Performance Computing (HPC) & Architecture
* **Parallel Execution:** Orchestrates multiple EnergyPlus instances simultaneously using advanced multiprocessing `spawn` contexts for strict memory safety.
* **Zero-Copy Shared Memory:** Efficiently shares surface temperature data across processes via OS-level `SharedMemory` buffers, minimizing I/O overhead during radiative balance calculations.
* **Self-Healing Synchronization:** Implements strict `multiprocessing.Barrier` synchronizers across the C++ callback loops to ensure physical consistency at every simulation interval, with active polling to prevent deadlocks if a child process crashes.

### 2. Scientific Modeling & Boundary Forcing
* **Radiosity Formulation:** Resolves the global radiation balance for all outdoor surfaces by calculating the net long-wave flux between building instances in the urban canopy.
* **Dynamic Boundary Forcing:** Implements a novel coupling logic to bypass the limitations of native building engines. The framework calculates the net radiative exchange and converts it into an **Equivalent Surrounding Surface Temperature ($T_{eq}$)**.
* **API Injection:** This $T_{eq}$ value is "forced" into the *EnergyPlus* engine via API Schedule Actuators at every timestep, overriding default boundary conditions to reflect the real-time thermal state of neighboring facades.
* **UHI Research Integration:** Models the critical radiative trapping component of the **Urban Heat Island (UHI)** effect. For comprehensive UHI analysis, this tool can be seamlessly coupled with atmospheric tools like the Urban Weather Generator (UWG) that provide urban-adjusted `.epw` forcing files.

### 3. Industry Best Practices & Reliability
* **Robust OOP & Validation:** Built entirely on modern Object-Oriented paradigms with strict schema validation powered by `Pydantic`.
* **Exhaustive Testing:** Features a deep CI/CD testing pipeline, including unit tests, integration tests, and **heavy un-mocked system simulations** utilizing dynamically generated `pyenergyplus` model structures to guarantee stability against the real native C++ engine.
* **Type Safety & Linting:** Enforces strict code quality using Ruff and Pyright for complete runtime integrity.

### 4. Integration & Extensibility
* **Standard Python Ecosystem:** Runs smoothly within standard Python virtual environments utilizing the native `pyenergyplus` library.
* **Flexible Data Ingestion:** Compatible with pre-computed view factors from any source, with native support for the **[View_Factor_Computation_With_Radiance](https://github.com/Eliewiii/View_Factor_Computation_With_Radiance)** package.

---

## 🛠️ Prerequisites

* **Python:** 3.12 or higher.
* **EnergyPlus™:** Version **26.1.0** installed natively on your workstation (which exposes the `pyenergyplus` Python bindings).
* **Simulation Inputs:** Requires dynamically generated or pre-existing EnergyPlus model components, valid weather files (`.epw`), and a pre-computed sparse View Factor matrix (`.npz`).

---

## 📦 Installation

**LWR-EPCoupling** is designed to be installed directly into your standard Python virtual environment without modifying the base EnergyPlus installation.

1. Ensure your system meets the Python and EnergyPlus prerequisites.
2. Download the latest stable release from the **[Releases](../../releases)** page.
3. Install the package using `pip`:

``` bash
# Install directly from the downloaded release wheel
pip install lwrepcoupling-x.x.x-py3-none-any.whl
```

*Note: If you are setting up the project for local development or running the test suite, clone the repository and install it in editable mode:*

``` bash
git clone https://github.com/Eliewiii/LWR-EPCoupling.git
cd LWR-EPCoupling
pip install -e .
```

---

## 🏗️ Architecture & Logic

This package functions as a defensive "Master Controller" for a distributed simulation ensemble.

1.  **Workspace Compilation (`EpLwrSimulationManager`):** Safely validates the execution environment, isolates individual building sandbox directories, loads global CSR sparse matrices, and mathematically slices customized dense resolution matrices for each building.
2.  **Process Spawning:** Launches highly isolated `EpSimulationRuntimeWorker` processes to prevent C++ state memory corruption across building instances.
3.  **Timestep Interception:** Using the `callback_end_zone_timestep_after_zone_reporting` API hook, the simulation pauses at the exact same timestep across all isolated building processes.
4.  **Shared Memory Balance:** Surface temperatures ($T^4$) are flushed to the OS shared memory block. A global radiosity matrix is instantly solved to find the equivalent surrounding temperatures.
5.  **Feedback Injection:** Updated $T_{eq}$ boundary conditions are injected back into EnergyPlus via `set_actuator_value` before the Barrier is released and the next timestep proceeds.

---

## 📂 Project Structure

* `src/lwrepcoupling/`: Core coupling engine, state managers, schema validation, and `pyenergyplus` API wrappers.
* `tests/`: Extensive Pytest suite covering isolated unit tests, IO bounds, and full-scale un-mocked parallel simulation benchmarks.
* `docs/`: Technical documentation and theoretical background on inter-building long-wave radiation *(derived from thesis, to be added soon)*.

---

## 🔮 Future Development

As the package transitions from its legacy thesis architecture to this robust, production-grade framework, near-term development goals include:
* **Standardization:** Minor adjustments across all docstrings to strictly align with Google Python Style Guide standards.
* **Testing Expansion:** Continued expansion of the unit and edge-case testing coverage.
* **Physics Validation Pipelines:** Implementation of automated simulation testing to cross-verify output metrics (surface temperatures and energy consumption fluxes) against the established baseline results from the original Ph.D. research.
* **Ground Surface Coupling:** Extending the inversion logic to natively account for dynamic inter-reflections from the urban ground plane.

---

## 🎓 Context & Credits

**Author:** Elie Medioni, Ph.D.  
**Institution:** Technion - Israel Institute of Technology  

This package was developed as part of a Ph.D. : **Toward Fully Automated Inter-Building Coupling in Urban Energy Simulation: From Building Models to Integrated Longwave Radiation Workflows**. 

**Technical Verification:** Special thanks to Ivan Girault for his help verifying the theoretical modeling and the long-wave radiation equation systems in the initial research phase.

---

## 📄 License

This project is licensed under the **MIT License**. It interfaces with `pyenergyplus` (EnergyPlus™), subject to the original EnergyPlus™ license agreement.