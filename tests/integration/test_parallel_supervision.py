"""Integration tests verifying parallel worker tracking loops and failure-containment networks."""

import unittest.mock as mock

import pytest
from lwrepcoupling.ep_coupled_simulation_manager import EpLwrSimulationManager


@pytest.mark.integration
def test_simulation_supervisor_aborts_barrier_on_worker_crash(mock_manifest) -> None:
    """
    Verify that a single background process crash forces the polling engine to
    collapse the synchronization barrier.
    """
    manager = EpLwrSimulationManager(mock_manifest)

    # 1. Create fake process objects to simulate background workers
    mock_alive_process = mock.Mock()
    mock_alive_process.is_alive.side_effect = [
        True,
        True,
        False,
    ]  # Alive for two ticks, then exits cleanly
    mock_alive_process.exitcode = 0

    mock_crashing_process = mock.Mock()
    mock_crashing_process.is_alive.return_value = False  # Exited immediately
    mock_crashing_process.exitcode = 1  # CRASH EXIT CODE!

    # 2. Inject mock processes sequentially when the engine attempts to boot them
    with (
        mock.patch("multiprocessing.ctx_spawn.SpawnContext.Process") as mock_process_spawn,
        mock.patch("multiprocessing.managers.Barrier") as mock_barrier_class,
        mock.patch("multiprocessing.shared_memory.SharedMemory"),
    ):
        # Setup mock barrier instance to monitor if .abort() gets called
        mock_barrier_instance = mock.Mock()
        mock_barrier_class.return_value = mock_barrier_instance

        # The manager has 1 building in mock_manifest, let's patch it to test multi-process routing:
        with mock.patch.object(manager, "num_buildings", new=2):
            mock_process_spawn.side_effect = [mock_alive_process, mock_crashing_process]

            # Execute the orchestrator
            exit_status = manager.run_lwr_coupled_simulation()

            # 3. VERIFICATIONS:
            # The global status must capture failure (return 1)
            assert exit_status == 1
            # The supervisor must actively collapse the barrier to prevent hanging deadlocks!
            mock_barrier_instance.abort.assert_called_once()
