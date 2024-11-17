import numpy as np
from multiprocessing import shared_memory, Manager
from concurrent.futures import ProcessPoolExecutor
import time

def process_task(shm_name, array_shape, str_size, index, barrier, lock):
    """
    Each process performs a loop with:
    1. Writing its own data.
    2. Reading data from shared memory.
    3. Synchronizing with other processes.
    """
    # Access shared memory
    shm = shared_memory.SharedMemory(name=shm_name)
    shared_array = np.ndarray(array_shape, dtype=f'S{str_size}', buffer=shm.buf)

    for step in range(3):  # Example loop with 3 steps
        # Step 1: Write this process's string
        with lock:
            shared_array[index] = f"Process_{index}_Step_{step}".encode('utf-8').ljust(str_size)
            print(f"Process {index} wrote at step {step}")

        # Synchronize all processes before reading
        barrier.wait()

        # Step 2: Read all strings
        combined_string = ""
        for i in range(array_shape[0]):
            combined_string += shared_array[i].decode('utf-8').strip() + " "
        print(f"Process {index} read combined data at step {step}: {combined_string.strip()}")

        # Synchronize all processes before moving to the next step
        barrier.wait()

    shm.close()


# Main function
def main():
    num_processes = 4
    str_size = 64  # Maximum string size (in bytes)
    array_shape = (num_processes,)  # Each process writes one string

    # Create shared memory
    shm = shared_memory.SharedMemory(create=True, size=array_shape[0] * str_size)

    # Use Manager to create synchronization objects (lock and barrier)
    with Manager() as manager:
        barrier = manager.Barrier(num_processes)  # Synchronization barrier
        lock = manager.Lock()  # Lock for synchronization

        try:
            # Initialize the shared memory array
            shared_array = np.ndarray(array_shape, dtype=f'S{str_size}', buffer=shm.buf)
            shared_array[:] = b""  # Clear all entries

            # Start tasks using ProcessPoolExecutor
            with ProcessPoolExecutor() as executor:
                futures = [
                    executor.submit(
                        process_task,
                        shm.name,
                        array_shape,
                        str_size,
                        i,
                        barrier,
                        lock
                    )
                    for i in range(num_processes)
                ]

                # Wait for all processes to complete
                for future in futures:
                    future.result()

        finally:
            # Cleanup
            shm.close()
            shm.unlink()


if __name__ == "__main__":
    main()
