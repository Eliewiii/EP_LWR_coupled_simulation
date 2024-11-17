import numpy as np
from multiprocessing import shared_memory, Manager
from concurrent.futures import ProcessPoolExecutor
import time

def process_task(shm_name, array_shape, lock, barrier, index):
    """
    Each process performs:
    1. Writing its own list of floats.
    2. Reading data from shared memory.
    3. Synchronizing with other processes.
    """
    shm = shared_memory.SharedMemory(name=shm_name)
    shared_array = np.ndarray(array_shape, dtype=np.float64, buffer=shm.buf)

    # Example: Process writes its own float list (different lengths)
    float_list = np.random.rand(np.random.randint(5, 10))  # Random list of floats with length between 5 and 10
    print(f"Process {index} generated float list: {float_list}")

    # Step 1: Write data to shared memory
    with lock:
        # Here we are writing the list as a slice of the shared memory
        shared_array[index:index+len(float_list)] = float_list
        print(f"Process {index} wrote data at index {index}")

    # Synchronize all processes before reading
    barrier.wait()

    # Step 2: Read all data from shared memory
    combined_data = []
    for i in range(array_shape[0]):
        combined_data.append(shared_array[i])  # Collect all data
    print(f"Process {index} read combined data: {combined_data}")

    # Synchronize all processes before finishing the task
    barrier.wait()
    print(f"Process {index} finished task")

    shm.close()


def main():
    num_processes = 4
    array_shape = (100,)  # Fixed size for shared memory buffer (adjust for actual needs)

    # Use Manager to create a shared Lock and Barrier
    with Manager() as manager:
        lock = manager.Lock()
        barrier = manager.Barrier(num_processes)

        # Create shared memory for float64 data (enough for all processes' lists)
        shm = shared_memory.SharedMemory(create=True, size=array_shape[0] * np.float64().itemsize)

        try:
            # Start tasks
            with ProcessPoolExecutor() as executor:
                futures = [
                    executor.submit(
                        process_task,
                        shm.name,
                        array_shape,
                        lock,
                        barrier,
                        i
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
