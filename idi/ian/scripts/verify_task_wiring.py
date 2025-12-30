
import asyncio
import sys
import logging
from idi.ian.network.production import TaskSupervisor, TaskState, SupervisedTask

# Setup Init logging to see kernel errors if any
logging.basicConfig(level=logging.INFO)

async def test_task_wiring():
    print("Test: Task Wiring Lifecycle")
    supervisor = TaskSupervisor("test_sup")
    
    # Define a task that crashes once then succeeds (or just runs)
    # Actually, let's make it crash until stopped
    crash_counter = 0
    
    async def crasher():
        nonlocal crash_counter
        crash_counter += 1
        print(f"  Task running (attempt {crash_counter})")
        if crash_counter <= 2:
            raise RuntimeError("Crash!")
        return "Success"

    # Spawn task with 3 restarts max
    supervisor.spawn(
        crasher, 
        name="crasher_task", 
        restart_on_failure=True, 
        max_restarts=3,
        restart_delay=0.1
    )
    
    task_obj = supervisor._tasks["crasher_task"]
    
    # 1. Verify Init State
    assert task_obj._kstate.state == "PENDING"
    assert task_obj.state == TaskState.PENDING
    print("  Init State: PENDING - OK")

    # 2. Wait for it to start
    # Supervisor starts generic task in background? 
    # spawn calls _start_task immediately.
    # Give it time to crash and restart
    await asyncio.sleep(0.5)
    
    # We expect: 
    # Attempt 1 -> Crash -> Restarting
    # Attempt 2 -> Crash -> Restarting
    # Attempt 3 -> Success -> Stopped
    
    print(f"  Final State: {task_obj.state}")
    print(f"  Final Restarts: {task_obj.restart_count}")
    
    # Verify Restart Logic
    # Should have restarted 2 times
    # crash_counter should be 3 (1 initial + 2 restarts)
    assert crash_counter == 3, f"Expected 3 runs, got {crash_counter}"
    assert task_obj.restart_count == 2, f"Expected 2 restarts, got {task_obj.restart_count}"
    assert task_obj.state == TaskState.STOPPED, f"Expected STOPPED, got {task_obj.state}"
    
    # Check Kernel State
    assert task_obj._kstate.state == "STOPPED"
    print("  Kernel sync verified - OK")
    
    await supervisor.shutdown()
    print("ALL TESTS PASSED")

if __name__ == "__main__":
    asyncio.run(test_task_wiring())
