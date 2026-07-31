"""Task registry — maps task IDs to TaskSpec implementations."""
from machina_arena.task_spec import TaskSpec
from machina_arena.tasks.pickcube_so100 import PickCubeSO100Task

_TASKS = {
    "pickcube-v1": PickCubeSO100Task,
}


def get_task(task_id: str) -> TaskSpec:
    """Get a task instance by ID."""
    if task_id not in _TASKS:
        raise ValueError(f"Unknown task: {task_id}. Available: {list(_TASKS.keys())}")
    return _TASKS[task_id]()


def list_tasks() -> list[str]:
    """List all available task IDs."""
    return list(_TASKS.keys())
