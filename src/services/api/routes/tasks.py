"""
File: /tasks.py
Created Date: Saturday December 13th 2025
Author: Christian Nonis <alch.infoemail@gmail.com>
-----
Last Modified: Saturday December 13th 2025
Modified By: the developer formerly known as Christian Nonis at <alch.infoemail@gmail.com>
-----
"""

import json
from src.utils.logging import log
from fastapi import APIRouter, Depends, HTTPException
from src.services.api.dependencies import get_brain_id
from src.services.kg_agent.main import cache_adapter
from src.services.api.constants.responses import TaskListResponse, TaskStateResponse

tasks_router = APIRouter(prefix="/tasks", tags=["tasks"])


@tasks_router.get("/", response_model=TaskListResponse)
async def get_tasks(brain_id: str = Depends(get_brain_id)):
    try:
        task_keys = cache_adapter.get_task_keys(brain_id)
        results = []
        for task_key in task_keys:
            task_id = task_key.split(":")[-1]
            str_result = cache_adapter.get_task(task_id, brain_id=brain_id)
            if str_result is None:
                continue
            result = json.loads(str_result)
            results.append(
                {
                    **result,
                    "id": task_id,
                    "status": result.get("status", "unknown"),
                }
            )
        return {"tasks": results}
    except Exception as e:
        log(f"Error in get_tasks: {type(e).__name__}: {str(e)}")
        raise HTTPException(
            status_code=500,
            detail="Could not retrieve tasks",
        ) from e


@tasks_router.get("/{task_id}", response_model=TaskStateResponse)
async def get_task(task_id: str, brain_id: str = Depends(get_brain_id)):
    """
    Get the result of a task by its ID.
    """
    try:
        str_result = cache_adapter.get_task(task_id, brain_id=brain_id)
        if str_result is None:
            raise HTTPException(status_code=404, detail="Task not found")
        if isinstance(str_result, bytes):
            result = json.loads(str_result.decode("utf-8"))
        else:
            result = json.loads(str_result)
        return {
            **result,
            "task_id": task_id,
            "status": result.get("status", "unknown"),
        }
    except HTTPException:
        raise
    except Exception as e:
        log(f"Error in get_task: {type(e).__name__}: {str(e)}")
        raise HTTPException(
            status_code=500,
            detail="Could not retrieve task",
        ) from e
