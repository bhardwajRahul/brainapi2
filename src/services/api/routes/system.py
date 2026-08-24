"""
File: /system.py
Created Date: Monday December 1st 2025
Author: Christian Nonis <alch.infoemail@gmail.com>
-----
Last Modified: Monday December 1st 2025 10:12:48 pm
Modified By: Christian Nonis <alch.infoemail@gmail.com>
-----
"""

from fastapi import APIRouter, Request
from src.services.api.constants.requests import CreateBrainRequest
from src.services.api.controllers.system import (
    get_brains_list as get_brains_list_controller,
    create_new_brain as create_new_brain_controller,
)
from src.constants.data import Brain
from src.services.api.errors import error_response

system_router = APIRouter(prefix="/system", tags=["system"])


@system_router.get(path="/brains-list", response_model=list[Brain])
async def get_brains_list():
    """
    Get the list of brains.
    """
    return await get_brains_list_controller()


@system_router.post(path="/brains", response_model=Brain)
async def create_brain(request: CreateBrainRequest):
    """
    Create a new brain
    """
    return await create_new_brain_controller(request)


@system_router.get(path="/brains/{brain_id}/reset", include_in_schema=False)  # TODO
async def reset(brain_id: str, request: Request):
    """
    Resets the brain
    """
    return _not_implemented(request, brain_id, "reset")


@system_router.get(path="/brains/{brain_id}/delete", include_in_schema=False)  # TODO
async def delete(brain_id: str, request: Request):
    """
    Deletes the brain and all its data
    """
    return _not_implemented(request, brain_id, "delete")


@system_router.post(
    path="/brains/{brain_id}/create-backup", include_in_schema=False
)  # TODO
async def create_backup(brain_id: str, request: Request):
    """
    Creates a backup of the brain and returns a task ID
    """
    return _not_implemented(request, brain_id, "create-backup")


def _not_implemented(request: Request, brain_id: str, operation: str):
    return error_response(
        request,
        status_code=501,
        detail="Not implemented",
        code="not_implemented",
        message=f"Brain {operation} is not implemented.",
        resolution="Use the profile-aware deploy backup tooling where applicable.",
        extra={"brain_id": brain_id, "operation": operation},
    )
