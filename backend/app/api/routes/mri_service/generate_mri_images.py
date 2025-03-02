from typing import Any

from fastapi import APIRouter, HTTPException

from app.api.deps import CurrentUser, SessionDep

from app.mri_service.awesomedemo import generate_mri_images

router = APIRouter(prefix="/mri_service", tags=["mri_service"])

@router.post("/generate_mri_images")
def generate_mri_images(
    session: SessionDep, current_user: CurrentUser # TODO: Pydantic Model for Model Configuration
) -> Any:
    """
    Generate MRI Images.
    TODO: use specific configuration provided by user in request to route
    """
    print("Called generate_mri_images route.")
    print(f"Called generate_mri_images: {generate_mri_images()}")

    return True
