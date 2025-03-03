from typing import Any, List

from fastapi import APIRouter, HTTPException
from fastapi.responses import FileResponse, JSONResponse
from fastapi import Depends

from app.api.deps import CurrentUser, SessionDep
from app.api.models.mri_service import MRIProcessParameters

from app.mri_service.awesomedemo import generate_synthetic_mri_images

router = APIRouter(prefix="/mri_service", tags=["mri_service"])

@router.post("/generate_mri_images", response_model=List[str])
def generate_mri_images(
    params: MRIProcessParameters,
    session: SessionDep, current_user: CurrentUser
) -> Any:
    """
    Generate MRI Images.
    """
    print("Called generate_mri_images route.")

    result = generate_synthetic_mri_images(params)
    if not result or not isinstance(result, dict):
        raise HTTPException(status_code=500, detail="Failed to generate MRI images.")
    
    output_dir = result.get("output_dir")
    comparison_path = result.get("comparison_image")
    result_images = result.get("result_images")
    
    if not output_dir or not comparison_path or not result_images:
        raise HTTPException(status_code=500, detail="Incomplete MRI image generation result.")

    print(f"Images saved in {output_dir} and comparison saved in {comparison_path}")

    # Return the list of image paths
    # return JSONResponse(content={
    #     "output_dir": output_dir,
    #     "comparison_image": comparison_path,
    #     "result_images": result_images
    # })
    return FileResponse(comparison_path)
