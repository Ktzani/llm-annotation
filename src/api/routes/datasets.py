from fastapi import APIRouter, HTTPException
from src.utils.data_loader import list_available_datasets

router = APIRouter(prefix="/datasets", tags=["Datasets"])


@router.get("/")
async def list_datasets():
    try:
        return {"datasets": list_available_datasets()}
    except Exception as e:
        raise HTTPException(500, str(e))
