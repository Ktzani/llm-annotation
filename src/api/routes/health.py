
from fastapi import APIRouter
from datetime import datetime

router = APIRouter(tags=["health"])

@router.get("/health")
async def health_check():
    return {"status": "healthy", "timestamp": datetime.now()}
