"""User router"""

from typing import List

from fastapi import APIRouter, Depends, HTTPException
from sqlalchemy.orm import Session

import database.db as userCRUD
import database.schemas as schemas
from database.connection import get_db

router = APIRouter(prefix="/users", tags=["users"])


@router.post("/", response_model=schemas.User)
async def create_user(user: schemas.UserCreate, db: Session = Depends(get_db)):
    """Create a new user"""

    db_user = userCRUD.get_user_by_email(db, email=user.email)
    if db_user:
        raise HTTPException(status_code=400, detail="Email already registered")
    return userCRUD.create_user(db=db, user=user)


@router.get("/", response_model=List[schemas.User])
async def get_all_users(skip: int = 0, limit: int = 100, db: Session = Depends(get_db)):
    """Get all users"""

    return userCRUD.get_users(db, skip=skip, limit=limit)


@router.get("/{user_id}", response_model=schemas.User)
async def read_user(user_id: int, db: Session = Depends(get_db)):
    """Get a user by ID"""

    db_user = userCRUD.get_user(db, user_id=user_id)
    if db_user is None:
        raise HTTPException(status_code=404, detail="User not found")
    return db_user


@router.put("/{user_id}", response_model=schemas.User)
async def update_user(
    user_id: int, user: schemas.UserUpdate, db: Session = Depends(get_db)
):
    """Update a user by ID"""

    db_user = userCRUD.update_user(db, user_id=user_id, user=user)
    if db_user is None:
        raise HTTPException(status_code=404, detail="User not found")
    return db_user


@router.delete("/{user_id}")
async def delete_user(user_id: int, db: Session = Depends(get_db)):
    """Delete a user by ID"""

    deleted = userCRUD.delete_user(db, user_id=user_id)
    if not deleted:
        raise HTTPException(status_code=404, detail="User not found")
    return {"message": "User deleted successfully"}
