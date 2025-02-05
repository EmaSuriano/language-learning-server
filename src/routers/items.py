"""Items router"""

from fastapi import APIRouter, HTTPException
from pydantic import BaseModel


router = APIRouter(prefix="/items", tags=["items"])


class Item(BaseModel):
    """Item model"""

    name: str
    description: str


items: list[Item] = []


@router.get("/items")
async def get_all_items():
    """Get all items"""
    return items


@router.post("/items/", response_model=Item)
async def create_item(item: Item):
    """Create a new item"""
    items.append(item)
    return item


@router.get("/items/{item_id}", response_model=Item)
async def read_item(item_id: int):
    """Get a single item"""
    if item_id < 0 or item_id >= len(items):
        raise HTTPException(status_code=404, detail="Item not found")
    return items[item_id]


@router.put("/items/{item_id}", response_model=Item)
async def update_item(item_id: int, item: Item):
    """Update an item"""
    if item_id < 0 or item_id >= len(items):
        raise HTTPException(status_code=404, detail="Item not found")

    items[item_id] = item
    return item


@router.delete("/items/{item_id}", response_model=Item)
async def delete_item(item_id: int):
    """Delete an item"""
    if item_id < 0 or item_id >= len(items):
        raise HTTPException(status_code=404, detail="Item not found")

    deleted_item = items.pop(item_id)
    return deleted_item
