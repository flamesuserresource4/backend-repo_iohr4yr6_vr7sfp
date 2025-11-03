"""
Database Schemas

Each Pydantic model corresponds to a MongoDB collection with the
collection name equal to the lowercase class name.
"""
from pydantic import BaseModel, Field
from typing import Optional, List, Dict, Any

class Stat(BaseModel):
    users: int = Field(0, description="Total registered or active users")
    co2_saved_tons: float = Field(0, description="Cumulative CO2 saved in tons")
    calculations: int = Field(0, description="Total number of ROI calculations performed")

class Calculation(BaseModel):
    city: str
    monthly_bill_pkr: float
    sun_hours: float
    estimated_savings_pkr: float
    system_kw: float
    efficiency: float
    payload: Optional[Dict[str, Any]] = None

class WeatherCache(BaseModel):
    city: str
    kind: str  # "current" or "forecast"
    data: Dict[str, Any]
    etag: str
    ttl_seconds: int = 900  # 15 minutes by default

# Example schemas left for reference; not used by app logic
class User(BaseModel):
    name: str
    email: str
    address: str
    age: Optional[int] = None
    is_active: bool = True

class Product(BaseModel):
    title: str
    description: Optional[str] = None
    price: float
    category: str
    in_stock: bool = True
