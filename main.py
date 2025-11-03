import os
import time
import hashlib
from typing import Optional, Dict, Any

import requests
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse

from database import db, create_document, get_documents
from schemas import Calculation, Stat, WeatherCache

app = FastAPI(title="EcoPulse Pakistan API")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

OPENWEATHER_API_KEY = os.getenv("OPENWEATHER_API_KEY", "")
GOOGLE_API_KEY = os.getenv("GOOGLE_API_KEY", "")

# Simple in-process cache with TTL
_memory_cache: Dict[str, Dict[str, Any]] = {}

def _cache_key(prefix: str, params: Dict[str, Any]) -> str:
    raw = prefix + "|" + "&".join(f"{k}={v}" for k, v in sorted(params.items()))
    return hashlib.sha256(raw.encode()).hexdigest()


def _json_with_cache(payload: Dict[str, Any], ttl: int = 900) -> JSONResponse:
    data_bytes = repr(payload).encode()
    etag = hashlib.md5(data_bytes).hexdigest()
    headers = {
        "Cache-Control": f"public, max-age={ttl}",
        "ETag": etag,
    }
    return JSONResponse(content=payload, headers=headers)


CITY_COORDS = {
    "lahore": {"lat": 31.5204, "lon": 74.3587, "sun_hours": 5.2},
    "karachi": {"lat": 24.8607, "lon": 67.0011, "sun_hours": 5.0},
    "islamabad": {"lat": 33.6844, "lon": 73.0479, "sun_hours": 5.4},
}


@app.get("/")
def root():
    return {"service": "EcoPulse Pakistan API", "status": "ok"}


@app.get("/test")
def test_database():
    response = {
        "backend": "✅ Running",
        "database": "❌ Not Available",
        "database_url": "✅ Set" if os.getenv("DATABASE_URL") else "❌ Not Set",
        "database_name": "✅ Set" if os.getenv("DATABASE_NAME") else "❌ Not Set",
        "connection_status": "Not Connected",
        "collections": [],
    }
    try:
        if db is not None:
            response["database"] = "✅ Connected"
            response["connection_status"] = "Connected"
            response["collections"] = db.list_collection_names()
    except Exception as e:
        response["database"] = f"⚠️ Error: {str(e)[:100]}"
    return response


@app.get("/api/stats")
def get_stats():
    # Use a single document in "stat" collection
    if db is None:
        raise HTTPException(500, "Database not configured")
    doc = db["stat"].find_one({})
    if not doc:
        doc = {"users": 2847, "co2_saved_tons": 15480.0, "calculations": 0}
        db["stat"].insert_one(doc)
    return _json_with_cache({
        "users": doc.get("users", 0),
        "co2_saved_tons": doc.get("co2_saved_tons", 0),
        "calculations": doc.get("calculations", 0),
    }, ttl=120)


@app.post("/api/calc/quick")
def quick_calc(payload: Dict[str, Any]):
    city = str(payload.get("city", "")).lower()
    monthly_bill_pkr = float(payload.get("monthly_bill_pkr", 0))
    if city not in CITY_COORDS:
        raise HTTPException(400, "Unsupported city")
    sun_hours = CITY_COORDS[city]["sun_hours"]
    # Very simple sizing model
    avg_tariff = 55.0  # PKR/kWh
    monthly_kwh = monthly_bill_pkr / avg_tariff
    system_kw = round(monthly_kwh / (30 * sun_hours), 2)
    efficiency = 0.68
    estimated_savings = round(monthly_kwh * avg_tariff * efficiency / 2, 0)  # conservative

    calc = Calculation(
        city=city,
        monthly_bill_pkr=monthly_bill_pkr,
        sun_hours=sun_hours,
        estimated_savings_pkr=estimated_savings,
        system_kw=system_kw,
        efficiency=efficiency,
        payload=payload,
    )
    try:
        create_document("calculation", calc)
        db["stat"].update_one({}, {"$inc": {"calculations": 1}}, upsert=True)
    except Exception:
        pass

    return _json_with_cache(calc.model_dump(), ttl=120)


@app.get("/api/weather")
def get_weather(city: str):
    if not OPENWEATHER_API_KEY:
        raise HTTPException(500, "OPENWEATHER_API_KEY not set")
    key = _cache_key("weather", {"city": city})
    now = time.time()
    if key in _memory_cache and _memory_cache[key]["exp"] > now:
        return _json_with_cache(_memory_cache[key]["data"], ttl=int(_memory_cache[key]["exp"]-now))

    c = CITY_COORDS.get(city.lower())
    if not c:
        raise HTTPException(400, "Unsupported city")
    url = "https://api.openweathermap.org/data/2.5/weather"
    params = {"lat": c["lat"], "lon": c["lon"], "units": "metric", "appid": OPENWEATHER_API_KEY}
    r = requests.get(url, params=params, timeout=10)
    if r.status_code != 200:
        raise HTTPException(r.status_code, r.text)
    data = r.json()
    payload = {
        "temp_c": data.get("main", {}).get("temp"),
        "humidity": data.get("main", {}).get("humidity"),
        "clouds": data.get("clouds", {}).get("all"),
        "wind_ms": data.get("wind", {}).get("speed"),
        "description": (data.get("weather") or [{}])[0].get("description"),
    }
    _memory_cache[key] = {"data": payload, "exp": now + 900}
    return _json_with_cache(payload, ttl=900)


@app.get("/api/forecast")
def get_forecast(city: str):
    if not OPENWEATHER_API_KEY:
        raise HTTPException(500, "OPENWEATHER_API_KEY not set")
    c = CITY_COORDS.get(city.lower())
    if not c:
        raise HTTPException(400, "Unsupported city")
    key = _cache_key("forecast", {"city": city})
    now = time.time()
    if key in _memory_cache and _memory_cache[key]["exp"] > now:
        return _json_with_cache(_memory_cache[key]["data"], ttl=int(_memory_cache[key]["exp"]-now))

    url = "https://api.openweathermap.org/data/2.5/forecast"
    params = {"lat": c["lat"], "lon": c["lon"], "units": "metric", "appid": OPENWEATHER_API_KEY}
    r = requests.get(url, params=params, timeout=10)
    if r.status_code != 200:
        raise HTTPException(r.status_code, r.text)
    data = r.json()
    # Reduce to daily best-hour window approximation
    buckets: Dict[str, Dict[str, Any]] = {}
    for item in data.get("list", []):
        dt_txt = item.get("dt_txt")  # "YYYY-MM-DD HH:MM:SS"
        if not dt_txt:
            continue
        day = dt_txt.split(" ")[0]
        clouds = item.get("clouds", {}).get("all", 100)
        temp = item.get("main", {}).get("temp")
        hour = int(dt_txt.split(" ")[1].split(":")[0])
        score = max(0, 100 - clouds) + (10 if 11 <= hour <= 14 else 0)
        b = buckets.setdefault(day, {"best": None})
        cur = {"hour": hour, "clouds": clouds, "temp_c": temp, "score": score}
        if b["best"] is None or score > b["best"]["score"]:
            b["best"] = cur
    simplified = [{"day": d, **v["best"]} for d, v in list(buckets.items())[:7] if v["best"]]
    payload = {"city": city.lower(), "best_hours": simplified}
    _memory_cache[key] = {"data": payload, "exp": now + 900}
    return _json_with_cache(payload, ttl=900)


@app.post("/api/predict")
def ai_predict(payload: Dict[str, Any]):
    city = str(payload.get("city", "")).lower()
    if city not in CITY_COORDS:
        raise HTTPException(400, "Unsupported city")
    advice = {
        "summary": "Run heavy loads between 11–2 PM, confidence high.",
        "confidence": "high",
        "city": city,
    }
    # If Gemini key is present, call model for richer summary
    if GOOGLE_API_KEY:
        try:
            import google.generativeai as genai
            genai.configure(api_key=GOOGLE_API_KEY)
            prompt = (
                f"Given Pakistani city {city}, sun hours {CITY_COORDS[city]['sun_hours']}, "
                "and typical load shedding cycles, suggest a 1-2 line actionable plan for solar usage today."
            )
            model = genai.GenerativeModel("gemini-1.5-flash")
            resp = model.generate_content(prompt)
            text = resp.text.strip() if hasattr(resp, "text") else None
            if text:
                advice["summary"] = text
        except Exception:
            pass
    return _json_with_cache(advice, ttl=300)
