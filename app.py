# backend/app.py
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
import pandas as pd
import json
from datetime import datetime

app = FastAPI(title="Kerala BioCycle API")

# CORS for frontend
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

# Load data
try:
    solution_df = pd.read_csv('submission/solution.csv')
    with open('submission/summary_metrics.json') as f:
        summary = json.load(f)
    print("✅ API: Loaded solution data")
except:
    solution_df = pd.DataFrame()
    summary = {}
    print("⚠️ API: Using placeholder data")

@app.get("/")
def read_root():
    return {"message": "Kerala BioCycle API", "status": "running"}

@app.get("/api/deliveries")
def get_deliveries(date: str = "2025-01-01"):
    """Get deliveries for a specific date"""
    if solution_df.empty:
        # Return sample data
        return {
            "date": date,
            "deliveries": [
                {"stp_id": "STP-1", "farm_id": 12, "tons": 4.2},
                {"stp_id": "STP-2", "farm_id": 45, "tons": 3.8}
            ]
        }
    
    day_data = solution_df[solution_df['date'] == date]
    deliveries = day_data.to_dict('records')
    
    return {
        "date": date,
        "deliveries": deliveries,
        "count": len(deliveries)
    }

@app.get("/api/summary")
def get_summary():
    """Get overall summary"""
    return summary

@app.get("/api/dates")
def get_available_dates():
    """Get all available simulation dates"""
    if solution_df.empty:
        return {"dates": ["2025-01-01", "2025-01-02", "2025-01-03"]}
    
    dates = sorted(solution_df['date'].unique())
    return {"dates": dates}

@app.get("/api/stats/daily")
def get_daily_stats():
    """Get daily statistics"""
    if solution_df.empty:
        return {"stats": []}
    
    daily_stats = []
    for date in sorted(solution_df['date'].unique()):
        day_data = solution_df[solution_df['date'] == date]
        daily_stats.append({
            "date": date,
            "deliveries": len(day_data),
            "tons": float(day_data['tons_delivered'].sum())
        })
    
    return {"stats": daily_stats}

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=5000)