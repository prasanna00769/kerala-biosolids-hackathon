import pandas as pd
import numpy as np
from datetime import datetime, timedelta

class DataSimulator:
    def __init__(self, num_farms=250, start_date="2025-01-01"):
        self.num_farms = num_farms
        self.start_date = datetime.strptime(start_date, "%Y-%m-%d")
        
    def generate_daily_demand(self, num_days=365):
        """Generate realistic daily nitrogen demand"""
        dates = [self.start_date + timedelta(days=i) for i in range(num_days)]
        data = []
        
        for date in dates:
            for farm_id in range(1, self.num_farms + 1):
                # Base demand with seasonality
                day_of_year = date.timetuple().tm_yday
                
                # Planting seasons in Kerala (simplified)
                if 60 <= day_of_year <= 150:  # Feb-May: First crop season
                    base_demand = np.random.uniform(50, 200)
                elif 180 <= day_of_year <= 270:  # Jun-Sep: Monsoon season (lower demand)
                    base_demand = np.random.uniform(10, 80)
                else:  # Oct-Jan: Second crop season
                    base_demand = np.random.uniform(30, 150)
                
                # Add randomness
                demand = max(0, base_demand + np.random.normal(0, 20))
                
                data.append({
                    'date': date.strftime("%Y-%m-%d"),
                    'farm_id': farm_id,
                    'n_demand_kg': round(demand, 2)
                })
        
        df = pd.DataFrame(data)
        df.to_csv("data/daily_n_demand.csv", index=False)
        print(f"Generated {len(df)} demand records")
        return df
    
    def generate_weather_forecast(self):
        """Generate 5-day weather forecast for each farm zone"""
        # Simplified: Assume 5 weather zones
        zones = ['North', 'Central', 'South', 'Highlands', 'Coastal']
        dates = [self.start_date + timedelta(days=i) for i in range(365)]
        
        weather_data = []
        for date in dates:
            month = date.month
            for zone in zones:
                # Monsoon months (Jun-Sep) have more rain
                if 6 <= month <= 9:
                    rainfall = np.random.exponential(scale=15)  # More rain
                else:
                    rainfall = np.random.exponential(scale=5)   # Less rain
                    
                # Occasional heavy rain events
                if np.random.random() < 0.1:  # 10% chance of heavy rain
                    rainfall = min(100, rainfall * 3)
                
                weather_data.append({
                    'date': date.strftime("%Y-%m-%d"),
                    'zone': zone,
                    'rainfall_mm': round(rainfall, 1),
                    'rain_lock': rainfall > 30  # 30mm threshold
                })
        
        df = pd.DataFrame(weather_data)
        df.to_csv("data/weather_forecast.csv", index=False)
        return df