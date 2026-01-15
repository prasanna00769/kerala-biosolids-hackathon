from geopy.distance import geodesic
import numpy as np

class STP:
    def __init__(self, stp_id, lat, lon, storage_max_tons, daily_output_tons):
        self.id = stp_id
        self.lat = lat
        self.lon = lon
        self.storage_max = storage_max_tons
        self.daily_output = daily_output_tons
        self.current_storage = 0  # Start empty
        self.yesterday_storage = 0
        
    def add_daily_output(self):
        """Add today's production to storage"""
        self.current_storage += self.daily_output
        
    def dispatch(self, amount):
        """Dispatch biosolids from storage"""
        if amount <= self.current_storage:
            self.current_storage -= amount
            return amount
        else:
            dispatched = self.current_storage
            self.current_storage = 0
            return dispatched
    
    def get_available_for_dispatch(self):
        """Max that can be dispatched today"""
        return min(self.current_storage, self.daily_output + self.yesterday_storage)
    
    def update_day(self):
        """End of day update"""
        self.yesterday_storage = self.current_storage

class Farm:
    def __init__(self, farm_id, lat, lon, zone):
        self.id = farm_id
        self.lat = lat
        self.lon = lon
        self.zone = zone
        self.daily_demand = 0
        self.applied_today = 0
        self.rain_locked = False
        self.soil_health = 100  # Start at 100%
        
    def set_demand(self, demand_kg):
        """Set today's nitrogen demand in kg"""
        self.daily_demand = demand_kg
        
    def set_rain_lock(self, is_locked):
        """Set rain lock status"""
        self.rain_locked = is_locked
        
    def apply_biosolids(self, tons, n_content_percent=0.03):
        """Apply biosolids and return actual nitrogen uptake"""
        if self.rain_locked:
            return 0  # No application during rain lock
            
        kg_n_applied = tons * 1000 * n_content_percent  # 3% N content typical
        
        # Uptake is limited by demand + 10% buffer
        max_uptake = self.daily_demand * 1.1
        
        if kg_n_applied <= max_uptake:
            actual_uptake = kg_n_applied
            excess = 0
        else:
            actual_uptake = max_uptake
            excess = kg_n_applied - max_uptake
            
        self.applied_today += tons
        return actual_uptake, excess