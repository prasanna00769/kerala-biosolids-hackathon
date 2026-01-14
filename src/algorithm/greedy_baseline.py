import pandas as pd
import numpy as np
from .core import STP, Farm
from ..utils.distance_calculator import DistanceCalculator

class GreedyBaseline:
    def __init__(self, params, stps_data, farms_data):
        self.params = params
        self.truck_capacity = 10  # tons
        self.emission_per_km = params.get('transport_emission_factor', 0.9)
        
        # Initialize STPs
        self.stps = []
        for _, row in stps_data.iterrows():
            stp = STP(
                stp_id=row['stp_id'],
                lat=row['latitude'],
                lon=row['longitude'],
                storage_max_tons=row['storage_max_tons'],
                daily_output_tons=row['daily_output_tons']
            )
            self.stps.append(stp)
            
        # Initialize Farms
        self.farms = []
        for _, row in farms_data.iterrows():
            farm = Farm(
                farm_id=row['farm_id'],
                lat=row['latitude'],
                lon=row['longitude'],
                zone=row['weather_zone']
            )
            self.farms.append(farm)
            
        # Create distance matrix
        self.dist_matrix = DistanceCalculator.create_distance_matrix(self.stps, self.farms)
        
        # Results storage
        self.deliveries = []
        self.daily_carbon_score = 0
        
    def run_day(self, date, demand_df, weather_df):
        """Run algorithm for one day"""
        # Reset farm applications
        for farm in self.farms:
            farm.applied_today = 0
            
        # Update STP storage with today's production
        for stp in self.stps:
            stp.add_daily_output()
            
        # Set farm demands and rain locks
        self._update_farm_status(date, demand_df, weather_df)
        
        # Find eligible farms (not rain locked, have demand)
        eligible_farms = [
            (i, farm) for i, farm in enumerate(self.farms)
            if not farm.rain_locked and farm.daily_demand > 0
        ]
        
        # Sort farms by demand (descending)
        eligible_farms.sort(key=lambda x: x[1].daily_demand, reverse=True)
        
        # Process each STP
        for stp_idx, stp in enumerate(self.stps):
            available = stp.get_available_for_dispatch()
            
            while available > 0 and eligible_farms:
                # Find closest eligible farm
                closest_farm_idx = None
                min_distance = float('inf')
                
                for farm_idx, farm in eligible_farms:
                    distance = self.dist_matrix[stp_idx, farm_idx]
                    if distance < min_distance:
                        min_distance = distance
                        closest_farm_idx = (farm_idx, farm)
                
                if closest_farm_idx is None:
                    break
                    
                farm_idx, farm = closest_farm_idx
                
                # Calculate how much to deliver
                # Convert demand from kg N to tons biosolids (assuming 3% N content)
                demand_tons = farm.daily_demand / (1000 * 0.03)
                deliver_amount = min(
                    self.truck_capacity,
                    demand_tons,
                    available
                )
                
                if deliver_amount > 0:
                    # Apply biosolids
                    actual_uptake, excess = farm.apply_biosolids(deliver_amount)
                    
                    # Dispatch from STP
                    stp.dispatch(deliver_amount)
                    available -= deliver_amount
                    
                    # Calculate carbon credits
                    transport_emission = -self.emission_per_km * min_distance
                    fertilizer_offset = actual_uptake * 5.0  # From parameters
                    soil_sequestration = deliver_amount * 1000 * 0.2  # 0.2 per kg
                    excess_penalty = -excess * 10.0 if excess > 0 else 0
                    
                    day_score = (transport_emission + fertilizer_offset + 
                                soil_sequestration + excess_penalty)
                    
                    # Record delivery
                    self.deliveries.append({
                        'date': date,
                        'stp_id': stp.id,
                        'farm_id': farm.id,
                        'tons_delivered': round(deliver_amount, 3),
                        'distance_km': round(min_distance, 2),
                        'carbon_impact': round(day_score, 2)
                    })
                    
                    self.daily_carbon_score += day_score
                    
                    # Remove farm if demand satisfied
                    farm.daily_demand -= actual_uptake
                    if farm.daily_demand <= 0:
                        eligible_farms = [(i, f) for i, f in eligible_farms if i != farm_idx]
        
        # Check for STP overflow penalty
        for stp in self.stps:
            if stp.current_storage > stp.storage_max:
                excess_tons = stp.current_storage - stp.storage_max
                penalty = -excess_tons * 1000  # -1000 per ton
                self.daily_carbon_score += penalty
                print(f"WARNING: STP {stp.id} overflow! Penalty: {penalty} CO2eq")
        
        # End of day update
        for stp in self.stps:
            stp.update_day()
            
        return self.daily_carbon_score
    
    def _update_farm_status(self, date, demand_df, weather_df):
        """Update farm demands and rain locks"""
        # Set demands
        day_demand = demand_df[demand_df['date'] == date]
        for _, row in day_demand.iterrows():
            farm_id = row['farm_id']
            if farm_id <= len(self.farms):
                self.farms[farm_id-1].set_demand(row['n_demand_kg'])
        
        # Set rain locks
        day_weather = weather_df[weather_df['date'] == date]
        for farm in self.farms:
            zone_weather = day_weather[day_weather['zone'] == farm.zone]
            if not zone_weather.empty:
                rain_lock = zone_weather.iloc[0]['rain_lock']
                farm.set_rain_lock(rain_lock)
                
    def get_solution_csv(self):
        """Generate solution.csv format"""
        df = pd.DataFrame(self.deliveries)
        # Keep only required columns
        solution_df = df[['date', 'stp_id', 'farm_id', 'tons_delivered']].copy()
        return solution_df