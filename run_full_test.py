import sys
import os
import pandas as pd
import numpy as np
import json

print("="*60)
print("QUICK 7-DAY SIMULATION (NO IMPORT ERRORS)")
print("="*60)

# Add src to path
current_dir = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, current_dir)
sys.path.insert(0, os.path.join(current_dir, 'src'))

# Define classes inline to avoid import issues
class STP:
    def __init__(self, stp_id, lat, lon, storage_max_tons, daily_output_tons):
        self.id = stp_id
        self.lat = lat
        self.lon = lon
        self.storage_max = storage_max_tons
        self.daily_output = daily_output_tons
        self.current_storage = 0
        self.yesterday_storage = 0
        
    def add_daily_output(self):
        self.current_storage += self.daily_output
        
    def dispatch(self, amount):
        if amount <= self.current_storage:
            self.current_storage -= amount
            return amount
        dispatched = self.current_storage
        self.current_storage = 0
        return dispatched
    
    def get_available_for_dispatch(self):
        return min(self.current_storage, self.daily_output + self.yesterday_storage)
    
    def update_day(self):
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
        
    def set_demand(self, demand_kg):
        self.daily_demand = demand_kg
        
    def set_rain_lock(self, is_locked):
        self.rain_locked = is_locked
        
    def apply_biosolids(self, tons, n_content_percent=0.03):
        if self.rain_locked:
            return 0, 0
        kg_n_applied = tons * 1000 * n_content_percent
        max_uptake = self.daily_demand * 1.1
        if kg_n_applied <= max_uptake:
            actual_uptake = kg_n_applied
            excess = 0
        else:
            actual_uptake = max_uptake
            excess = kg_n_applied - max_uptake
        self.applied_today += tons
        return actual_uptake, excess

class SimpleGreedyAlgorithm:
    def __init__(self, params, stps_data, farms_data):
        self.params = params
        self.truck_capacity = 10
        self.emission_per_km = params.get('transport_emission_factor', 0.9)
        
        # Create STPs
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
            
        # Create Farms
        self.farms = []
        for _, row in farms_data.iterrows():
            farm = Farm(
                farm_id=row['farm_id'],
                lat=row['latitude'],
                lon=row['longitude'],
                zone=row['weather_zone']
            )
            self.farms.append(farm)
        
        # Pre-calculate distances
        self.dist_matrix = self.create_distance_matrix()
        
        self.deliveries = []
        self.total_carbon = 0
        
    def haversine(self, lat1, lon1, lat2, lon2):
        lat1, lon1, lat2, lon2 = map(np.radians, [lat1, lon1, lat2, lon2])
        dlat = lat2 - lat1
        dlon = lon2 - lon1
        a = np.sin(dlat/2)**2 + np.cos(lat1) * np.cos(lat2) * np.sin(dlon/2)**2
        c = 2 * np.arcsin(np.sqrt(a))
        return c * 6371
    
    def create_distance_matrix(self):
        n_stps = len(self.stps)
        n_farms = len(self.farms)
        dist_matrix = np.zeros((n_stps, n_farms))
        for i, stp in enumerate(self.stps):
            for j, farm in enumerate(self.farms):
                dist_matrix[i, j] = self.haversine(
                    stp.lat, stp.lon, farm.lat, farm.lon
                )
        return dist_matrix
    
    def update_farm_status(self, date, demand_df, weather_df):
        today_demand = demand_df[demand_df['date'] == date]
        today_weather = weather_df[weather_df['date'] == date]
        
        for farm in self.farms:
            farm_demand = today_demand[today_demand['farm_id'] == farm.id]
            if not farm_demand.empty:
                farm.set_demand(farm_demand['n_demand_kg'].values[0])
            else:
                farm.set_demand(0)
            
            zone_weather = today_weather[today_weather['zone'] == farm.zone]
            if not zone_weather.empty:
                farm.set_rain_lock(zone_weather['rain_lock'].values[0])
            else:
                farm.set_rain_lock(False)
    
    def run_day(self, date, demand_df, weather_df):
        print(f"📅 {date}: ", end="")
        
        # Reset farms
        for farm in self.farms:
            farm.applied_today = 0
        
        # Update STPs
        for stp in self.stps:
            stp.add_daily_output()
        
        # Update farm status
        self.update_farm_status(date, demand_df, weather_df)
        
        daily_deliveries = 0
        daily_tons = 0
        daily_carbon = 0
        
        # Process each STP
        for stp_idx, stp in enumerate(self.stps):
            available = stp.get_available_for_dispatch()
            
            while available > 0.1:
                # Find closest eligible farm
                best_farm_idx = None
                min_distance = float('inf')
                
                for farm_idx, farm in enumerate(self.farms):
                    if farm.rain_locked or farm.daily_demand <= 0:
                        continue
                    
                    distance = self.dist_matrix[stp_idx, farm_idx]
                    if distance < min_distance:
                        min_distance = distance
                        best_farm_idx = farm_idx
                
                if best_farm_idx is None:
                    break
                
                farm = self.farms[best_farm_idx]
                distance = min_distance
                
                # Calculate delivery amount
                demand_tons = farm.daily_demand / (1000 * 0.03)
                deliver_amount = min(self.truck_capacity, demand_tons, available)
                
                if deliver_amount < 0.1:
                    break
                
                # Apply biosolids
                n_uptake_kg, excess_n_kg = farm.apply_biosolids(deliver_amount)
                
                # Dispatch
                stp.dispatch(deliver_amount)
                available -= deliver_amount
                
                # Calculate carbon
                transport_cost = -self.emission_per_km * distance
                fertilizer_offset = n_uptake_kg * 5.0
                soil_sequestration = deliver_amount * 1000 * 0.2
                excess_penalty = -excess_n_kg * 10.0 if excess_n_kg > 0 else 0
                
                carbon_total = transport_cost + fertilizer_offset + soil_sequestration + excess_penalty
                
                # Record delivery
                self.deliveries.append({
                    'date': date,
                    'stp_id': stp.id,
                    'farm_id': farm.id,
                    'tons_delivered': round(deliver_amount, 3)
                })
                
                daily_deliveries += 1
                daily_tons += deliver_amount
                daily_carbon += carbon_total
                
                # Update farm demand
                farm.daily_demand = max(0, farm.daily_demand - n_uptake_kg)
        
        # End of day
        for stp in self.stps:
            stp.update_day()
        
        self.total_carbon += daily_carbon
        print(f"{daily_deliveries} deliveries, {daily_tons:.1f} tons, Carbon: {daily_carbon:+.1f}")
        return daily_carbon
    
    def get_solution_csv(self, demand_df):
        # Create base frame from demand_df to ensure all 91250 rows are present
        submission_base = demand_df[['date', 'farm_id']].copy()
        
        # Prepare deliveries
        deliveries_df = pd.DataFrame(self.deliveries)
        
        if not deliveries_df.empty:
            # Aggregate if multiple deliveries per farm per day (though greedy shouldn't do this)
            deliveries_agg = deliveries_df.groupby(['date', 'farm_id']).agg({
                'tons_delivered': 'sum',
                'stp_id': 'first'  # Take the first STP if multiple
            }).reset_index()
            
            # Merge
            result = pd.merge(submission_base, deliveries_agg, on=['date', 'farm_id'], how='left')
        else:
            result = submission_base.copy()
            result['tons_delivered'] = 0
            result['stp_id'] = 'STP-1'

        # Fill NaNs
        result['tons_delivered'] = result['tons_delivered'].fillna(0)
        result['stp_id'] = result['stp_id'].fillna('STP-1')  # Default to STP-1 for non-delivered rows
        
        # Add ID column
        result['id'] = result.index
        
        # Reorder columns
        return result[['id', 'date', 'stp_id', 'farm_id', 'tons_delivered']]
    
    def get_summary_metrics(self):
        df = pd.DataFrame(self.deliveries)
        return {
            'total_carbon_credits_kg': float(round(self.total_carbon, 2)),
            'total_deliveries': len(self.deliveries),
            'total_tons_delivered': float(round(df['tons_delivered'].sum(), 2)) if not df.empty else 0,
            'average_tons_per_delivery': float(round(df['tons_delivered'].mean(), 3)) if not df.empty else 0,
            'simulation_days': len(set(df['date'])) if not df.empty else 0,
            'average_daily_carbon': float(round(self.total_carbon / len(set(df['date'])), 2)) if len(set(df['date'])) > 0 else 0
        }

# MAIN EXECUTION
print("\n📊 Creating test data...")

# Create data directory
data_dir = os.path.join(current_dir, 'data')
os.makedirs(data_dir, exist_ok=True)

# Check for existing data
if os.path.exists(os.path.join(data_dir, 'stp_registry.csv')):
    print("Loading existing data...")
    stps_df = pd.read_csv(os.path.join(data_dir, 'stp_registry.csv'))
    farms_df = pd.read_csv(os.path.join(data_dir, 'farm_locations.csv'))
    demand_df = pd.read_csv(os.path.join(data_dir, 'daily_n_demand.csv'))
    weather_df = pd.read_csv(os.path.join(data_dir, 'weather_forecast.csv'))
    
    # Load or create parameters
    params_path = os.path.join(data_dir, 'parameters.json')
    if os.path.exists(params_path):
        with open(params_path, 'r') as f:
            params = json.load(f)
    else:
        params = {
            'transport_emission_factor': 0.9,
            'fertilizer_offset_factor': 5.0,
            'soil_sequestration_factor': 0.2,
            'excess_n_penalty': 10.0,
            'overflow_penalty_per_ton': 1000
        }
else:
    print("Creating test data...")
    
    # Create STP data
    stp_data = {
        'stp_id': ['STP-1', 'STP-2', 'STP-3', 'STP-4'],
        'latitude': [9.9312, 8.5241, 11.2588, 10.5276],
        'longitude': [76.2673, 76.9366, 75.7804, 76.2144],
        'storage_max_tons': [100, 120, 80, 90],
        'daily_output_tons': [15, 18, 12, 14]
    }
    stps_df = pd.DataFrame(stp_data)
    
    # Create farm data (50 farms for speed)
    farm_ids = list(range(1, 51))
    farm_data = {
        'farm_id': farm_ids,
        'latitude': [8.5 + np.random.random() * 4.5 for _ in farm_ids],
        'longitude': [74.5 + np.random.random() * 3.0 for _ in farm_ids],
        'weather_zone': np.random.choice(['North', 'Central', 'South'], size=len(farm_ids))
    }
    farms_df = pd.DataFrame(farm_data)
    
    # Create demand data for 7 days
    dates = pd.date_range('2025-01-01', periods=7).strftime('%Y-%m-%d')
    demand_records = []
    for date in dates:
        for farm_id in farm_ids:
            demand = max(10, np.random.normal(100, 30))
            demand_records.append({
                'date': date,
                'farm_id': farm_id,
                'n_demand_kg': round(demand, 2)
            })
    demand_df = pd.DataFrame(demand_records)
    
    # Create weather data
    zones = ['North', 'Central', 'South']
    weather_records = []
    for date in dates:
        for zone in zones:
            rainfall = np.random.exponential(5)
            weather_records.append({
                'date': date,
                'zone': zone,
                'rainfall_mm': round(rainfall, 1),
                'rain_lock': rainfall > 30
            })
    weather_df = pd.DataFrame(weather_records)
    
    params = {
        'transport_emission_factor': 0.9,
        'fertilizer_offset_factor': 5.0,
        'soil_sequestration_factor': 0.2,
        'excess_n_penalty': 10.0,
        'overflow_penalty_per_ton': 1000
    }

print(f"\n📦 Data loaded:")
print(f"   STPs: {len(stps_df)}")
print(f"   Farms: {len(farms_df)}")
print(f"   Demand records: {len(demand_df)}")

# Create and run algorithm
print("\n🤖 Creating algorithm...")
algorithm = SimpleGreedyAlgorithm(params, stps_df, farms_df)

# Run 7-day simulation
# Run full simulation
print("\n▶️ Running full simulation...")
dates = sorted(demand_df['date'].unique())

for date in dates:
    algorithm.run_day(date, demand_df, weather_df)

print(f"\n✅ Simulation complete!")
print(f"Total deliveries: {len(algorithm.deliveries)}")
print(f"Total carbon: {algorithm.total_carbon:+.1f} kg CO₂eq")

# Generate outputs
print("\n📄 Generating solution.csv...")
solution_df = algorithm.get_solution_csv(demand_df)

# Save to submission folder
submission_dir = os.path.join(current_dir, 'submission')
os.makedirs(submission_dir, exist_ok=True)

solution_path = os.path.join(submission_dir, 'solution.csv')
solution_df.to_csv(solution_path, index=False)
print(f"💾 Saved to: {solution_path}")
print(f"   Deliveries: {len(solution_df)}")
print(f"   Total tons: {solution_df['tons_delivered'].sum():.1f}")

# Generate summary
print("\n📊 Generating summary_metrics.json...")
summary = algorithm.get_summary_metrics()

summary_path = os.path.join(submission_dir, 'summary_metrics.json')
with open(summary_path, 'w') as f:
    json.dump(summary, f, indent=2)

print(f"💾 Saved to: {summary_path}")
print("\nSummary:")
for key, value in summary.items():
    print(f"  {key}: {value}")

print("\n" + "="*60)
print("🎉 STEP 9 COMPLETE - READY FOR SUBMISSION!")
print("="*60)
print("\nFiles in 'submission/' folder:")
print("  1. solution.csv - Delivery schedule")
print("  2. summary_metrics.json - Performance metrics")
print(f"\nTotal carbon credits: {summary['total_carbon_credits_kg']} kg CO₂eq")
print(f"Created at: {current_dir}\\submission\\")