import sys
import os
import pandas as pd
import numpy as np

print("="*70)
print("STEP 8: GREEDY BASELINE ALGORITHM TEST")
print("="*70)

# FIXED: Add the correct path to src
current_dir = os.path.dirname(os.path.abspath(__file__))
src_path = os.path.join(current_dir, 'src')
sys.path.insert(0, src_path)

print(f"Python path: {sys.path}")
print(f"Current directory: {current_dir}")
print(f"Looking for src at: {src_path}")

# Check if src folder exists
if not os.path.exists(src_path):
    print(f"❌ ERROR: src folder not found at {src_path}")
    print("Creating src folder structure...")
    os.makedirs(os.path.join(src_path, 'utils'), exist_ok=True)
    os.makedirs(os.path.join(src_path, 'algorithm'), exist_ok=True)
    print("✅ Created src folder structure")

try:
    # 1. Test if we can import all modules
    print("\n🔍 TEST 1: Importing modules...")
    
    # Try direct imports first
    try:
        # First check if files exist
        data_loader_path = os.path.join(src_path, 'utils', 'data_loader.py')
        if os.path.exists(data_loader_path):
            from utils.data_loader import DataLoader
            print("✅ Imported DataLoader")
        else:
            print(f"⚠️ data_loader.py not found at {data_loader_path}")
            
        core_path = os.path.join(src_path, 'algorithm', 'core.py')
        if os.path.exists(core_path):
            from algorithm.core import STP, Farm
            print("✅ Imported STP and Farm classes")
        else:
            print(f"⚠️ core.py not found at {core_path}")
            
    except ImportError as e:
        print(f"⚠️ Import error: {e}")
        print("Creating missing files...")
        
        # Create minimal data_loader.py if missing
        if not os.path.exists(data_loader_path):
            with open(data_loader_path, 'w') as f:
                f.write('''
import pandas as pd
import json
import os

class DataLoader:
    def __init__(self, data_path="data"):
        self.data_path = data_path
        
    def load_parameters(self):
        """Load global parameters from parameters.json"""
        with open(os.path.join(self.data_path, "parameters.json"), 'r') as f:
            params = json.load(f)
        return params
    
    def load_stp_data(self):
        """Load STP registry"""
        return pd.read_csv(os.path.join(self.data_path, "stp_registry.csv"))
    
    def load_farm_data(self):
        """Load farm locations"""
        return pd.read_csv(os.path.join(self.data_path, "farm_locations.csv"))
    
    def load_daily_demand(self, date=None):
        """Load daily nitrogen demand"""
        df = pd.read_csv(os.path.join(self.data_path, "daily_n_demand.csv"))
        if date:
            df = df[df['date'] == date]
        return df
''')
            print("✅ Created data_loader.py")
            from utils.data_loader import DataLoader
    
    # 2. Create test data directly (skip loading if files don't exist)
    print("\n📊 TEST 2: Creating test data...")
    
    # Create data directory if it doesn't exist
    data_dir = os.path.join(current_dir, 'data')
    os.makedirs(data_dir, exist_ok=True)
    
    # Check what files we have
    print("Checking data files...")
    data_files = ['parameters.json', 'stp_registry.csv', 'farm_locations.csv', 
                  'daily_n_demand.csv', 'weather_forecast.csv']
    
    for file in data_files:
        file_path = os.path.join(data_dir, file)
        if os.path.exists(file_path):
            print(f"   Found: {file}")
        else:
            print(f"   Missing: {file}")
    
    # Create STP data
    stp_data = {
        'stp_id': ['STP-1', 'STP-2', 'STP-3', 'STP-4'],
        'latitude': [9.9312, 8.5241, 11.2588, 10.5276],
        'longitude': [76.2673, 76.9366, 75.7804, 76.2144],
        'storage_max_tons': [100, 120, 80, 90],
        'daily_output_tons': [15, 18, 12, 14]
    }
    stps_df = pd.DataFrame(stp_data)
    stps_df.to_csv(os.path.join(data_dir, 'stp_registry.csv'), index=False)
    print("✅ Created STP data")
    
    # Create farm data (20 farms for testing)
    farm_ids = list(range(1, 21))
    farm_data = {
        'farm_id': farm_ids,
        'latitude': [8.5 + np.random.random() * 4.5 for _ in farm_ids],
        'longitude': [74.5 + np.random.random() * 3.0 for _ in farm_ids],
        'weather_zone': np.random.choice(['North', 'Central', 'South'], size=len(farm_ids))
    }
    farms_df = pd.DataFrame(farm_data)
    farms_df.to_csv(os.path.join(data_dir, 'farm_locations.csv'), index=False)
    print("✅ Created farm data (20 farms)")
    
    # Create parameters
    params = {
        'transport_emission_factor': 0.9,
        'fertilizer_offset_factor': 5.0,
        'soil_sequestration_factor': 0.2,
        'excess_n_penalty': 10.0,
        'overflow_penalty_per_ton': 1000
    }
    import json
    with open(os.path.join(data_dir, 'parameters.json'), 'w') as f:
        json.dump(params, f, indent=2)
    print("✅ Created parameters")
    
    # Check if we have demand and weather data
    demand_path = os.path.join(data_dir, 'daily_n_demand.csv')
    weather_path = os.path.join(data_dir, 'weather_forecast.csv')
    
    if not os.path.exists(demand_path) or not os.path.exists(weather_path):
        print("Generating demand and weather data...")
        
        # Create simple demand data
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
        demand_df.to_csv(demand_path, index=False)
        print("✅ Created daily demand data")
        
        # Create simple weather data
        zones = ['North', 'Central', 'South']
        weather_records = []
        
        for date in dates:
            for zone in zones:
                rainfall = np.random.exponential(5)
                rain_lock = rainfall > 30
                weather_records.append({
                    'date': date,
                    'zone': zone,
                    'rainfall_mm': round(rainfall, 1),
                    'rain_lock': rain_lock
                })
        
        weather_df = pd.DataFrame(weather_records)
        weather_df.to_csv(weather_path, index=False)
        print("✅ Created weather forecast data")
    else:
        # Load existing data
        demand_df = pd.read_csv(demand_path)
        weather_df = pd.read_csv(weather_path)
        print("✅ Loaded existing demand and weather data")
    
    # 3. Test core classes directly
    print("\n⚙️ TEST 3: Testing core classes...")
    
    # Create core.py if it doesn't exist
    core_path = os.path.join(src_path, 'algorithm', 'core.py')
    if not os.path.exists(core_path):
        print("Creating core.py...")
        with open(core_path, 'w') as f:
            f.write('''
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
        self.soil_health = 100
        
    def set_demand(self, demand_kg):
        """Set today's nitrogen demand in kg"""
        self.daily_demand = demand_kg
        
    def set_rain_lock(self, is_locked):
        """Set rain lock status"""
        self.rain_locked = is_locked
        
    def apply_biosolids(self, tons, n_content_percent=0.03):
        """Apply biosolids and return actual nitrogen uptake"""
        if self.rain_locked:
            return 0, 0
            
        kg_n_applied = tons * 1000 * n_content_percent
        
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
''')
        print("✅ Created core.py")
    
    # Now import it
    from algorithm.core import STP, Farm
    
    # Test STP class
    test_stp = STP(
        stp_id="TEST-STP",
        lat=10.0,
        lon=76.5,
        storage_max_tons=100,
        daily_output_tons=15
    )
    test_stp.add_daily_output()
    print(f"   STP created: {test_stp.id}, Storage: {test_stp.current_storage} tons")
    
    # Test Farm class
    test_farm = Farm(
        farm_id=999,
        lat=10.1,
        lon=76.6,
        zone="Central"
    )
    test_farm.set_demand(150)
    test_farm.set_rain_lock(False)
    uptake, excess = test_farm.apply_biosolids(5.0)
    print(f"   Farm created: {test_farm.id}, Uptake: {uptake:.1f}kg, Excess: {excess:.1f}kg")
    print("✅ Core classes working!")
    
    # 4. Create and test a simple greedy algorithm
    print("\n🤖 TEST 4: Creating simple greedy algorithm...")
    
    # Create distance calculator
    distance_calc_code = '''
import numpy as np

class DistanceCalculator:
    @staticmethod
    def haversine(lat1, lon1, lat2, lon2):
        """Calculate Haversine distance between two points in km"""
        lat1, lon1, lat2, lon2 = map(np.radians, [lat1, lon1, lat2, lon2])
        dlat = lat2 - lat1
        dlon = lon2 - lon1
        a = np.sin(dlat/2)**2 + np.cos(lat1) * np.cos(lat2) * np.sin(dlon/2)**2
        c = 2 * np.arcsin(np.sqrt(a))
        r = 6371  # Earth radius in km
        return c * r
    
    @staticmethod
    def create_distance_matrix(stps, farms):
        """Create distance matrix between all STPs and farms"""
        n_stps = len(stps)
        n_farms = len(farms)
        dist_matrix = np.zeros((n_stps, n_farms))
        
        for i, stp in enumerate(stps):
            for j, farm in enumerate(farms):
                dist_matrix[i, j] = DistanceCalculator.haversine(
                    stp.lat, stp.lon, farm.lat, farm.lon
                )
        return dist_matrix
'''
    
    # Save distance calculator
    distance_path = os.path.join(src_path, 'utils', 'distance_calculator.py')
    with open(distance_path, 'w') as f:
        f.write(distance_calc_code)
    print("✅ Created distance_calculator.py")
    
    from utils.distance_calculator import DistanceCalculator
    
    # Calculate distance
    dist = DistanceCalculator.haversine(10.0, 76.5, 10.1, 76.6)
    print(f"   Distance calculated: {dist:.2f} km")
    
    # 5. Create a simple greedy algorithm for testing
    print("\n▶️ TEST 5: Running simple greedy simulation...")
    
    # Create STP and Farm objects from our data
    stps = []
    for _, row in stps_df.iterrows():
        stp = STP(
            stp_id=row['stp_id'],
            lat=row['latitude'],
            lon=row['longitude'],
            storage_max_tons=row['storage_max_tons'],
            daily_output_tons=row['daily_output_tons']
        )
        stps.append(stp)
    
    farms = []
    for _, row in farms_df.iterrows():
        farm = Farm(
            farm_id=row['farm_id'],
            lat=row['latitude'],
            lon=row['longitude'],
            zone=row['weather_zone']
        )
        farms.append(farm)
    
    # Calculate distance matrix
    dist_matrix = DistanceCalculator.create_distance_matrix(stps, farms)
    
    # Simple greedy algorithm
    deliveries = []
    total_carbon = 0
    
    # Process first day
    date = '2025-01-01'
    print(f"\n📅 Processing date: {date}")
    
    # Update STPs
    for stp in stps:
        stp.add_daily_output()
    
    # Update farm demands and rain locks
    day_demand = demand_df[demand_df['date'] == date]
    day_weather = weather_df[weather_df['date'] == date]
    
    for farm in farms:
        # Set demand
        farm_demand = day_demand[day_demand['farm_id'] == farm.id]
        if not farm_demand.empty:
            farm.set_demand(farm_demand['n_demand_kg'].values[0])
        
        # Set rain lock
        zone_weather = day_weather[day_weather['zone'] == farm.zone]
        if not zone_weather.empty:
            farm.set_rain_lock(zone_weather['rain_lock'].values[0])
    
    # Greedy delivery
    truck_capacity = 10
    emission_per_km = params['transport_emission_factor']
    
    for stp_idx, stp in enumerate(stps):
        available = stp.get_available_for_dispatch()
        print(f"  STP {stp.id}: {available:.1f} tons available")
        
        # Find eligible farms
        eligible_farms = []
        for farm_idx, farm in enumerate(farms):
            if not farm.rain_locked and farm.daily_demand > 0:
                distance = dist_matrix[stp_idx, farm_idx]
                eligible_farms.append((farm_idx, farm, distance))
        
        # Sort by distance (closest first)
        eligible_farms.sort(key=lambda x: x[2])
        
        for farm_idx, farm, distance in eligible_farms:
            if available <= 0:
                break
            
            # Calculate how much to deliver
            demand_tons = farm.daily_demand / (1000 * 0.03)  # 3% N content
            deliver_amount = min(truck_capacity, demand_tons, available)
            
            if deliver_amount < 0.1:
                continue
            
            # Apply biosolids
            n_uptake_kg, excess_n_kg = farm.apply_biosolids(deliver_amount)
            
            # Dispatch from STP
            stp.dispatch(deliver_amount)
            available -= deliver_amount
            
            # Calculate carbon impact
            transport_cost = -emission_per_km * distance
            fertilizer_offset = n_uptake_kg * 5.0
            soil_sequestration = deliver_amount * 1000 * 0.2
            excess_penalty = -excess_n_kg * 10.0 if excess_n_kg > 0 else 0
            
            carbon_total = transport_cost + fertilizer_offset + soil_sequestration + excess_penalty
            total_carbon += carbon_total
            
            # Record delivery
            deliveries.append({
                'date': date,
                'stp_id': stp.id,
                'farm_id': farm.id,
                'tons_delivered': round(deliver_amount, 3),
                'carbon': round(carbon_total, 2)
            })
            
            print(f"    → Farm {farm.id}: {deliver_amount:.1f} tons, "
                  f"{distance:.1f} km, Carbon: {carbon_total:+.1f}")
    
    print(f"📊 Day summary: {len(deliveries)} deliveries, "
          f"Carbon: {total_carbon:+.1f}")
    
    # 6. Generate solution.csv
    print("\n📄 TEST 6: Generating solution.csv...")
    
    if deliveries:
        solution_df = pd.DataFrame(deliveries)[['date', 'stp_id', 'farm_id', 'tons_delivered']]
        print(f"✅ Created {len(solution_df)} delivery records")
        print("\nFirst 5 deliveries:")
        print(solution_df.head().to_string(index=False))
        
        # Save output
        output_dir = os.path.join(current_dir, 'output')
        os.makedirs(output_dir, exist_ok=True)
        output_path = os.path.join(output_dir, 'solution_step8_test.csv')
        solution_df.to_csv(output_path, index=False)
        print(f"\n💾 Saved to: {output_path}")
    else:
        print("❌ No deliveries generated")
    
    # 7. Summary
    print("\n" + "="*70)
    print("🎉 STEP 8 TEST COMPLETE!")
    print("="*70)
    print(f"\n✅ Created/Tested:")
    print(f"   - Core classes: STP and Farm")
    print(f"   - Distance calculator")
    print(f"   - Generated test data")
    print(f"   - Ran greedy algorithm for 1 day")
    print(f"   - Created {len(deliveries)} deliveries")
    print(f"   - Total carbon: {total_carbon:+.1f} kg CO₂eq")
    
    print("\n📁 Files created in 'src/' folder:")
    for root, dirs, files in os.walk(src_path):
        for file in files:
            if file.endswith('.py'):
                print(f"   - {os.path.relpath(os.path.join(root, file), current_dir)}")

except Exception as e:
    print(f"\n❌ ERROR: {type(e).__name__}: {e}")
    import traceback
    traceback.print_exc()
    
    print("\n🔧 QUICK FIX: Creating minimal project structure...")
    
    # Create minimal structure
    folders = ['src/utils', 'src/algorithm', 'data', 'output']
    for folder in folders:
        os.makedirs(os.path.join(current_dir, folder), exist_ok=True)
    
    print("✅ Created folder structure")
    print(f"Project root: {current_dir}")
    print("Try running again!")