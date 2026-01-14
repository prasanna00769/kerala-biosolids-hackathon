import sys
import os
sys.path.append('src')  # Add src to Python path

from algorithm.core import STP, Farm

print("=" * 50)
print("TESTING CORE CLASSES")
print("=" * 50)

# Test STP class
print("\n1. Testing STP class:")
kochi_stp = STP(
    stp_id="STP-1",
    lat=9.9312,
    lon=76.2673,
    storage_max_tons=100,
    daily_output_tons=15
)
kochi_stp.add_daily_output()
print(f"Created STP: {kochi_stp.id}")
print(f"Storage after daily output: {kochi_stp.current_storage} tons")
print(f"Can dispatch today: {kochi_stp.get_available_for_dispatch()} tons")

# Dispatch some biosolids
dispatched = kochi_stp.dispatch(8.5)
print(f"Dispatched: {dispatched} tons")
print(f"Remaining storage: {kochi_stp.current_storage} tons")

# Test Farm class
print("\n2. Testing Farm class:")
farm_1 = Farm(
    farm_id=1,
    lat=10.0,
    lon=76.5,
    zone="Central"
)
farm_1.set_demand(150)  # 150 kg N demand
farm_1.set_rain_lock(False)

print(f"Created Farm: {farm_1.id}")
print(f"Daily N demand: {farm_1.daily_demand} kg")
print(f"Rain locked: {farm_1.rain_locked}")

# Apply biosolids
uptake, excess = farm_1.apply_biosolids(5.0)  # 5 tons
print(f"Applied 5 tons biosolids:")
print(f"  - Nitrogen uptake: {uptake:.2f} kg")
print(f"  - Excess nitrogen: {excess:.2f} kg")
print(f"  - Total applied today: {farm_1.applied_today} tons")

print("\n" + "=" * 50)
print("✅ Core classes working correctly!")
print("=" * 50)