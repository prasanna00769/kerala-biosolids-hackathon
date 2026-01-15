import sys
import os
sys.path.append('src')

from utils.data_loader import DataLoader
from algorithm.greedy_baseline import GreedyBaseline
import pandas as pd
import json

print("="*60)
print("7-DAY FULL SIMULATION")
print("="*60)

# Load data
loader = DataLoader()
params = loader.load_parameters()
stps_df = loader.load_stp_data()
farms_df = loader.load_farm_data()
demand_df = loader.load_daily_demand()
weather_df = pd.read_csv('data/weather_forecast.csv')

print(f"Loaded: {len(stps_df)} STPs, {len(farms_df)} farms")
print(f"Demand records: {len(demand_df)}")
print(f"Weather records: {len(weather_df)}")

# Create algorithm
print("\nCreating algorithm...")
algorithm = GreedyBaseline(params, stps_df, farms_df)

# Run 7 days (Jan 1-7)
dates = sorted(demand_df['date'].unique())[:7]
print(f"\nSimulating dates: {dates}")

total_carbon = 0
for date in dates:
    daily_carbon = algorithm.run_day(date, demand_df, weather_df)
    total_carbon += daily_carbon

print(f"\n✅ 7-day simulation complete!")
print(f"Total carbon: {total_carbon:+.1f} kg CO₂eq")
print(f"Total deliveries: {len(algorithm.deliveries)}")

# Generate solution.csv
solution_df = algorithm.get_solution_csv()
os.makedirs('submission', exist_ok=True)
solution_path = 'submission/solution.csv'
solution_df.to_csv(solution_path, index=False)
print(f"\n💾 Saved to: {solution_path}")
print(f"Shape: {solution_df.shape}")

# Generate summary_metrics.json
summary = algorithm.get_summary_metrics()
summary_path = 'submission/summary_metrics.json'
with open(summary_path, 'w') as f:
    json.dump(summary, f, indent=2)
print(f"💾 Saved to: {summary_path}")

print("\n" + "="*60)
print("✅ READY FOR SUBMISSION!")
print("="*60)
print("Files in 'submission/' folder:")
print("  1. solution.csv - Your delivery schedule")
print("  2. summary_metrics.json - Performance metrics")