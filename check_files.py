import pandas as pd, json, os

# Check solution.csv
df = pd.read_csv('submission/solution.csv')
print(f"✅ solution.csv: {len(df)} deliveries")
print(f"   Dates: {sorted(df['date'].unique())[:3]}...")
print(f"   Total tons: {df['tons_delivered'].sum():.1f}")

# Check summary_metrics.json
with open('submission/summary_metrics.json', 'r') as f:
    summary = json.load(f)
print(f"\n✅ summary_metrics.json:")
print(f"   Carbon credits: {summary.get('total_carbon_credits_kg', 0)} kg")