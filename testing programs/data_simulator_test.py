import pandas as pd
import numpy as np
import os
from datetime import datetime, timedelta

class DataSimulator:
    """
    Generates synthetic daily nitrogen demand and weather forecast data.
    This is needed if the hackathon doesn't provide daily_n_demand.csv.
    """
    
    def __init__(self, data_path="data", num_farms=250, start_date="2025-01-01"):
        self.data_path = data_path
        self.num_farms = num_farms
        self.start_date = datetime.strptime(start_date, "%Y-%m-%d")
        
        # Create data directory if it doesn't exist
        os.makedirs(data_path, exist_ok=True)
        
    def generate_daily_demand(self, num_days=365, save_to_file=True):
        """
        Generate realistic daily nitrogen demand for all farms.
        
        Args:
            num_days: Number of days to simulate (default: 365 for full year)
            save_to_file: Whether to save to CSV
            
        Returns:
            DataFrame with columns: date, farm_id, n_demand_kg
        """
        print("=" * 60)
        print("GENERATING DAILY NITROGEN DEMAND DATA")
        print("=" * 60)
        
        dates = [self.start_date + timedelta(days=i) for i in range(num_days)]
        all_data = []
        
        total_records = num_days * self.num_farms
        print(f"Creating {total_records:,} demand records...")
        
        for day_idx, date in enumerate(dates):
            day_of_year = date.timetuple().tm_yday
            
            for farm_id in range(1, self.num_farms + 1):
                # Add some uniqueness per farm
                farm_seed = (farm_id * 17) % 100
                
                # Base seasonal pattern for Kerala agriculture
                # Planting seasons: Jan-Mar (rice), Jun-Aug (monsoon crops), Sep-Dec (second crop)
                if 1 <= day_of_year <= 90:  # Jan-Mar: First crop season
                    base_demand = 100 + farm_seed * 0.5
                    season_factor = 1.3
                elif 151 <= day_of_year <= 240:  # Jun-Aug: Monsoon season
                    base_demand = 60 + farm_seed * 0.3
                    season_factor = 0.7  # Lower demand during heavy rain
                elif 241 <= day_of_year <= 300:  # Sep-Oct: Post-monsoon
                    base_demand = 80 + farm_seed * 0.4
                    season_factor = 1.1
                else:  # Other periods
                    base_demand = 70 + farm_seed * 0.4
                    season_factor = 1.0
                
                # Weekly pattern (higher demand on weekdays)
                weekday_factor = 1.2 if date.weekday() < 5 else 0.8
                
                # Random variation (but not too random for realism)
                random_factor = 1 + np.random.normal(0, 0.15)
                
                # Calculate final demand (ensure non-negative)
                demand = max(10, base_demand * season_factor * weekday_factor * random_factor)
                
                all_data.append({
                    'date': date.strftime("%Y-%m-%d"),
                    'farm_id': farm_id,
                    'n_demand_kg': round(demand, 2)
                })
            
            # Progress indicator
            if (day_idx + 1) % 50 == 0:
                print(f"  Generated demand for day {day_idx + 1}/{num_days}")
        
        df = pd.DataFrame(all_data)
        
        if save_to_file:
            output_path = os.path.join(self.data_path, "daily_n_demand.csv")
            df.to_csv(output_path, index=False)
            print(f"\n✅ Saved daily demand data to: {output_path}")
            print(f"   Total records: {len(df):,}")
            print(f"   Date range: {df['date'].min()} to {df['date'].max()}")
            print(f"   Average daily demand per farm: {df['n_demand_kg'].mean():.1f} kg")
        
        return df
    
    def generate_weather_forecast(self, num_days=365, save_to_file=True):
        """
        Generate 5-day weather forecast for Kerala's 5 zones.
        
        Args:
            num_days: Number of days to forecast
            save_to_file: Whether to save to CSV
            
        Returns:
            DataFrame with columns: date, zone, rainfall_mm, rain_lock
        """
        print("\n" + "=" * 60)
        print("GENERATING WEATHER FORECAST DATA")
        print("=" * 60)
        
        # Kerala's 5 weather zones
        zones = ['North', 'Central', 'South', 'Highlands', 'Coastal']
        
        dates = [self.start_date + timedelta(days=i) for i in range(num_days)]
        weather_data = []
        
        print(f"Creating weather forecast for {num_days} days across {len(zones)} zones...")
        
        for date in dates:
            month = date.month
            day_of_year = date.timetuple().tm_yday
            
            for zone in zones:
                # Base rainfall pattern for Kerala
                # Peak monsoon: June to September
                if 6 <= month <= 9:  # Monsoon months
                    if zone == 'Highlands':
                        base_rain = np.random.exponential(scale=20)  # Highest in highlands
                    elif zone == 'Coastal':
                        base_rain = np.random.exponential(scale=15)  # High in coastal
                    else:
                        base_rain = np.random.exponential(scale=12)  # Moderate elsewhere
                elif month in [10, 11]:  # Post-monsoon
                    base_rain = np.random.exponential(scale=8)
                elif month in [4, 5]:  # Pre-monsoon showers
                    base_rain = np.random.exponential(scale=6)
                else:  # Dry season
                    base_rain = np.random.exponential(scale=3)
                
                # Add occasional heavy rain events
                if np.random.random() < 0.05:  # 5% chance of heavy rain event
                    base_rain = base_rain * np.random.uniform(2, 5)
                
                # Add some zone-specific variation
                zone_factor = {
                    'North': 1.1,
                    'Central': 1.0,
                    'South': 0.9,
                    'Highlands': 1.3,
                    'Coastal': 1.2
                }[zone]
                
                rainfall = max(0, base_rain * zone_factor + np.random.normal(0, 2))
                rainfall = round(rainfall, 1)
                
                # Determine rain lock (30mm threshold)
                rain_lock = rainfall > 30
                
                weather_data.append({
                    'date': date.strftime("%Y-%m-%d"),
                    'zone': zone,
                    'rainfall_mm': rainfall,
                    'rain_lock': rain_lock
                })
        
        df = pd.DataFrame(weather_data)
        
        if save_to_file:
            output_path = os.path.join(self.data_path, "weather_forecast.csv")
            df.to_csv(output_path, index=False)
            
            # Calculate some stats
            rain_lock_days = df['rain_lock'].sum()
            total_days = len(dates) * len(zones)
            
            print(f"\n✅ Saved weather forecast to: {output_path}")
            print(f"   Total records: {len(df):,}")
            print(f"   Rain-lock events: {rain_lock_days:,} ({rain_lock_days/total_days*100:.1f}% of zone-days)")
            print(f"   Average rainfall: {df['rainfall_mm'].mean():.1f} mm")
            print(f"   Max rainfall: {df['rainfall_mm'].max()} mm")
            
            # Show rain lock distribution by zone
            print("\n   Rain-lock days by zone:")
            for zone in zones:
                zone_locks = df[df['zone'] == zone]['rain_lock'].sum()
                print(f"     {zone}: {zone_locks} days")
        
        return df
    
    def generate_all_data(self):
        """Generate both demand and weather data"""
        demand_df = self.generate_daily_demand()
        weather_df = self.generate_weather_forecast()
        return demand_df, weather_df

# ============================================================================
# MAIN EXECUTION - This runs when you execute the file directly
# ============================================================================

if __name__ == "__main__":
    print("\n" + "=" * 60)
    print("KERALA BIOSOLIDS DATA SIMULATOR")
    print("=" * 60)
    
    # Create simulator
    simulator = DataSimulator(data_path="data", num_farms=250, start_date="2025-01-01")
    
    # Generate all data
    print("\nStarting data generation...")
    demand_df, weather_df = simulator.generate_all_data()
    
    print("\n" + "=" * 60)
    print("DATA GENERATION COMPLETE!")
    print("=" * 60)
    
    # Show sample of generated data
    print("\n📊 SAMPLE OF GENERATED DATA:")
    print("\nDaily Demand (first 5 farms, first 3 days):")
    sample_demand = demand_df[
        (demand_df['farm_id'] <= 5) & 
        (demand_df['date'] <= '2025-01-03')
    ].pivot(index='date', columns='farm_id', values='n_demand_kg')
    print(sample_demand.to_string())
    
    print("\nWeather Forecast (first 3 days, all zones):")
    sample_weather = weather_df[weather_df['date'] <= '2025-01-03']
    print(sample_weather.to_string(index=False))
    
    print("\n✅ Ready for algorithm development!")
    print("Files created in 'data/' folder:")
    print("  1. daily_n_demand.csv")
    print("  2. weather_forecast.csv")