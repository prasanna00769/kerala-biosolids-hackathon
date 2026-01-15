import json
import pandas as pd

class MetricsCalculator:
    def __init__(self, params, deliveries_df):
        self.params = params
        self.deliveries = deliveries_df
        
    def calculate_summary(self):
        """Calculate summary metrics for the entire period"""
        if self.deliveries.empty:
            return {
                "total_carbon_credits": 0,
                "total_deliveries": 0,
                "total_tons_delivered": 0,
                "average_distance_km": 0,
                "days_with_overflow": 0,
                "days_with_rain_lock": 0
            }
        
        # Calculate totals
        total_tons = self.deliveries['tons_delivered'].sum()
        total_deliveries = len(self.deliveries)
        
        # Calculate days covered
        unique_dates = self.deliveries['date'].nunique()
        
        # For now, return basic metrics
        # Note: Actual carbon calculation should be done in algorithm
        
        summary = {
            "total_carbon_credits": "TBD",  # Will be filled by algorithm
            "total_deliveries": int(total_deliveries),
            "total_tons_delivered": float(round(total_tons, 2)),
            "average_tons_per_delivery": float(round(total_tons/total_deliveries, 2)) if total_deliveries > 0 else 0,
            "days_operated": int(unique_dates),
            "stp_utilization_rate": "TBD",
            "farm_coverage_rate": "TBD"
        }
        
        return summary
    
    def save_summary_json(self, output_path="submission/summary_metrics.json"):
        """Save summary metrics to JSON file"""
        summary = self.calculate_summary()
        with open(output_path, 'w') as f:
            json.dump(summary, f, indent=2)
        return summary