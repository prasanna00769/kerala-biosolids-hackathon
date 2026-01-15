import pandas as pd
import json
from pathlib import Path
from typing import Optional, Dict, Tuple
import logging


class DataLoader:
    """
    A robust data loader for STP (Sewage Treatment Plant) and farm data.

    Handles loading and validation of:
    - Global parameters (JSON)
    - STP registry (CSV)
    - Farm locations (CSV)
    - Daily nitrogen demand (CSV)
    """

    def __init__(self, data_path: str = "data", logger: Optional[logging.Logger] = None):
        """
        Initialize DataLoader.

        Args:
            data_path: Path to the data directory (relative to project root)
            logger: Optional logger instance
        """
        # Resolve data path relative to project root (safe)
        self.data_path = Path(__file__).resolve().parents[2] / data_path

        self.logger = logger or self._setup_logger()

        if not self.data_path.exists():
            raise FileNotFoundError(f"Data directory not found: {self.data_path}")

        # Logical → physical filename mapping (NO renaming required)
        self.file_map = {
            "parameters": "config.json",
            "stp": "stp_registry.csv",
            "farms": "farm_locations.csv",
            "daily_demand": "daily_n_demand.csv",
        }

    def _setup_logger(self) -> logging.Logger:
        logger = logging.getLogger(__name__)
        logger.setLevel(logging.INFO)

        if not logger.handlers:
            handler = logging.StreamHandler()
            formatter = logging.Formatter(
                "%(asctime)s - %(levelname)s - %(message)s"
            )
            handler.setFormatter(formatter)
            logger.addHandler(handler)

        return logger

    def _load_file(self, filename: str, file_type: str):
        file_path = self.data_path / filename

        if not file_path.exists():
            raise FileNotFoundError(f"File not found: {file_path}")

        try:
            if file_type == "csv":
                df = pd.read_csv(file_path)
                self.logger.info(f"Loaded {filename} ({len(df)} records)")
                return df

            if file_type == "json":
                with open(file_path, "r") as f:
                    data = json.load(f)
                self.logger.info(f"Loaded {filename}")
                return data

            raise ValueError(f"Unsupported file type: {file_type}")

        except Exception as e:
            self.logger.error(f"Failed to load {filename}: {e}")
            raise

    # ------------------ Public Loaders ------------------

    def load_parameters(self) -> Dict:
        return self._load_file(self.file_map["parameters"], "json")

    def load_stp_data(self) -> pd.DataFrame:
        df = self._load_file(self.file_map["stp"], "csv")

        if "stp_id" not in df.columns:
            self.logger.warning("Missing column: stp_id")

        return df

    def load_farm_data(self) -> pd.DataFrame:
        df = self._load_file(self.file_map["farms"], "csv")

        if "farm_id" not in df.columns:
            self.logger.warning("Missing column: farm_id")

        return df

    def load_daily_demand(self, date: Optional[str] = None) -> pd.DataFrame:
        df = self._load_file(self.file_map["daily_demand"], "csv")

        required = {"date", "farm_id", "n_demand_kg"}
        missing = required - set(df.columns)
        if missing:
            raise ValueError(f"Missing required columns: {missing}")

        df["date"] = pd.to_datetime(df["date"])

        if date:
            df = df[df["date"] == pd.to_datetime(date)]
            self.logger.info(f"Filtered daily demand for {date}: {len(df)} records")

        return df

    # ------------------ Validation ------------------

    def validate_data(self) -> Tuple[Dict, pd.DataFrame, pd.DataFrame]:
        print("=" * 70)
        print("DATA VALIDATION REPORT".center(70))
        print("=" * 70)

        params = self.load_parameters()
        stps = self.load_stp_data()
        farms = self.load_farm_data()

        print("\nGLOBAL PARAMETERS")
        print("-" * 70)
        for k, v in params.items():
            print(f"{k:30s}: {v}")

        print(f"\nSTP REGISTRY ({len(stps)} records)")
        print(stps.head())

        print("\nFARM LOCATIONS ({len(farms)} records)")
        print(farms.head())

        print("\nVALIDATION COMPLETE")
        print("=" * 70)

        return params, stps, farms

    def get_data_summary(self) -> Dict:
        summary = {}

        try:
            summary["parameters"] = len(self.load_parameters())
        except Exception as e:
            summary["parameters"] = str(e)

        try:
            stps = self.load_stp_data()
            summary["stp_count"] = len(stps)
        except Exception as e:
            summary["stp_count"] = str(e)

        try:
            farms = self.load_farm_data()
            summary["farm_count"] = len(farms)
        except Exception as e:
            summary["farm_count"] = str(e)

        try:
            demand = self.load_daily_demand()
            summary["demand_records"] = len(demand)
        except Exception as e:
            summary["demand_records"] = str(e)

        return summary


# ------------------ Run Directly ------------------

if __name__ == "__main__":
    loader = DataLoader(data_path="data")

    params, stps, farms = loader.validate_data()

    summary = loader.get_data_summary()
    print("\nDATA SUMMARY")
    print(json.dumps(summary, indent=2, default=str))
