
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
