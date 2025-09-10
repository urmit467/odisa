# simple_fertilizer_recommender.py
def recommend_fertilizer(crop, soil_N=None, soil_P=None, soil_K=None):
    # Simple fertilizer recommendation based on crop type and soil nutrients
    # These are example values - you should adjust based on agronomic knowledge
    crop_recommendations = {
        'rice': {'N': 120, 'P': 60, 'K': 60},
        'maize': {'N': 100, 'P': 50, 'K': 50},
        'groundnut': {'N': 20, 'P': 40, 'K': 40},
        'pulses': {'N': 20, 'P': 40, 'K': 20}
    }
    
    if crop not in crop_recommendations:
        return {'N': 0, 'P': 0, 'K': 0}
    
    base_rec = crop_recommendations[crop].copy()
    
    # Adjust based on soil nutrients if available
    if soil_N is not None:
        base_rec['N'] = max(0, base_rec['N'] - soil_N * 0.3)
    if soil_P is not None:
        base_rec['P'] = max(0, base_rec['P'] - soil_P * 0.3)
    if soil_K is not None:
        base_rec['K'] = max(0, base_rec['K'] - soil_K * 0.3)
    
    return {k: round(v, 2) for k, v in base_rec.items()}