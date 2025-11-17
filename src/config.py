# src/config.py

# Data Settings
DATA_FILENAME = "Data Pasoh.csv"
MODEL_FILENAME = "forest_growth_model.pkl"
ENCODER_FILENAME = "species_encoder.pkl"

# Spatial Parameters
DENSITY_RADIUS = 5.0    # Meters for local density count
COMPETITION_RADIUS = 6.0 # Meters for Competition Index (neighbors that influence growth)

# Model Settings
TEST_SIZE = 0.2
RANDOM_STATE = 42

# Default Filter Values
DEFAULT_GROWTH_PERCENTILE = 25
DEFAULT_PROXIMITY = 10.0