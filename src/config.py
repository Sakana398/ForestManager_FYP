# src/config.py
import os

# --- DATASET CONFIGURATION ---
DATA_FILENAME = "Data_DBH_50_hec.csv"

# Column Definitions
COL_ID = 'TAG'
COL_SPECIES = 'SP'
COL_SPECIES_GRP = 'Species Group'
COL_X = 'XCO'
COL_Y = 'YCO'

# Time-Step Logic
COL_YEARS = {
    1995: 'D95',
    2000: 'D00',
    2005: 'D05', 
    2010: 'D10'
}

COL_CURRENT = 'D05'   
COL_HISTORY = 'D00'   
COL_TARGET  = 'D10'   

# Settings
DEFAULT_MIN_DBH = 5.0 

# Physics / Geography Constants
# approx 111,111 meters per degree at the equator
METERS_PER_DEGREE = 111111.0
DEG_PER_METER = 1.0 / METERS_PER_DEGREE

# Parameters
DENSITY_RADIUS = 5.0       # Meters
COMPETITION_RADIUS = 6.0   # Meters
TEST_SIZE = 0.2
RANDOM_STATE = 42
DEFAULT_GROWTH_PERCENTILE = 25

# Model Artifacts
MODEL_FILENAME = "forest_growth_model.pkl"       
MORTALITY_MODEL_FILENAME = "forest_mortality_model.pkl" 
ENCODER_FILENAME = "species_encoder.pkl"