# src/config.py
import os

# --- DATASET CONFIGURATION ---
DATA_FILENAME = "Data_DBH_50_hec.csv"

# Column Definitions
COL_ID = 'TAG'
COL_SPECIES = 'SP'
COL_SPECIES_GRP = 'Species Group'
COL_X = 'XCO'
COL_Y = 'YCO'   # Column for X Coordinate

# Time-Step Logic (Extended for Trend Analysis)
# We map the CSV columns to Years
COL_YEARS = {
    1995: 'D95',  # Added for longer history
    2000: 'D00', # Historical Baseline
    2005: 'D05',  # Current State
    2010: 'D10'   # Target/Validation
}

COL_CURRENT = 'D05'   
COL_HISTORY = 'D00'   
COL_TARGET  = 'D10'   

# Settings
DEFAULT_MIN_DBH = 5.0 

# Model Artifacts
MODEL_FILENAME = "forest_growth_model.pkl"       
MORTALITY_MODEL_FILENAME = "forest_mortality_model.pkl" 
ENCODER_FILENAME = "species_encoder.pkl"

# Parameters
DENSITY_RADIUS = 5.0
COMPETITION_RADIUS = 6.0
TEST_SIZE = 0.2
RANDOM_STATE = 42
DEFAULT_GROWTH_PERCENTILE = 25