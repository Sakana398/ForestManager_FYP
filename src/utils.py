# src/utils.py
import pandas as pd
import numpy as np
import joblib
import streamlit as st
from scipy.spatial import cKDTree
from sklearn.base import BaseEstimator
from sklearn.preprocessing import LabelEncoder
from typing import Tuple, Optional
from .config import *

# ==========================================
# 1. COORDINATE & SPATIAL UTILITIES
# ==========================================
def standardize_coordinates(df: pd.DataFrame) -> pd.DataFrame:
    """
    Auto-detects if coordinates are in Meters or GPS (Degrees).
    Forces everything to GPS (Lat/Lon) for consistent mapping.
    """
    df = df.copy()
    
    # Pasoh Forest Anchor (Fixed)
    PASOH_LAT = 2.9788
    PASOH_LON = 102.3131
    
    # Check if X looks like Meters (Values > 180 usually mean meters in this context)
    if COL_X in df.columns and df[COL_X].max() > 180:
        # Convert Meters -> Degrees
        df['lon'] = PASOH_LON + (df[COL_X] * DEG_PER_METER)
        df['lat'] = PASOH_LAT + (df[COL_Y] * DEG_PER_METER)
    elif COL_X in df.columns:
        # Already GPS
        df['lon'] = df[COL_X]
        df['lat'] = df[COL_Y]
        
    return df

def add_spatial_features(df: pd.DataFrame) -> pd.DataFrame:
    """
    Calculates Density, Nearest Neighbor, Competition Index, AND Interactions.
    Centralized logic used by both Training and Inference.
    """
    # Filter valid rows for spatial calculation
    # We need X, Y and Current Size
    valid = df.dropna(subset=[COL_X, COL_Y, COL_CURRENT]).copy()
    
    if valid.empty:
        return df

    # Ensure we are working with Degrees for search, but Meters for values
    coords = valid[[COL_X, COL_Y]].values
    tree_idx = cKDTree(coords)
    
    # A. Nearest Neighbor (Distance in Meters)
    dists, _ = tree_idx.query(coords, k=2)
    # dists[:, 1] is the distance to the 2nd closest point (1st is itself)
    # We assume coords are in DEGREES, so we convert to METERS
    valid['Nearest_Neighbor_Dist'] = dists[:, 1] * METERS_PER_DEGREE
    
    # B. Local Density (Count within Radius)
    radius_deg = DENSITY_RADIUS * DEG_PER_METER
    counts = tree_idx.query_ball_point(coords, r=radius_deg, return_length=True)
    valid['Local_Density'] = counts - 1 # Exclude self

    # C. Competition Index (Hegyi)
    # Search Radius = 6m
    radius_ci_deg = COMPETITION_RADIUS * DEG_PER_METER
    neighbor_indices = tree_idx.query_ball_point(coords, r=radius_ci_deg)
    dbh_vals = valid[COL_CURRENT].values
    
    ci_list = []
    
    for i, neighbors in enumerate(neighbor_indices):
        subject_dbh = dbh_vals[i]
        ci_score = 0.0
        
        if subject_dbh > 0:
            for n_idx in neighbors:
                if i == n_idx: continue
                
                # Distance in Degrees -> Meters
                dist_deg = np.linalg.norm(coords[i] - coords[n_idx])
                dist_m = dist_deg * METERS_PER_DEGREE
                
                if dist_m > 0.1: # Prevent division by zero
                    # Hegyi: Sum( (NeighborDBH / SubjectDBH) / Distance )
                    ci_score += (dbh_vals[n_idx] / subject_dbh) / dist_m
                    
        ci_list.append(ci_score)
        
    valid['Competition_Index'] = ci_list

    # D. INTERACTION FEATURES (Species * Competition)
    # -----------------------------------------------------
    # We calculate the "Mean Growth" per species to act as a proxy for "Vigor/Tolerance".
    # Fast-growing species (Light Demanding) usually suffer more from competition.
    if COL_SPECIES in valid.columns and COL_CURRENT in valid.columns:
        # Calculate proxy for species vigor (Mean Size in this plot)
        # Using transform ensures we get a value for every row aligned with index
        sp_means = valid.groupby(COL_SPECIES)[COL_CURRENT].transform('mean')
        
        # Interaction: Competition * Species Vigor
        # This helps the model see: "High Competition is WORSE for Fast Growers"
        valid['Interaction_Vigor_Comp'] = valid['Competition_Index'] * sp_means
    else:
        valid['Interaction_Vigor_Comp'] = 0.0
    
    # Merge back features to original dataframe
    # We use left merge to keep all original rows
    cols_to_merge = [COL_ID, 'Nearest_Neighbor_Dist', 'Local_Density', 'Competition_Index', 'Interaction_Vigor_Comp']
    
    # Drop columns if they already exist to avoid _x _y duplicates
    df = df.drop(columns=[c for c in cols_to_merge if c in df.columns and c != COL_ID], errors='ignore')
    
    return df.merge(valid[cols_to_merge], on=COL_ID, how='left')

# ==========================================
# 2. DATA LOADING
# ==========================================
@st.cache_data(show_spinner=True)
def load_and_process_data(csv_path: str, min_dbh_limit: float = DEFAULT_MIN_DBH) -> pd.DataFrame:
    try:
        try:
            df = pd.read_csv(csv_path, encoding='utf-8-sig')
        except:
            df = pd.read_csv(csv_path, encoding='latin1')

        df.columns = df.columns.str.strip()
        
        # String Cleaning
        for col in [COL_SPECIES, COL_SPECIES_GRP]:
            if col in df.columns:
                df[col] = df[col].astype(str).str.strip()
        
        # Numeric Cleaning
        for c in [COL_HISTORY, COL_CURRENT, COL_TARGET]:
            if c in df.columns:
                df[c] = pd.to_numeric(df[c], errors='coerce')

        # 1. Filter Size
        if COL_CURRENT in df.columns:
            df = df[df[COL_CURRENT] >= min_dbh_limit]

        # 2. Calculate Growth History
        if COL_HISTORY in df.columns and COL_CURRENT in df.columns:
            hist_vals = df[COL_HISTORY].fillna(df[COL_CURRENT])
            df['GROWTH_HIST'] = df[COL_CURRENT] - hist_vals

        # 3. Standardize Coords (Add lat/lon columns)
        df = standardize_coordinates(df)
        
        # 4. Add Spatial Features (The heavy math)
        df = add_spatial_features(df)

        return df

    except Exception as e:
        st.error(f"Data Load Error: {e}")
        return pd.DataFrame()

# ==========================================
# 3. MODEL UTILITIES
# ==========================================
@st.cache_resource
def load_model_resources():
    try:
        model_grow = joblib.load(MODEL_FILENAME)
        try:
            model_mort = joblib.load(MORTALITY_MODEL_FILENAME)
        except:
            model_mort = None
        encoder = joblib.load(ENCODER_FILENAME)
        return model_grow, model_mort, encoder
    except FileNotFoundError:
        return None, None, None

def run_predictions(
    df: pd.DataFrame, 
    model_grow: BaseEstimator, 
    model_mort: Optional[BaseEstimator], 
    encoder: LabelEncoder
) -> pd.DataFrame:
    """Runs inference for Growth and Mortality."""
    
    # Encode Species
    if 'SP_Encoded' not in df.columns:
        known_classes = encoder.classes_
        # Safe map that handles unknown species by assigning -1
        map_dict = {sp: i for i, sp in enumerate(known_classes)}
        df['SP_Encoded'] = df[COL_SPECIES].map(map_dict).fillna(-1) 
    
    # Feature list matching the training script
    features = [
        COL_CURRENT, 'GROWTH_HIST', 
        'Nearest_Neighbor_Dist', 'Local_Density', 
        'Competition_Index', 'Interaction_Vigor_Comp', 
        'SP_Encoded'
    ]
    
    # Prepare Input
    pred_df = df[features].dropna()
    pred_df = pred_df[pred_df['SP_Encoded'] != -1]
    
    if not pred_df.empty:
        # A. Predict Growth (Increment)
        pred_inc = model_grow.predict(pred_df)
        
        # Future Size = Current + Increment (Prevent shrinking)
        future_size = np.maximum(pred_df[COL_CURRENT] + pred_inc, pred_df[COL_CURRENT])
        
        df.loc[pred_df.index, 'Predicted_Size'] = future_size
        df.loc[pred_df.index, 'Predicted_Growth'] = pred_inc
        
        # B. Predict Mortality
        if model_mort:
            # Predict Probability of Class 1 (Dead)
            probs = model_mort.predict_proba(pred_df)[:, 1]
            df.loc[pred_df.index, 'Mortality_Risk'] = probs
        else:
            df['Mortality_Risk'] = 0.0
            
    return df

def load_css(file_name="style.css"):
    """
    Loads a local CSS file and injects it into the Streamlit app.
    Call this at the top of every page to apply global styling.
    """
    try:
        with open(file_name) as f:
            st.markdown(f'<style>{f.read()}</style>', unsafe_allow_html=True)
    except FileNotFoundError:
        st.warning(f"⚠️ CSS file not found: {file_name}")