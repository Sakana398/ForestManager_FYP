# Smart Forests Management: ForestManager App 🌳

**Predicting Plant Growth and Supporting Silviculture Thinning Decisions Using Spatial Data**

* **Author:** Syazmeel Adam Bin Shariful Azri (214637)
* **Supervisor:** Dr. Raihani Mohamed
* **Institution:** Universiti Putra Malaysia (UPM)
* **Faculty:** Faculty of Computer Science and Information Technology
* **Semester:** 2 2024/2025

---

## 📖 Project Overview

**ForestManager** is an interactive decision-support system designed to assist forest managers in the Pasoh Forest Reserve. Unlike traditional growth models that often ignore spatial context, this application integrates **Machine Learning (Linear Regression)** with **Spatial Competition Indices (Hegyi’s CI)** to predict individual tree growth and recommend precise silvicultural thinning operations.

### Key Problems Addressed:
1.  **Complexity of Tropical Forests:** Managing over 800 species requires data-driven tools rather than manual estimation.
2.  **Spatial Context:** Trees do not grow in isolation; their growth is heavily influenced by neighbors. This app models those interactions.
3.  **Actionable Insights:** Bridges the gap between raw ecological data and field-level decisions (e.g., "Which specific trees should be removed?").

---

## ✨ Key Features

* **📈 AI Growth Prediction:** Predicts future Diameter at Breast Height (DBH) using historical data and spatial features.
* **📍 Spatial Analysis Engine:** Automatically calculates:
    * **Hegyi's Competition Index (CI):** Measures pressure from neighboring trees.
    * **Local Density:** Trees within a 5m radius.
    * **Nearest Neighbor Distance:** Proximity to the closest competitor.
* **🌲 Dynamic Thinning Simulation:** Interactive sliders to filter trees based on growth performance, crowding, and proximity.
* **🗺️ "Before & After" Visualization:** Spatial maps that allow managers to toggle between the current forest state and the post-thinning scenario.
* **📥 Exportable Reports:** Generate and download CSV lists of thinning candidates for field use.

---

## 🛠️ Installation & Setup

Follow these steps to run the project locally on your machine.

### Prerequisites
* Python 3.10+
* Git

### 1. Clone the Repository
```bash
git clone [https://github.com/Syazmeel/ForestManager_FYP.git](https://github.com/Syazmeel/ForestManager_FYP.git)
cd ForestManager_FYP
````

### 2\. Create a Virtual Environment

It is recommended to use a virtual environment to manage dependencies.

```bash
# Windows
python -m venv venv
.\venv\Scripts\activate

# macOS/Linux
python3 -m venv venv
source venv/bin/activate
```

### 3\. Install Dependencies

```bash
pip install -r requirements.txt
```

-----

## 🚀 Usage Guide

The application consists of two parts: the **Model Training Pipeline** and the **Streamlit Dashboard**.

### Step 1: Train the Model (Run Locally First)

Before starting the app, you must process the raw data and train the machine learning model. This script calculates complex spatial metrics which takes time, so it is run offline.

```bash
python train_model.py
```

  * **Input:** `Data Pasoh.csv`
  * **Output:** `forest_growth_model.pkl` (Trained Model) & `species_encoder.pkl` (Label Encoder)

### Step 2: Launch the Dashboard

Once the model files are generated, launch the interactive interface:

```bash
streamlit run ForestManager_app.py
```

The app will open automatically in your default web browser (usually at `http://localhost:8501`).

-----

## 📂 Project Structure

This project uses a modular "Modulith" architecture for scalability.

```text
ForestManager_FYP/
│
├── Data Pasoh.csv             # Raw Dataset (Pasoh Forest Reserve)
├── requirements.txt           # Python dependencies
├── README.md                  # Project documentation
│
├── train_model.py             # Offline script for Feature Engineering & Model Training
├── ForestManager_app.py       # Main Entry Point (Home Page)
│
├── src/                       # Source Code & Logic
│   ├── config.py              # Configuration constants (Radius settings, File paths)
│   ├── utils.py               # Core logic: Spatial Calc, Data Loading, Prediction
│   └── components.py          # UI Components: Sidebars, Sliders
│
└── pages/                     # Streamlit Multi-Page Navigation
    ├── 1_Spatial_Map.py       # Visualization Page (Before/After View)
    └── 2_Individual_Growth.py # Tree-level historical trend viewer
```

-----

## 📊 Methodology Summary

1.  **Data Preprocessing:** Cleaning `Data Pasoh.csv`, handling missing values, and removing BOM artifacts.
2.  **Spatial Feature Engineering:** Using `scipy.spatial.cKDTree` to calculate Hegyi's Competition Index and neighbor distances.
3.  **Machine Learning:** A Linear Regression model trained on historical growth (2017-2019) to predict 2021 DBH.
      * *Performance:* Evaluated using RMSE, MAE, and R².
4.  **Deployment:** The interface is built with Streamlit for rapid prototyping and interactivity.

-----

## 📜 License & Acknowledgements

  * **Data Source:** Plot Ekologi 2ha 03 Pasoh (FRIM).
  * **Frameworks:** Streamlit, Scikit-Learn, Altair, Pandas.

<!-- end list -->

```
```
