import streamlit as st
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import skfuzzy as fuzz
import folium
from streamlit_folium import st_folium
st.set_page_config(page_title="Global Humanitarian Donation Priority System", layout="wide")
# ------------------------------------------------------------# PREDEFINED CRISIS DATA FOR FOUR REGIONS# ------------------------------------------------------------
regions_data = {
    "Palestine (2024)": {
        "population": 98,
        "aid_access": 5,
        "impact": 100,
        "location": [31.5, 34.4]
    },
    "Sudan (Conflict)": {
        "population": 76,
        "aid_access": 20,
        "impact": 85,
        "location": [15.5, 32.5]
    },
    "Congo (Humanitarian Crisis)": {
        "population": 69,
        "aid_access": 25,
        "impact": 80,
        "location": [0.8, 25.5]
    },
    "Punjab Flood (India)": {
        "population": 45,
        "aid_access": 50,
        "impact": 40,
        "location": [31.1, 75.0]
    }
}

# ------------------------------------------------------------# FUZZY MEMBERSHIP DEFINITIONS (CLEAN)# ------------------------------------------------------------

pop = np.arange(0, 101, 1)
aid = np.arange(0, 101, 1)
impact = np.arange(0, 101, 1)

pop_low = fuzz.trimf(pop, [0, 0, 30])
pop_med = fuzz.trimf(pop, [20, 50, 80])
pop_high = fuzz.trimf(pop, [60, 100, 100])

aid_low = fuzz.trimf(aid, [0, 0, 40])
aid_med = fuzz.trimf(aid, [30, 50, 70])
aid_high = fuzz.trimf(aid, [60, 100, 100])

impact_low = fuzz.trimf(impact, [0, 0, 40])
impact_med = fuzz.trimf(impact, [30, 60, 80])
impact_high = fuzz.trimf(impact, [70, 100, 100])

# ------------------------------------------------------------# FUZZY INFERENCE FUNCTION# ------------------------------------------------------------

def compute_priority(population, aid_level, impact_level):

    p_low = fuzz.interp_membership(pop, pop_low, population)
    p_med = fuzz.interp_membership(pop, pop_med, population)
    p_high = fuzz.interp_membership(pop, pop_high, population)

    a_low = fuzz.interp_membership(aid, aid_low, aid_level)
    a_med = fuzz.interp_membership(aid, aid_med, aid_level)
    a_high = fuzz.interp_membership(aid, aid_high, aid_level)

    i_low = fuzz.interp_membership(impact, impact_low, impact_level)
    i_med = fuzz.interp_membership(impact, impact_med, impact_level)
    i_high = fuzz.interp_membership(impact, impact_high, impact_level)

    priority_low = np.fmax(p_low, np.fmax(a_high, i_low))
    priority_med = np.fmax(p_med, a_med)
    priority_high = np.fmax(p_high, np.fmax(a_low, i_high))

    priority = np.arange(0, 101, 1)
    low_mem = fuzz.trimf(priority, [0, 0, 40])
    med_mem = fuzz.trimf(priority, [30, 50, 70])
    high_mem = fuzz.trimf(priority, [60, 100, 100])

    aggregated = np.fmax(priority_low * low_mem,
                         np.fmax(priority_med * med_mem,
                                 priority_high * high_mem))

    priority_score = fuzz.defuzz(priority, aggregated, 'centroid')
    return priority_score

# ------------------------------------------------------------# STREAMLIT INTERFACE# ------------------------------------------------------------

st.title("Global Humanitarian Donation Priority System")
st.write("AI and Fuzzy Logic Based Crisis Assessment and Donation Allocation Model.")

st.sidebar.title("Select Region for Analysis")
selected_region = st.sidebar.selectbox("Choose Region", list(regions_data.keys()))

data = regions_data[selected_region]

priority_value = compute_priority(
    data["population"],
    data["aid_access"],
    data["impact"])

# ------------------------------------------------------------# DISPLAY REGION DETAILS AND SCORE# ------------------------------------------------------------

col1, col2 = st.columns(2)

with col1:
    st.subheader("Region Crisis Indicators")
    st.write(f"Population Density: {data['population']}%")
    st.write(f"Aid Access Level: {data['aid_access']}%")
    st.write(f"Crisis Impact Level: {data['impact']}%")

with col2:
    st.subheader("Predicted Donation Urgency Score")
    st.metric("Urgency Score", f"{priority_value:.2f} / 100")

# ------------------------------------------------------------# MAP VISUALIZATION# ------------------------------------------------------------

m = folium.Map(location=data["location"], zoom_start=6)
folium.Marker(data["location"], popup=selected_region).add_to(m)
st_folium(m, width=700, height=450)

# ------------------------------------------------------------# DONATION SUGGESTION CHART# ------------------------------------------------------------

st.subheader("Suggested Donation Allocation (AI Estimated)")

allocated = {
    "Food and Water": priority_value * 0.4,
    "Medical Aid": priority_value * 0.3,
    "Shelter Support": priority_value * 0.2,
    "Rehabilitation Programs": priority_value * 0.1
}

df = pd.DataFrame.from_dict(allocated, orient="index", columns=["Allocation Score"])

st.bar_chart(df)

st.success("Analysis Completed.")