import streamlit as st
import pandas as pd
import numpy as np
import joblib

# ==========================================
# Page Configuration
# ==========================================

st.set_page_config(
    page_title="FIFA Overall Predictor",
    page_icon="⚽",
    layout="wide",
    initial_sidebar_state="collapsed"
)

# ==========================================
# Custom CSS
# ==========================================

st.markdown("""
<style>

#MainMenu{
    visibility:hidden;
}

footer{
    visibility:hidden;
}

header{
    visibility:hidden;
}

.block-container{
    padding-top:2rem;
    padding-bottom:2rem;
}

/* Hero Card */

.hero{

    background:linear-gradient(135deg,#11998e,#38ef7d);

    padding:35px;

    border-radius:18px;

    color:white;

    box-shadow:0px 8px 25px rgba(0,0,0,.25);

}

.hero h1{

    font-size:42px;

    margin-bottom:10px;

}

.hero p{

    font-size:18px;

    opacity:.95;

}

/* Metric Cards */

.metric-card{

    background:#1f2937;

    padding:22px;

    border-radius:16px;

    text-align:center;

    border:1px solid #333;

    transition:.3s;

}

.metric-card:hover{

    transform:translateY(-6px);

    box-shadow:0px 12px 20px rgba(0,0,0,.25);

}

/* Prediction Card */

.prediction-card{

    background:#111827;

    padding:25px;

    border-radius:18px;

    border:1px solid #2d3748;

}

</style>
""", unsafe_allow_html=True)

# ==========================================
# Load Model
# ==========================================

@st.cache_resource
def load_model():
    return joblib.load("player_overall_model.pkl")

model = load_model()

# ==========================================
# Hero Section
# ==========================================

st.markdown("""

<div class="hero">

<h1>⚽ FIFA 22 Overall Rating Prediction</h1>

<p>

Predict the Overall Rating of FIFA 22 players using a Machine Learning model
trained on thousands of professional football players.

</p>

</div>

""", unsafe_allow_html=True)

st.write("")


# ====================================================
# Player Information
# ====================================================

st.markdown("## Player Information")

left, right = st.columns(2, gap="large")

with left:

    preferred_foot = st.selectbox(
        "Preferred Foot",
        ["Left", "Right"]
    )

    position = st.selectbox(
        "Position",
        [
            "GK","CB","LB","RB","LWB","RWB",
            "CDM","LDM","RDM",
            "CM","LCM","RCM",
            "CAM","LAM","RAM",
            "LM","RM","LW","RW",
            "ST","CF","LF","RF","LS","RS"
        ]
    )

    work_rate = st.selectbox(
        "Work Rate",
        [
            "High/High",
            "High/Medium",
            "High/Low",
            "Medium/High",
            "Medium/Medium",
            "Medium/Low",
            "Low/High",
            "Low/Medium",
            "Low/Low"
        ]
    )

    height = st.slider(
        "Height (cm)",
        140,
        220,
        180
    )

    weight = st.slider(
        "Weight (kg)",
        40,
        130,
        75
    )

with right:

    potential = st.slider(
        "Potential",
        40,
        99,
        80
    )

    special = st.slider(
        "Special",
        1000,
        2500,
        1800
    )

    value = st.number_input(
        "Market Value (€)",
        min_value=0.0,
        value=1000000.0,
        step=100000.0,
        format="%.0f"
    )

    wage = st.number_input(
        "Weekly Wage (€)",
        min_value=0.0,
        value=10000.0,
        step=1000.0,
        format="%.0f"
    )

    release_clause = st.number_input(
        "Release Clause (€)",
        min_value=0.0,
        value=5000000.0,
        step=100000.0,
        format="%.0f"
    )

st.divider()

# ====================================================
# Player Attributes
# ====================================================

st.markdown("## Player Attributes")

tech, physical, reputation = st.columns(3, gap="large")

# ====================================================
# Technical Attributes
# ====================================================

with tech:

    st.markdown("### Technical")

    crossing = st.slider(
        "Crossing",
        1,
        99,
        60
    )

    finishing = st.slider(
        "Finishing",
        1,
        99,
        60
    )

    skill_moves = st.slider(
        "Skill Moves",
        1,
        5,
        3
    )

    weak_foot = st.slider(
        "Weak Foot",
        1,
        5,
        3
    )

# ====================================================
# Physical Attributes
# ====================================================

with physical:

    st.markdown("### Physical")

    sprint_speed = st.slider(
        "Sprint Speed",
        1,
        99,
        70
    )

    agility = st.slider(
        "Agility",
        1,
        99,
        70
    )

    stamina = st.slider(
        "Stamina",
        1,
        99,
        70
    )

    strength = st.slider(
        "Strength",
        1,
        99,
        70
    )

# ====================================================
# Reputation
# ====================================================

with reputation:

    st.markdown("### Reputation")

    international_rep = st.slider(
        "International Reputation",
        1,
        5,
        3
    )

    st.metric(
        "Current Potential",
        potential
    )

    st.metric(
        "Preferred Position",
        position
    )

    st.metric(
        "Preferred Foot",
        preferred_foot
    )

st.divider()

# ====================================================
# Predict Button
# ====================================================

predict = st.button(
    "Predict Overall Rating",
    use_container_width=True,
    type="primary"
)


# =====================================================
# Prediction
# =====================================================

if predict:

    with st.spinner("🤖 AI Model is analyzing player data..."):

        # ------------------------------
        # Work Rate
        # ------------------------------

        attacking_wr, defensive_wr = work_rate.split("/")

        attacking_wr = attacking_wr.strip()
        defensive_wr = defensive_wr.strip()

        # ------------------------------
        # BMI
        # ------------------------------

        height_meter = height / 100

        bmi = weight / (height_meter ** 2)

        if bmi < 18.5:
            fitness = "Underweight"

        elif bmi < 25:
            fitness = "Fit"

        elif bmi < 30:
            fitness = "Overweight"

        else:
            fitness = "Obese"

        # ------------------------------
        # Same preprocessing used
        # during training
        # ------------------------------

        value_log = np.log1p(value)
        wage_log = np.log1p(wage)
        release_clause_log = np.log1p(release_clause)
        special_log = np.log1p(special)

        # ------------------------------
        # DataFrame
        # ------------------------------

        input_df = pd.DataFrame({

            "Preferred_Foot":[preferred_foot],
            "Attacking_WorkRate":[attacking_wr],
            "Defensive_WorkRate":[defensive_wr],
            "Fitness_level":[fitness],
            "Position":[position],
            "Potential":[potential],
            "Value":[value_log],
            "Wage":[wage_log],
            "Special":[special_log],
            "International_Reputation":[international_rep],
            "Weak_Foot":[weak_foot],
            "Skill_Moves":[skill_moves],
            "Height":[height],
            "Weight":[weight],
            "Release_Clause":[release_clause_log],
            "Crossing":[crossing],
            "Finishing":[finishing],
            "SprintSpeed":[sprint_speed],
            "Agility":[agility],
            "Stamina":[stamina],
            "Strength":[strength]

        })

        prediction = model.predict(input_df)[0]

    st.divider()

    if prediction < 60:

        level = "Low Player"

    elif prediction < 70:

        level = "Average Player"

    elif prediction < 80:

        level = "Good Player"

    elif prediction < 90:

        level = "Very Good Player"

    else:

        level = "World Class ⭐"

        st.balloons()


    st.markdown(f"""

    <div class="fifa-card">

    <div style="display:flex;justify-content:space-between;align-items:flex-start;">

    <div>

    <div class="rating">{prediction:.0f}</div>

    <div class="position">{position}</div>

    </div>

    <div style="text-align:right;">

    <div class="small-info">

    🦶 {preferred_foot}

    </div>

    <div class="small-info">

    ⭐ Skill Moves : {skill_moves}

    </div>

    <div class="small-info">

    ⭐ Weak Foot : {weak_foot}

    </div>

    </div>

    </div>

    <hr style="opacity:.3;">

    <table class="stats-table">

    <tr>

    <td>PAC</td>
    <td>{sprint_speed}</td>

    <td>SHO</td>
    <td>{finishing}</td>

    </tr>

    <tr>

    <td>PAS</td>
    <td>{crossing}</td>

    <td>DRI</td>
    <td>{agility}</td>

    </tr>

    <tr>

    <td>PHY</td>
    <td>{strength}</td>

    <td>POT</td>
    <td>{potential}</td>

    </tr>

    </table>

    <div class="level">

    🏆 {level}

    </div>

    </div>

    """,unsafe_allow_html=True)

    # ==========================================
    # Statistics
    # ==========================================

st.markdown("""
<style>

.center-metric{
    background:#1f2937;
    padding:20px;
    border-radius:15px;
    text-align:center;
    border:1px solid #374151;
}

.center-title{
    font-size:18px;
    color:#9CA3AF;
    margin-bottom:10px;
}

.center-value{
    font-size:42px;
    font-weight:bold;
    color:white;
}

</style>
""", unsafe_allow_html=True)

c1, c2, c3 = st.columns(3)

with c1:
    st.markdown(f"""
    <div class="center-metric">
        <div class="center-title">BMI</div>
        <div class="center-value">{bmi:.1f}</div>
    </div>
    """, unsafe_allow_html=True)

with c2:
    st.markdown(f"""
    <div class="center-metric">
        <div class="center-title">Fitness</div>
        <div class="center-value">{fitness}</div>
    </div>
    """, unsafe_allow_html=True)

with c3:
    st.markdown(f"""
    <div class="center-metric">
        <div class="center-title">Potential</div>
        <div class="center-value">{potential}</div>
    </div>
    """, unsafe_allow_html=True)
    st.divider()

st.markdown("""

<div style="text-align:center;color:gray;">

Developed by <b>Kareem Salah</b>

</div>

""", unsafe_allow_html=True)

st.markdown("""
<style>

.fifa-card{

background:linear-gradient(145deg,#5528b8,#8b5cf6,#2dd4bf);

border-radius:28px;

padding:28px;

color:white;

box-shadow:0 20px 40px rgba(0,0,0,.35);

border:3px solid rgba(255,255,255,.15);

position:relative;

overflow:hidden;

}

.fifa-card::before{

content:"";

position:absolute;

width:220px;

height:220px;

background:rgba(255,255,255,.08);

border-radius:50%;

top:-60px;

right:-70px;

}

.fifa-card::after{

content:"";

position:absolute;

width:170px;

height:170px;

background:rgba(255,255,255,.06);

border-radius:50%;

bottom:-60px;

left:-50px;

}

.rating{

font-size:72px;

font-weight:900;

line-height:1;

}

.position{

font-size:28px;

font-weight:bold;

opacity:.9;

}

.player-name{

font-size:32px;

font-weight:800;

text-align:center;

margin:20px 0;

}

.stats-table{

width:100%;

font-size:24px;

font-weight:bold;

}

.stats-table td{

padding:6px;

}

.small-info{

font-size:18px;

opacity:.9;

}

.level{

text-align:center;

font-size:22px;

font-weight:bold;

margin-top:20px;

padding:10px;

border-radius:12px;

background:rgba(255,255,255,.15);

backdrop-filter:blur(10px);

}

</style>

""",unsafe_allow_html=True)