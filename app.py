import streamlit as st
import pandas as pd
import joblib

# --- Configuration --- #
nutriscore_mapping_reverse = {0.0: "A", 1.0: "B", 2.0: "C", 3.0: "D", 4.0: "E"}

st.set_page_config(page_title="Nutri-Score Predictor", page_icon="🥗", layout="wide")

# --- Custom Styling --- #
st.markdown("""
<style>
body {
    background-color: #f9fafb;
}
.hero {
    padding: 1.5rem;
    border-radius: 20px;
    background: linear-gradient(135deg, #4f46e5, #22c55e);
    color: white;
    text-align: center;
    margin-bottom: 1.5rem;
}
.card {
    background: white;
    padding: 1rem;
    border-radius: 15px;
    box-shadow: 0 4px 12px rgba(0,0,0,0.08);
    margin-bottom: 1rem;
}
.metric {
    text-align: center;
    padding: 0.8rem;
    border-radius: 12px;
    background: #eef2ff;
}
.result {
    text-align: center;
    padding: 1.5rem;
    border-radius: 20px;
    background: white;
    box-shadow: 0 6px 20px rgba(0,0,0,0.1);
}
</style>
""", unsafe_allow_html=True)


@st.cache_resource
def load_model():
    return joblib.load("tuned_nutriscore_model.pkl")

model = load_model()

# --- Header --- #
st.markdown("""
<div class="hero">
    <h1>🥗 Nutri-Score Predictor</h1>
    <p>Estimate food quality instantly using nutritional values</p>
</div>
""", unsafe_allow_html=True)

# --- Layout --- #
left, right = st.columns([1, 2])

# --- Sidebar Inputs --- #
st.sidebar.title("🔧 Input Parameters")

nova_group = st.sidebar.selectbox("Processing Level (NOVA)", [1,2,3,4])
energy_kcal = st.sidebar.slider("Energy (kcal)", 0, 900, 200)
fat = st.sidebar.slider("Fat (g)", 0.0, 100.0, 10.0)
sat_fat = st.sidebar.slider("Saturated Fat (g)", 0.0, 100.0, 3.0)
carbs = st.sidebar.slider("Carbs (g)", 0.0, 100.0, 30.0)
sugar = st.sidebar.slider("Sugar (g)", 0.0, 100.0, 15.0)
fiber = st.sidebar.slider("Fiber (g)", 0.0, 10.0, 2.0)
protein = st.sidebar.slider("Protein (g)", 0.0, 100.0, 5.0)
sodium = st.sidebar.slider("Sodium (g)", 0.0, 4.0, 0.2)

st.sidebar.markdown("### Allergens")
gluten = st.sidebar.checkbox("Gluten")
dairy = st.sidebar.checkbox("Dairy")
nuts = st.sidebar.checkbox("Nuts")
soy = st.sidebar.checkbox("Soy")
eggs = st.sidebar.checkbox("Eggs")
fish = st.sidebar.checkbox("Fish")

predict = st.sidebar.button("🚀 Predict")

# --- Left Info Panel --- #
with left:
    st.markdown('<div class="card">', unsafe_allow_html=True)
    st.subheader("ℹ️ About")
    st.write("This app predicts Nutri-Score using a trained ML model.")
    st.markdown("- A → Best\n- E → Worst")
    st.markdown('</div>', unsafe_allow_html=True)

# --- Prediction --- #
if predict:
    fat_ratio = fat / protein if protein else 0
    sugar_ratio = sugar / fiber if fiber else 0
    sodium_ratio = sodium / energy_kcal if energy_kcal else 0

    allergen_count = sum([gluten, dairy, nuts, soy, eggs, fish])

    df = pd.DataFrame([[ 
        nova_group, energy_kcal, fat, sat_fat, carbs, sugar,
        fiber, protein, 0, sodium,
        gluten, dairy, nuts, soy, eggs, fish,
        True, int(allergen_count>0),
        fat_ratio, sugar_ratio, sodium_ratio, allergen_count
    ]])

    pred = model.predict(df)[0]
    grade = nutriscore_mapping_reverse.get(pred, "?")

    colors = {
        "A":"#22c55e","B":"#84cc16","C":"#f59e0b","D":"#ef4444","E":"#b91c1c"
    }

    with right:
        st.markdown('<div class="result">', unsafe_allow_html=True)
        st.markdown(f"<h2 style='color:{colors.get(grade)}'>Grade {grade}</h2>", unsafe_allow_html=True)
        st.progress((5 - list(colors.keys()).index(grade)) / 5)
        st.markdown('</div>', unsafe_allow_html=True)

        st.markdown("### 🔍 Input Summary")
        st.dataframe(df)

else:
    with right:
        st.info("Enter values and click Predict 🚀")

# --- Footer --- #
st.markdown("---")
st.caption("Built with ❤️ using Streamlit")
