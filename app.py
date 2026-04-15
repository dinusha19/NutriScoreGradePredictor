import streamlit as st
import pandas as pd
import joblib

# --- Configuration --- #
nutriscore_mapping_reverse = {0.0: "A", 1.0: "B", 2.0: "C", 3.0: "D", 4.0: "E"}
expected_feature_columns = [
    "nova_group",
    "energy_kcal",
    "fat_100g",
    "saturated_fat_100g",
    "carbs_100g",
    "sugars_100g",
    "fiber_100g",
    "proteins_100g",
    "salt_100g",
    "sodium_100g",
    "contains_gluten",
    "contains_dairy",
    "contains_nuts",
    "contains_soy",
    "contains_eggs",
    "contains_fish",
    "food_type_Branded/Packaged",
    "has_allergens",
    "fat_to_protein_ratio",
    "sugar_to_fiber_ratio",
    "sodium_to_energy_ratio",
    "total_allergens_present",
]

st.set_page_config(page_title="Nutri-Score Predictor", page_icon="🍎", layout="wide")

st.markdown(
    """
    <style>
    .block-container {
        padding-top: 2rem;
        padding-bottom: 1.5rem;
    }
    .hero-card {
        background: linear-gradient(135deg, #f8fafc 0%, #eef2ff 100%);
        border: 1px solid #dbeafe;
        border-radius: 16px;
        padding: 1.1rem 1.2rem;
        margin-bottom: 1rem;
    }
    .result-card {
        border-radius: 16px;
        padding: 1rem 1.2rem;
        border: 1px solid #e5e7eb;
        background: #ffffff;
        box-shadow: 0 2px 12px rgba(17, 24, 39, 0.05);
    }
    .grade-pill {
        display: inline-block;
        padding: 0.3rem 0.75rem;
        border-radius: 999px;
        font-weight: 700;
        font-size: 0.9rem;
        letter-spacing: 0.02em;
    }
    </style>
    """,
    unsafe_allow_html=True,
)


@st.cache_resource
def load_model():
    try:
        return joblib.load("tuned_nutriscore_model.pkl")
    except FileNotFoundError:
        st.error("Model file `tuned_nutriscore_model.pkl` was not found in this folder.")
        st.stop()
    except Exception as exc:
        st.error(f"Model could not be loaded: {exc}")
        st.stop()


loaded_model_pipeline = load_model()

st.markdown(
    """
    <div class="hero-card">
        <h1 style="margin:0 0 0.4rem 0;">🍎 Nutri-Score Predictor</h1>
        <p style="margin:0; font-size:1.02rem;">
            Enter nutrition values per 100g and allergen details in the sidebar.
            The app predicts the Nutri-Score grade (A to E) using your trained model.
        </p>
    </div>
    """,
    unsafe_allow_html=True,
)

col1, col2, col3 = st.columns(3)
col1.metric("Scale", "A to E")
col2.metric("Model Input", "Per 100g")
col3.metric("Food Type", "Branded/Packaged")

st.sidebar.header("Product Inputs")
with st.sidebar.expander("Nutrition values (per 100g)", expanded=True):
    nova_group = st.slider(
        "NOVA group (processing level)",
        1,
        4,
        3,
        help=(
            "1 = Unprocessed/minimally processed, 2 = Processed ingredients, "
            "3 = Processed foods, 4 = Ultra-processed."
        ),
    )
    energy_kcal = st.number_input("Energy (kcal)", 0.0, 900.0, 200.0, 1.0)
    fat_100g = st.number_input("Fat (g)", 0.0, 100.0, 10.0, 0.1)
    saturated_fat_100g = st.number_input("Saturated fat (g)", 0.0, 100.0, 3.0, 0.1)
    carbs_100g = st.number_input("Carbohydrates (g)", 0.0, 100.0, 30.0, 0.1)
    sugars_100g = st.number_input("Sugars (g)", 0.0, 100.0, 15.0, 0.1)
    fiber_100g = st.number_input("Fiber (g)", 0.0, 10.0, 2.0, 0.1)
    proteins_100g = st.number_input("Protein (g)", 0.0, 100.0, 5.0, 0.1)
    salt_100g = st.number_input("Salt (g)", 0.0, 10.0, 0.5, 0.01)
    sodium_100g = st.number_input("Sodium (g)", 0.0, 4.0, 0.2, 0.001)

with st.sidebar.expander("Allergen information", expanded=True):
    contains_gluten = st.checkbox("Contains gluten")
    contains_dairy = st.checkbox("Contains dairy")
    contains_nuts = st.checkbox("Contains nuts")
    contains_soy = st.checkbox("Contains soy")
    contains_eggs = st.checkbox("Contains eggs")
    contains_fish = st.checkbox("Contains fish")

predict_clicked = st.sidebar.button("Predict Nutri-Score", use_container_width=True, type="primary")
food_type_branded_packaged = True

if predict_clicked:
    fat_to_protein_ratio = fat_100g / proteins_100g if proteins_100g else 0.0
    sugar_to_fiber_ratio = sugars_100g / fiber_100g if fiber_100g else 0.0
    sodium_to_energy_ratio = sodium_100g / energy_kcal if energy_kcal else 0.0

    total_allergens_present_count = (
        int(contains_gluten)
        + int(contains_dairy)
        + int(contains_nuts)
        + int(contains_soy)
        + int(contains_eggs)
        + int(contains_fish)
    )
    has_allergens_bool = 1 if total_allergens_present_count > 0 else 0

    input_data_values = [
        nova_group,
        energy_kcal,
        fat_100g,
        saturated_fat_100g,
        carbs_100g,
        sugars_100g,
        fiber_100g,
        proteins_100g,
        salt_100g,
        sodium_100g,
        contains_gluten,
        contains_dairy,
        contains_nuts,
        contains_soy,
        contains_eggs,
        contains_fish,
        food_type_branded_packaged,
        has_allergens_bool,
        fat_to_protein_ratio,
        sugar_to_fiber_ratio,
        sodium_to_energy_ratio,
        total_allergens_present_count,
    ]

    input_df = pd.DataFrame([input_data_values], columns=expected_feature_columns)
    for col in [
        "contains_gluten",
        "contains_dairy",
        "contains_nuts",
        "contains_soy",
        "contains_eggs",
        "contains_fish",
        "food_type_Branded/Packaged",
    ]:
        input_df[col] = input_df[col].astype(bool)

    prediction_encoded = loaded_model_pipeline.predict(input_df)[0]
    predicted_grade = nutriscore_mapping_reverse.get(prediction_encoded, "Unknown")

    grade_colors = {
        "A": ("#15803d", "#dcfce7"),
        "B": ("#65a30d", "#ecfccb"),
        "C": ("#d97706", "#fef3c7"),
        "D": ("#dc2626", "#fee2e2"),
        "E": ("#991b1b", "#fee2e2"),
        "Unknown": ("#374151", "#f3f4f6"),
    }
    text_color, bg_color = grade_colors[predicted_grade]

    st.markdown("## Prediction Result")
    st.markdown(
        f"""
        <div class="result-card">
            <div style="font-size:1rem; color:#374151; margin-bottom:0.5rem;">Predicted Nutri-Score</div>
            <div style="font-size:2.1rem; font-weight:700; color:{text_color}; margin-bottom:0.5rem;">
                Grade {predicted_grade}
            </div>
            <span class="grade-pill" style="color:{text_color}; background:{bg_color};">
                Nutritional quality indicator
            </span>
        </div>
        """,
        unsafe_allow_html=True,
    )

    if predicted_grade in ["A", "B"]:
        st.success("Great nutritional profile based on the provided values.")
        st.balloons()
    elif predicted_grade == "C":
        st.info("Moderate nutritional profile. Small improvements may help.")
    else:
        st.warning("Lower nutritional profile. Consider healthier alternatives.")

    with st.expander("Show model input used for this prediction"):
        st.dataframe(input_df, use_container_width=True)
else:
    st.info("Fill in the sidebar and click **Predict Nutri-Score** to generate a result.")

st.markdown("### Run locally")
st.code(
    "pip install -r requirements.txt\nstreamlit run app.py",
    language="bash",
)
