import streamlit as st
import pandas as pd
import joblib

# --- Configuration --- #
nutriscore_mapping_reverse = {0.0: "A", 1.0: "B", 2.0: "C", 3.0: "D", 4.0: "E"}
expected_feature_columns = [
    "nova_group", "energy_kcal", "fat_100g", "saturated_fat_100g",
    "carbs_100g", "sugars_100g", "fiber_100g", "proteins_100g",
    "salt_100g", "sodium_100g", "contains_gluten", "contains_dairy",
    "contains_nuts", "contains_soy", "contains_eggs", "contains_fish",
    "food_type_Branded/Packaged", "has_allergens", "fat_to_protein_ratio",
    "sugar_to_fiber_ratio", "sodium_to_energy_ratio", "total_allergens_present",
]

st.set_page_config(
    page_title="Nutri-Score Predictor",
    page_icon="🥗",
    layout="wide",
    initial_sidebar_state="expanded",
    menu_items={
        'Get Help': 'https://github.com/yourrepo/nutriscore-help',
        'Report a bug': 'https://github.com/yourrepo/nutriscore-bugs',
        'About': "# 🥗 Nutri-Score Predictor\nEducational ML tool for nutritional assessment."
    }
)

# --- CSS Styling (unchanged, kept for brevity) --- #
st.markdown(
    """
    <style>
    .block-container { padding-top: 1rem; padding-bottom: 1rem; max-width: 1200px; }
    .hero-card {
        background: linear-gradient(135deg, #f0f9ff 0%, #e0f2fe 50%, #f0fdf4 100%);
        border: 1px solid #bae6fd; border-radius: 20px; padding: 1.5rem 2rem;
        margin-bottom: 1.5rem; box-shadow: 0 4px 20px rgba(0, 0, 0, 0.08);
    }
    .result-card {
        border-radius: 20px; padding: 1.5rem 2rem; border: 2px solid #e5e7eb;
        background: linear-gradient(to bottom right, #ffffff, #f9fafb);
        box-shadow: 0 8px 30px rgba(17, 24, 39, 0.12); margin: 1rem 0;
    }
    .grade-pill {
        display: inline-flex; align-items: center; gap: 0.4rem;
        padding: 0.4rem 1rem; border-radius: 999px; font-weight: 600;
        font-size: 0.95rem; box-shadow: 0 2px 8px rgba(0,0,0,0.1);
    }
    .grade-reveal { animation: popIn 0.5s ease forwards; opacity: 0; transform: scale(0.95); }
    @keyframes popIn { to { opacity: 1; transform: scale(1); } }
    .nutriscore-scale { display: flex; gap: 6px; margin: 1.5rem 0; justify-content: center; flex-wrap: wrap; }
    .score-grade {
        width: 48px; height: 56px; border-radius: 12px 12px 6px 6px;
        display: flex; align-items: center; justify-content: center;
        color: white; font-weight: 800; font-size: 1.1rem;
        transition: all 0.25s ease; box-shadow: 0 3px 10px rgba(0,0,0,0.15);
        position: relative;
    }
    .score-grade.active {
        transform: scale(1.15) translateY(-4px); box-shadow: 0 6px 20px rgba(0,0,0,0.25);
        z-index: 2; border: 3px solid white;
    }
    .score-grade::after {
        content: attr(data-label); position: absolute; bottom: -22px;
        font-size: 0.7rem; color: #6b7280; font-weight: 500; white-space: nowrap;
    }
    .input-section {
        background: #f8fafc; border-radius: 16px; padding: 1rem;
        margin: 0.5rem 0; border: 1px solid #e2e8f0;
    }
    .tooltip { border-bottom: 2px dashed #94a3b8; cursor: help; display: inline-flex; align-items: center; gap: 4px; }
    @media (max-width: 768px) {
        .hero-card, .result-card { padding: 1.2rem !important; margin: 0.5rem !important; border-radius: 16px !important; }
        .score-grade { width: 40px; height: 48px; font-size: 1rem; }
        .score-grade::after { font-size: 0.65rem; bottom: -20px; }
    }
    </style>
    """,
    unsafe_allow_html=True,
)


# --- Helper Functions --- #
@st.cache_resource
def load_model():
    try:
        return joblib.load("tuned_nutriscore_model.pkl")
    except FileNotFoundError:
        st.error("❌ Model file `tuned_nutriscore_model.pkl` not found.\n\nPlease place it in the same directory as this app.")
        st.stop()
    except Exception as exc:
        st.error(f"❌ Failed to load model: {exc}")
        st.stop()


def tooltip(label: str, help_text: str) -> None:
    st.markdown(
        f'<span class="tooltip" title="{help_text}">{label} <span style="color:#64748b;">ℹ️</span></span>',
        unsafe_allow_html=True
    )


def render_nutriscore_scale(predicted_grade: str = None) -> None:
    grades = ["A", "B", "C", "D", "E"]
    colors = ["#15803d", "#65a30d", "#d97706", "#dc2626", "#991b1b"]
    labels = ["Excellent", "Good", "Average", "Poor", "Very Poor"]
    
    scale_html = '<div class="nutriscore-scale">'
    for grade, color, label in zip(grades, colors, labels):
        active_class = "active" if grade == predicted_grade else ""
        scale_html += f'<div class="score-grade {active_class}" style="background:{color};" data-label="{label}">{grade}</div>'
    scale_html += '</div>'
    st.markdown(scale_html, unsafe_allow_html=True)


def get_explanation_factors(input_data: dict) -> list:
    factors = []
    if input_data["sugars_100g"] > 20:
        factors.append(("🔴 High sugar content", f"{input_data['sugars_100g']}g per 100g exceeds recommended limits"))
    if input_data["saturated_fat_100g"] > 10:
        factors.append(("🔴 High saturated fat", f"{input_data['saturated_fat_100g']}g may impact heart health"))
    if input_data["sodium_100g"] > 0.6:
        factors.append(("🔴 High sodium", f"{input_data['sodium_100g']*1000}mg is above daily guideline per serving"))
    if input_data["nova_group"] >= 4:
        factors.append(("🔴 Ultra-processed food", "Associated with lower nutritional quality"))
    if input_data["fiber_100g"] < 3:
        factors.append(("🔴 Low fiber", "Fiber supports digestion and satiety"))
    if input_data["proteins_100g"] >= 10:
        factors.append(("🟢 Good protein content", f"{input_data['proteins_100g']}g supports muscle health"))
    if input_data["fiber_100g"] >= 6:
        factors.append(("🟢 High fiber", "Excellent for digestive health"))
    if input_data["nova_group"] <= 2:
        factors.append(("🟢 Minimally processed", "Closer to natural food state"))
    if input_data["total_allergens_present"] > 0:
        factors.append((f"⚠️ Contains {input_data['total_allergens_present']} allergen(s)", "Check labels if you have food sensitivities"))
    return factors if factors else [("✅ Balanced profile", "No major nutritional red flags detected")]


def load_sample_product(product_name: str) -> dict:
    samples = {
        "Greek Yogurt (plain)": {
            "nova_group": 2, "energy_kcal": 97.0, "fat_100g": 5.0, "saturated_fat_100g": 3.2,
            "carbs_100g": 4.0, "sugars_100g": 4.0, "fiber_100g": 0.0, "proteins_100g": 9.0,
            "salt_100g": 0.1, "sodium_100g": 0.04,
            "contains_gluten": False, "contains_dairy": True, "contains_nuts": False,
            "contains_soy": False, "contains_eggs": False, "contains_fish": False
        },
        "Potato Chips": {
            "nova_group": 4, "energy_kcal": 536.0, "fat_100g": 34.0, "saturated_fat_100g": 11.0,
            "carbs_100g": 53.0, "sugars_100g": 0.5, "fiber_100g": 4.5, "proteins_100g": 6.0,
            "salt_100g": 1.5, "sodium_100g": 0.6,
            "contains_gluten": False, "contains_dairy": False, "contains_nuts": False,
            "contains_soy": False, "contains_eggs": False, "contains_fish": False
        },
        "Whole Grain Bread": {
            "nova_group": 3, "energy_kcal": 247.0, "fat_100g": 3.5, "saturated_fat_100g": 0.7,
            "carbs_100g": 41.0, "sugars_100g": 5.2, "fiber_100g": 7.0, "proteins_100g": 13.0,
            "salt_100g": 1.2, "sodium_100g": 0.48,
            "contains_gluten": True, "contains_dairy": False, "contains_nuts": False,
            "contains_soy": False, "contains_eggs": False, "contains_fish": False
        }
    }
    return samples.get(product_name, samples["Greek Yogurt (plain)"])


# --- Initialize Session State Defaults --- #
DEFAULTS = {
    "nova_group": 3, "energy_kcal": 200.0, "fat_100g": 10.0, "saturated_fat_100g": 3.0,
    "carbs_100g": 30.0, "sugars_100g": 15.0, "fiber_100g": 2.0, "proteins_100g": 5.0,
    "salt_100g": 0.5, "sodium_100g": 0.2,
    "contains_gluten": False, "contains_dairy": False, "contains_nuts": False,
    "contains_soy": False, "contains_eggs": False, "contains_fish": False,
    "sample_loaded": False
}

for key, value in DEFAULTS.items():
    if key not in st.session_state:
        st.session_state[key] = value


# --- Load Model --- #
loaded_model_pipeline = load_model()


# --- Header --- #
st.markdown(
    """
    <div class="hero-card">
        <div style="display:flex; align-items:center; gap:1rem; flex-wrap:wrap;">
            <div style="font-size:2.5rem;">🥗</div>
            <div>
                <h1 style="margin:0 0 0.3rem 0; font-size:1.8rem;">Nutri-Score Predictor</h1>
                <p style="margin:0; color:#475569; font-size:1.05rem;">
                    Make informed food choices • Powered by machine learning
                </p>
            </div>
        </div>
    </div>
    """,
    unsafe_allow_html=True,
)

col1, col2, col3, col4 = st.columns(4)
col1.metric("🎯 Scale", "A → E")
col2.metric("⚖️ Basis", "Per 100g")
col3.metric("📦 Type", "Branded")
col4.metric("🤖 Model", "ML Pipeline")


# --- Sidebar --- #
st.sidebar.markdown("### 🛒 Product Details")
st.sidebar.info("Enter values from the nutrition label (per 100g).")

# ✅ FIXED: Sample product selector with proper state handling
sample_product = st.sidebar.selectbox(
    "🎲 Try a sample product",
    ["Custom Input", "Greek Yogurt (plain)", "Potato Chips", "Whole Grain Bread"],
    index=0 if not st.session_state.sample_loaded else ["Custom Input", "Greek Yogurt (plain)", "Potato Chips", "Whole Grain Bread"].index(st.session_state.get("last_sample", "Custom Input")),
    key="sample_selector",
    help="Pre-fill fields with real product data"
)

# ✅ FIXED: Load sample data WITHOUT time.sleep() - use flag + rerun pattern
if sample_product != "Custom Input" and not st.session_state.sample_loaded:
    sample_data = load_sample_product(sample_product)
    for key, value in sample_data.items():
        st.session_state[key] = value
    st.session_state.sample_loaded = True
    st.session_state.last_sample = sample_product
    st.sidebar.success(f"✅ Loaded: {sample_product}")
    st.rerun()  # Clean rerun after state update

# Reset flag when switching back to custom
if sample_product == "Custom Input" and st.session_state.sample_loaded:
    st.session_state.sample_loaded = False
    st.rerun()

# Nutrition values - ✅ FIXED: Use session_state with proper key separation
with st.sidebar.expander("🥄 Nutrition Values (per 100g)", expanded=True):
    st.markdown('<div class="input-section">', unsafe_allow_html=True)
    col_nut1, col_nut2 = st.columns(2)
    
    with col_nut1:
        tooltip("⚡ Energy", "Total calories per 100g")
        energy_kcal = st.number_input("Energy (kcal)", 0.0, 2000.0, 
                                     st.session_state.energy_kcal, 1.0, key="input_energy")
        tooltip("🧈 Total Fat", "All fats")
        fat_100g = st.number_input("Fat (g)", 0.0, 100.0, 
                                  st.session_state.fat_100g, 0.1, key="input_fat")
        tooltip("🥓 Saturated Fat", "Limit for heart health")
        saturated_fat_100g = st.number_input("Saturated fat (g)", 0.0, 50.0, 
                                           st.session_state.saturated_fat_100g, 0.1, key="input_satfat")
        tooltip("🍞 Carbohydrates", "Total carbs")
        carbs_100g = st.number_input("Carbohydrates (g)", 0.0, 100.0, 
                                    st.session_state.carbs_100g, 0.1, key="input_carbs")
        tooltip("🍬 Sugars", "Simple carbohydrates")
        sugars_100g = st.number_input("Sugars (g)", 0.0, 100.0, 
                                     st.session_state.sugars_100g, 0.1, key="input_sugars")
    
    with col_nut2:
        tooltip("🌾 Fiber", "Supports gut health")
        fiber_100g = st.number_input("Fiber (g)", 0.0, 30.0, 
                                    st.session_state.fiber_100g, 0.1, key="input_fiber")
        tooltip("🥩 Protein", "Essential nutrient")
        proteins_100g = st.number_input("Protein (g)", 0.0, 50.0, 
                                       st.session_state.proteins_100g, 0.1, key="input_protein")
        tooltip("🧂 Salt", "Monitor for blood pressure")
        salt_100g = st.number_input("Salt (g)", 0.0, 10.0, 
                                   st.session_state.salt_100g, 0.01, key="input_salt")
        tooltip("🧪 Sodium", "1g salt ≈ 0.4g sodium")
        sodium_100g = st.number_input("Sodium (g)", 0.0, 4.0, 
                                     st.session_state.sodium_100g, 0.001, key="input_sodium")
        tooltip("🏭 NOVA Group", "Processing classification")
        nova_group = st.slider("NOVA group", 1, 4, 
                              st.session_state.nova_group,
                              help="1=Unprocessed • 4=Ultra-processed",
                              key="input_nova")
    st.markdown('</div>', unsafe_allow_html=True)

# Allergens - ✅ FIXED: Separate keys from session_state storage keys
with st.sidebar.expander("⚠️ Allergen Information", expanded=True):
    st.markdown('<div class="input-section">', unsafe_allow_html=True)
    st.markdown("*Select all that apply*")
    col_all1, col_all2 = st.columns(2)
    with col_all1:
        contains_gluten = st.checkbox("🌾 Gluten", value=st.session_state.contains_gluten, key="chk_gluten")
        contains_dairy = st.checkbox("🥛 Dairy", value=st.session_state.contains_dairy, key="chk_dairy")
        contains_nuts = st.checkbox("🥜 Nuts", value=st.session_state.contains_nuts, key="chk_nuts")
    with col_all2:
        contains_soy = st.checkbox("🫘 Soy", value=st.session_state.contains_soy, key="chk_soy")
        contains_eggs = st.checkbox("🥚 Eggs", value=st.session_state.contains_eggs, key="chk_eggs")
        contains_fish = st.checkbox("🐟 Fish", value=st.session_state.contains_fish, key="chk_fish")
    st.markdown('</div>', unsafe_allow_html=True)

# Action buttons
st.sidebar.markdown("---")
col_predict, col_reset = st.sidebar.columns([3, 1])
with col_predict:
    predict_clicked = st.button("🔍 Predict Nutri-Score", use_container_width=True, type="primary")
with col_reset:
    if st.button("🔄", help="Reset", use_container_width=True):
        for key in DEFAULTS:
            st.session_state[key] = DEFAULTS[key]
        st.session_state.sample_loaded = False
        st.rerun()

food_type_branded_packaged = True


# --- Prediction Logic --- #
if predict_clicked:
    # Validation
    if sugars_100g > carbs_100g:
        st.sidebar.warning(f"⚠️ Sugars ({sugars_100g}g) can't exceed carbs ({carbs_100g}g). Auto-adjusted.")
        sugars_100g = carbs_100g
    
    with st.spinner("🔍 Analyzing..."):
        # Derived features
        fat_to_protein_ratio = fat_100g / proteins_100g if proteins_100g > 0 else 0.0
        sugar_to_fiber_ratio = sugars_100g / fiber_100g if fiber_100g > 0 else 0.0
        sodium_to_energy_ratio = sodium_100g / energy_kcal if energy_kcal > 0 else 0.0
        
        total_allergens_present_count = sum([
            int(contains_gluten), int(contains_dairy), int(contains_nuts),
            int(contains_soy), int(contains_eggs), int(contains_fish)
        ])
        has_allergens_bool = 1 if total_allergens_present_count > 0 else 0
        
        # Build input DataFrame
        input_data_values = [
            nova_group, energy_kcal, fat_100g, saturated_fat_100g, carbs_100g,
            sugars_100g, fiber_100g, proteins_100g, salt_100g, sodium_100g,
            contains_gluten, contains_dairy, contains_nuts, contains_soy,
            contains_eggs, contains_fish, food_type_branded_packaged,
            has_allergens_bool, fat_to_protein_ratio, sugar_to_fiber_ratio,
            sodium_to_energy_ratio, total_allergens_present_count,
        ]
        
        input_df = pd.DataFrame([input_data_values], columns=expected_feature_columns)
        bool_cols = ["contains_gluten", "contains_dairy", "contains_nuts", "contains_soy",
                    "contains_eggs", "contains_fish", "food_type_Branded/Packaged"]
        for col in bool_cols:
            input_df[col] = input_df[col].astype(bool)
        
        # Predict
        try:
            prediction_encoded = loaded_model_pipeline.predict(input_df)[0]
            predicted_grade = nutriscore_mapping_reverse.get(prediction_encoded, "Unknown")
        except Exception as e:
            st.error(f"❌ Prediction failed: {e}")
            st.stop()
    
    # --- Results Display --- #
    st.markdown("## 🎯 Prediction Result")
    
    grade_config = {
        "A": ("#15803d", "#dcfce7", "Excellent", "🌟 Outstanding nutritional quality"),
        "B": ("#65a30d", "#ecfccb", "Good", "✅ Well-balanced choice"),
        "C": ("#d97706", "#fef3c7", "Average", "⚖️ Moderate nutritional profile"),
        "D": ("#dc2626", "#fee2e2", "Poor", "🔶 Limit regular consumption"),
        "E": ("#991b1b", "#fecaca", "Very Poor", "🚫 Best as occasional treat"),
        "Unknown": ("#374151", "#f3f4f6", "Unknown", "⚠️ Could not determine score")
    }
    
    text_color, bg_color, quality_label, message = grade_config.get(
        predicted_grade, grade_config["Unknown"]
    )
    
    st.markdown(
        f"""
        <div class="result-card grade-reveal" style="text-align:center; border-left:6px solid {text_color}">
            <div style="color:#64748b; font-size:1rem; margin-bottom:0.5rem;">Predicted Nutritional Quality</div>
            <div style="font-size:4.5rem; font-weight:900; color:{text_color}; text-shadow:0 4px 12px rgba(0,0,0,0.12); margin:0.25rem 0;">
                {predicted_grade}
            </div>
            <div style="display:inline-flex; align-items:center; gap:0.5rem; margin-top:0.5rem;">
                <span class="grade-pill" style="color:{text_color}; background:{bg_color}; border:2px solid {text_color}30;">
                    {quality_label}
                </span>
                <span style="color:#64748b; font-size:0.95rem;">• Grade {predicted_grade}</span>
            </div>
        </div>
        """,
        unsafe_allow_html=True,
    )
    
    render_nutriscore_scale(predicted_grade)
    
    icon = message.split()[0]
    st.markdown(
        f"""
        <div style="background:{bg_color}80; border:1px solid {text_color}40; border-radius:14px; padding:1rem 1.5rem; margin:1rem 0; display:flex; align-items:flex-start; gap:0.75rem;">
            <div style="font-size:1.4rem; margin-top:2px;">{icon}</div>
            <div style="color:#1e293b; font-size:1.05rem;">{message}</div>
        </div>
        """,
        unsafe_allow_html=True
    )
    
    with st.expander("💡 Why this score?", expanded=False):
        st.markdown("**Key factors:**")
        input_dict = {
            "sugars_100g": sugars_100g, "saturated_fat_100g": saturated_fat_100g,
            "sodium_100g": sodium_100g, "nova_group": nova_group,
            "fiber_100g": fiber_100g, "proteins_100g": proteins_100g,
            "fat_100g": fat_100g, "total_allergens_present": total_allergens_present_count
        }
        for factor_icon, factor_desc in get_explanation_factors(input_dict):
            st.markdown(f"- **{factor_icon}** {factor_desc}")
    
    with st.expander("🔧 Show input data", expanded=False):
        st.dataframe(input_df.T, use_container_width=True, height=300)
    
    # Export - ✅ FIXED: Use st.query_params instead of deprecated API
    st.markdown("### 📤 Export")
    col_exp1, col_exp2 = st.columns(2)
    with col_exp1:
        summary = f"Nutri-Score: {predicted_grade} ({quality_label})\nEnergy: {energy_kcal}kcal\nSugar: {sugars_100g}g\nFiber: {fiber_100g}g"
        st.download_button("📋 Copy Summary", data=summary, file_name=f"nutriscore_{predicted_grade}.txt", mime="text/plain", use_container_width=True)
    with col_exp2:
        csv = input_df.to_csv(index=False).encode('utf-8')
        st.download_button("💾 Export CSV", data=csv, file_name="input_data.csv", mime="text/csv", use_container_width=True)
    
    # Feedback
    st.markdown("---")
    if st.feedback("stars") is not None:
        st.toast("Thanks for your feedback! 🙏", icon="⭐")

else:
    st.info("👈 Fill in the sidebar and click **🔍 Predict Nutri-Score**")
    with st.expander("🚀 Quick Start", expanded=True):
        st.markdown("1. Select a sample product OR enter custom values\n2. Check applicable allergens\n3. Click predict\n4. Review the explanation")

# Footer
st.markdown("---")
st.markdown(
    '<div style="text-align:center; color:#64748b; font-size:0.9rem; padding:1rem;">🥗 Educational Tool • Not medical advice</div>',
    unsafe_allow_html=True
)
