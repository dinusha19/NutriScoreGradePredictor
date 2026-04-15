import streamlit as st
import pandas as pd
import joblib
# import time # Removed this import as time.sleep will be removed

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

# Page config with enhanced settings
st.set_page_config(
    page_title="Nutri-Score Predictor",
    page_icon="🥗",
    layout="wide",
    initial_sidebar_state="expanded",
    menu_items={
        'Get Help': 'https://github.com/yourrepo/nutriscore-help',
        'Report a bug': 'https://github.com/yourrepo/nutriscore-bugs',
        'About': "# 🥗 Nutri-Score Predictor\nPowered by machine learning to help you make healthier food choices."
    }
)

# --- Enhanced CSS Styling --- #
st.markdown(
    """
    <style>
    /* Base layout improvements */
    .block-container {
        padding-top: 1rem;
        padding-bottom: 1rem;
        max-width: 1200px;
    }
    
    /* Hero card with gradient and shadow */
    .hero-card {
        background: linear-gradient(135deg, #f0f9ff 0%, #e0f2fe 50%, #f0fdf4 100%);
        border: 1px solid #bae6fd;
        border-radius: 20px;
        padding: 1.5rem 2rem;
        margin-bottom: 1.5rem;
        box-shadow: 0 4px 20px rgba(0, 0, 0, 0.08);
        transition: transform 0.2s ease;
    }
    .hero-card:hover {
        transform: translateY(-2px);
    }
    
    /* Result card with animated border */
    .result-card {
        border-radius: 20px;
        padding: 1.5rem 2rem;
        border: 2px solid #e5e7eb;
        background: linear-gradient(to bottom right, #ffffff, #f9fafb);
        box-shadow: 0 8px 30px rgba(17, 24, 39, 0.12);
        transition: all 0.3s ease;
        margin: 1rem 0;
    }
    .result-card:hover {
        box-shadow: 0 12px 40px rgba(17, 24, 39, 0.18);
        border-color: #cbd5e1;
    }
    
    /* Grade pill badge */
    .grade-pill {
        display: inline-flex;
        align-items: center;
        gap: 0.4rem;
        padding: 0.4rem 1rem;
        border-radius: 999px;
        font-weight: 600;
        font-size: 0.95rem;
        letter-spacing: 0.01em;
        box-shadow: 0 2px 8px rgba(0,0,0,0.1);
    }
    
    /* Animated grade reveal */
    .grade-reveal {
        animation: popIn 0.5s cubic-bezier(0.175, 0.885, 0.32, 1.275) forwards;
        opacity: 0;
        transform: scale(0.95);
    }
    @keyframes popIn {
        to {
            opacity: 1;
            transform: scale(1);
        }
    }
    
    /* Nutri-Score scale */
    .nutriscore-scale {
        display: flex;
        gap: 6px;
        margin: 1.5rem 0;
        justify-content: center;
        flex-wrap: wrap;
    }
    .score-grade {
        width: 48px;
        height: 56px;
        border-radius: 12px 12px 6px 6px;
        display: flex;
        align-items: center;
        justify-content: center;
        color: white;
        font-weight: 800;
        font-size: 1.1rem;
        transition: all 0.25s ease;
        box-shadow: 0 3px 10px rgba(0,0,0,0.15);
        position: relative;
    }
    .score-grade.active {
        transform: scale(1.15) translateY(-4px);
        box-shadow: 0 6px 20px rgba(0,0,0,0.25);
        z-index: 2;
        border: 3px solid white;
    }
    .score-grade::after {
        content: attr(data-label);
        position: absolute;
        bottom: -22px;
        font-size: 0.7rem;
        color: #6b7280;
        font-weight: 500;
        white-space: nowrap;
    }
    
    /* Input section styling */
    .input-section {
        background: #f8fafc;
        border-radius: 16px;
        padding: 1rem;
        margin: 0.5rem 0;
        border: 1px solid #e2e8f0;
    }
    
    /* Tooltip style */
    .tooltip {
        border-bottom: 2px dashed #94a3b8;
        cursor: help;
        position: relative;
        display: inline-flex;
        align-items: center;
        gap: 4px;
    }
    
    /* Mobile responsiveness */
    @media (max-width: 768px) {
        .hero-card, .result-card {
            padding: 1.2rem !important;
            margin: 0.5rem !important;
            border-radius: 16px !important;
        }
        .score-grade {
            width: 40px;
            height: 48px;
            font-size: 1rem;
        }
        .score-grade::after {
            font-size: 0.65rem;
            bottom: -20px;
        }
        .grade-pill {
            font-size: 0.9rem !important;
            padding: 0.35rem 0.9rem !important;
        }
    }
    
    /* High contrast mode */
    @media (prefers-contrast: high) {
        .result-card {
            border: 3px solid #000 !important;
        }
        .score-grade {
            border: 2px solid #000 !important;
        }
    }
    
    /* Smooth scrolling */
    html {
        scroll-behavior: smooth;
    }
    
    /* Button hover effects */
    .stButton > button {
        transition: all 0.2s ease;
        border-radius: 12px !important;
        font-weight: 600 !important;
    }
    .stButton > button:hover {
        transform: translateY(-1px);
        box-shadow: 0 4px 12px rgba(0,0,0,0.15) !important;
    }
    </style>
    """,
    unsafe_allow_html=True,
)


# --- Helper Functions --- #
@st.cache_resource
def load_model():
    """Load the trained model with error handling."""
    try:
        return joblib.load("tuned_nutriscore_model.pkl")
    except FileNotFoundError:
        st.error("❌ Model file `tuned_nutriscore_model.pkl` not found.\n\nPlease ensure the model file is in the same directory as this app.")
        st.stop()
    except Exception as exc:
        st.error(f"❌ Failed to load model: {exc}")
        st.stop()


def tooltip(label: str, help_text: str) -> None:
    """Render a label with tooltip help text."""
    st.markdown(
        f"""
        <span class="tooltip" title="{help_text}">
            {label} <span style="color:#64748b; font-size:0.9rem;">ℹ️</span>
        </span>
        """,
        unsafe_allow_html=True
    )


def render_nutriscore_scale(predicted_grade: str = None) -> None:
    """Render an interactive visual Nutri-Score scale."""
    grades = ["A", "B", "C", "D", "E"]
    colors = ["#15803d", "#65a30d", "#d97706", "#dc2626", "#991b1b"]
    labels = ["Excellent", "Good", "Average", "Poor", "Very Poor"]
    
    scale_html = '<div class="nutriscore-scale">'
    for grade, color, label in zip(grades, colors, labels):
        active_class = "active" if grade == predicted_grade else ""
        scale_html += f'''
        <div class="score-grade {active_class}" 
             style="background:{color};" 
             data-label="{label}">
            {grade}
        </div>'''
    scale_html += '</div>'
    st.markdown(scale_html, unsafe_allow_html=True)


def get_explanation_factors(input_data: dict) -> list:
    """Generate human-readable factors influencing the Nutri-Score."""
    factors = []
    
    # Negative factors (lower score)
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
    
    # Positive factors (higher score)
    if input_data["proteins_100g"] >= 10:
        factors.append(("🟢 Good protein content", f"{input_data['proteins_100g']}g supports muscle health"))
    if input_data["fiber_100g"] >= 6:
        factors.append(("🟢 High fiber", "Excellent for digestive health"))
    if input_data["nova_group"] <= 2:
        factors.append(("🟢 Minimally processed", "Closer to natural food state"))
    if input_data["fat_100g"] < 5 and input_data["sugars_100g"] < 10:
        factors.append(("🟢 Low fat & sugar", "Supports balanced energy intake"))
    
    # Allergen note
    if input_data["total_allergens_present"] > 0:
        factors.append((f"⚠️ Contains {input_data['total_allergens_present']} allergen(s)", 
                       "Check labels if you have food sensitivities"))
    
    return factors if factors else [("✅ Balanced profile", "No major nutritional red flags detected")]


def load_sample_product(product_name: str) -> dict:
    """Return pre-filled values for demo products."""
    samples = {
        "Greek Yogurt (plain)": {
            "nova_group": 2, "energy_kcal": 97, "fat_100g": 5.0, "saturated_fat_100g": 3.2,
            "carbs_100g": 4.0, "sugars_100g": 4.0, "fiber_100g": 0.0, "proteins_100g": 9.0,
            "salt_100g": 0.1, "sodium_100g": 0.04,
            "contains_gluten": False, "contains_dairy": True, "contains_nuts": False,
            "contains_soy": False, "contains_eggs": False, "contains_fish": False
        },
        "Potato Chips": {
            "nova_group": 4, "energy_kcal": 536, "fat_100g": 34.0, "saturated_fat_100g": 11.0,
            "carbs_100g": 53.0, "sugars_100g": 0.5, "fiber_100g": 4.5, "proteins_100g": 6.0,
            "salt_100g": 1.5, "sodium_100g": 0.6,
            "contains_gluten": False, "contains_dairy": False, "contains_nuts": False,
            "contains_soy": False, "contains_eggs": False, "contains_fish": False
        },
        "Whole Grain Bread": {
            "nova_group": 3, "energy_kcal": 247, "fat_100g": 3.5, "saturated_fat_100g": 0.7,
            "carbs_100g": 41.0, "sugars_100g": 5.2, "fiber_100g": 7.0, "proteins_100g": 13.0,
            "salt_100g": 1.2, "sodium_100g": 0.48,
            "contains_gluten": True, "contains_dairy": False, "contains_nuts": False,
            "contains_soy": False, "contains_eggs": False, "contains_fish": False
        }
    }
    return samples.get(product_name, samples["Greek Yogurt (plain)"])


# --- Load Model --- #
loaded_model_pipeline = load_model()


# --- Header Section --- #
st.markdown(
    """
    <div class="hero-card">
        <div style="display:flex; align-items:center; gap:1rem; flex-wrap:wrap;">
            <div style="font-size:2.5rem;">🥗</div>
            <div>
                <h1 style="margin:0 0 0.3rem 0; font-size:1.8rem;">Nutri-Score Predictor</h1>
                <p style="margin:0; color:#475569; font-size:1.05rem;">
                    Make informed food choices • Powered by machine learning • Results in seconds
                </p>
            </div>
        </div>
    </div>
    """,
    unsafe_allow_html=True,
)

# Info metrics
col1, col2, col3, col4 = st.columns(4)
col1.metric("🎯 Scale", "A (Best) → E")
col2.metric("⚖️ Basis", "Per 100g")
col3.metric("📦 Type", "Branded/Packaged")
col4.metric("🤖 Model", "Tuned ML Pipeline")


# --- Sidebar: Product Inputs --- #
st.sidebar.markdown("### 🛒 Product Details")
st.sidebar.info("Enter values exactly as shown on the nutrition label (per 100g).")

# Sample product selector
sample_product = st.sidebar.selectbox(
    "🍚 Try a sample product",
    ["Custom Input", "Greek Yogurt (plain)", "Potato Chips", "Whole Grain Bread"],
    index=0,
    help="Pre-fill fields with real product data for testing"
)

if sample_product != "Custom Input":
    sample_data = load_sample_product(sample_product)
    for key, value in sample_data.items():
        st.session_state[key] = value
    st.sidebar.success(f"✅ Loaded: {sample_product}")
    # Removed time.sleep(1.5)
    st.rerun() # Added st.rerun() to update inputs with new session state

# Nutrition values section
with st.sidebar.expander("🥄 Nutrition Values (per 100g)", expanded=True):
    st.markdown('<div class="input-section">', unsafe_allow_html=True)
    
    # Two-column layout for compact inputs
    col_nut1, col_nut2 = st.columns(2)
    
    with col_nut1:
        tooltip("⚡ Energy", "Total calories per 100g of product")
        energy_kcal = st.number_input("Energy (kcal)", 0.0, 2000.0, 
                                     st.session_state.get("energy_kcal", 200.0), 1.0, key="energy_kcal")
        
        tooltip("🧈 Total Fat", "All fats including saturated, unsaturated, and trans fats")
        fat_100g = st.number_input("Fat (g)", 0.0, 100.0, 
                                  st.session_state.get("fat_100g", 10.0), 0.1, key="fat_100g")
        
        tooltip("🥓 Saturated Fat", "Fats solid at room temperature; limit for heart health")
        saturated_fat_100g = st.number_input("Saturated fat (g)", 0.0, 50.0, 
                                           st.session_state.get("saturated_fat_100g", 3.0), 0.1, key="saturated_fat_100g")
        
        tooltip("🍞 Carbohydrates", "Total carbs including sugars and fiber")
        carbs_100g = st.number_input("Carbohydrates (g)", 0.0, 100.0, 
                                    st.session_state.get("carbs_100g", 30.0), 0.1, key="carbs_100g")
        
        tooltip("🍬 Sugars", "Simple carbohydrates; includes added and natural sugars")
        sugars_100g = st.number_input("Sugars (g)", 0.0, 100.0, 
                                     st.session_state.get("sugars_100g", 15.0), 0.1, key="sugars_100g")
    
    with col_nut2:
        tooltip("🌾 Fiber", "Indigestible carbs that support gut health")
        fiber_100g = st.number_input("Fiber (g)", 0.0, 30.0, 
                                    st.session_state.get("fiber_100g", 2.0), 0.1, key="fiber_100g")
        
        tooltip("🥩 Protein", "Essential for muscle repair and satiety")
        proteins_100g = st.number_input("Protein (g)", 0.0, 50.0, 
                                       st.session_state.get("proteins_100g", 5.0), 0.1, key="proteins_100g")
        
        tooltip("🧂 Salt", "Sodium chloride; monitor for blood pressure")
        salt_100g = st.number_input("Salt (g)", 0.0, 10.0, 
                                   st.session_state.get("salt_100g", 0.5), 0.01, key="salt_100g")
        
        tooltip("🧪 Sodium", "Mineral component of salt; 1g salt ≈ 0.4g sodium")
        sodium_100g = st.number_input("Sodium (g)", 0.0, 4.0, 
                                     st.session_state.get("sodium_100g", 0.2), 0.001, key="sodium_100g")
        
        tooltip("🏭 NOVA Group", "Food processing classification system")
        nova_group = st.slider(
            "NOVA group", 1, 4, 
            st.session_state.get("nova_group", 3),
            help="1=Unprocessed • 2=Processed ingredients • 3=Processed foods • 4=Ultra-processed",
            key="nova_group"
        )
    
    st.markdown('</div>', unsafe_allow_html=True)

# Allergen section
with st.sidebar.expander("⚠️ Allergen Information", expanded=True):
    st.markdown('<div class="input-section">', unsafe_allow_html=True)
    st.markdown("*Select all that apply to this product*")
    
    col_all1, col_all2 = st.columns(2)
    with col_all1:
        contains_gluten = st.checkbox("🌾 Gluten", value=st.session_state.get("contains_gluten", False), key="contains_gluten")
        contains_dairy = st.checkbox("🥛 Dairy", value=st.session_state.get("contains_dairy", False), key="contains_dairy")
        contains_nuts = st.checkbox("🥜 Nuts", value=st.session_state.get("contains_nuts", False), key="contains_nuts")
    with col_all2:
        contains_soy = st.checkbox("🫘 Soy", value=st.session_state.get("contains_soy", False), key="contains_soy")
        contains_eggs = st.checkbox("🥚 Eggs", value=st.session_state.get("contains_eggs", False), key="contains_eggs")
        contains_fish = st.checkbox("🐟 Fish", value=st.session_state.get("contains_fish", False), key="contains_fish")
    st.markdown('</div>', unsafe_allow_html=True)

# Action buttons
st.sidebar.markdown("---")
col_predict, col_reset = st.sidebar.columns([3, 1])

with col_predict:
    predict_clicked = st.button("🔍 Predict Nutri-Score", use_container_width=True, type="primary")

with col_reset:
    if st.button("🔄", help="Reset all fields", use_container_width=True):
        for key in list(st.session_state.keys()):
            if key not in ["loaded_model_pipeline"]:  # Preserve model
                del st.session_state[key]
        st.rerun()

# Fixed food type for this app version
food_type_branded_packaged = True


# --- Main Prediction Logic --- #
if predict_clicked:
    # Input validation & auto-correction
    if sugars_100g > carbs_100g:
        st.sidebar.warning(f"⚠️ Sugars ({sugars_100g}g) can't exceed carbs ({carbs_100g}g). Auto-adjusted.")
        sugars_100g = carbs_100g
    
    if sodium_100g > salt_100g * 0.45:  # Sodium is ~40% of salt by weight
        st.sidebar.warning("⚠️ Sodium value seems high relative to salt. Double-check units.")
    
    with st.spinner("🔍 Analyzing nutritional profile..."):
        # time.sleep(0.4)  # Removed time.sleep
        
        # Calculate derived features
        fat_to_protein_ratio = fat_100g / proteins_100g if proteins_100g > 0 else 0.0
        sugar_to_fiber_ratio = sugars_100g / fiber_100g if fiber_100g > 0 else 0.0
        sodium_to_energy_ratio = sodium_100g / energy_kcal if energy_kcal > 0 else 0.0
        
        total_allergens_present_count = sum([
            int(contains_gluten), int(contains_dairy), int(contains_nuts),
            int(contains_soy), int(contains_eggs), int(contains_fish)
        ])
        has_allergens_bool = 1 if total_allergens_present_count > 0 else 0
        
        # Prepare input DataFrame
        input_data_values = [
            nova_group, energy_kcal, fat_100g, saturated_fat_100g, carbs_100g,
            sugars_100g, fiber_100g, proteins_100g, salt_100g, sodium_100g,
            contains_gluten, contains_dairy, contains_nuts, contains_soy,
            contains_eggs, contains_fish, food_type_branded_packaged,
            has_allergens_bool, fat_to_protein_ratio, sugar_to_fiber_ratio,
            sodium_to_energy_ratio, total_allergens_present_count,
        ]
        
        input_df = pd.DataFrame([input_data_values], columns=expected_feature_columns)
        
        # Ensure boolean columns are properly typed
        bool_cols = [
            "contains_gluten", "contains_dairy", "contains_nuts", "contains_soy",
            "contains_eggs", "contains_fish", "food_type_Branded/Packaged"
        ]
        for col in bool_cols:
            input_df[col] = input_df[col].astype(bool)
        
        # Make prediction
        try:
            prediction_encoded = loaded_model_pipeline.predict(input_df)[0]
            predicted_grade = nutriscore_mapping_reverse.get(prediction_encoded, "Unknown")
        except Exception as e:
            st.error(f"❌ Prediction failed: {e}")
            st.stop()
    
    # --- Display Results --- #
    st.markdown("## 🎯 Prediction Result")
    
    # Grade configuration
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
    
    # Animated result card
    st.markdown(
        f"""
        <div class="result-card grade-reveal" style="text-align:center; border-left:6px solid {text_color}">
            <div style="color:#64748b; font-size:1rem; margin-bottom:0.5rem;">
                Predicted Nutritional Quality
            </div>
            <div style="
                font-size:4.5rem; 
                font-weight:900; 
                color:{text_color};
                text-shadow:0 4px 12px rgba(0,0,0,0.12);
                line-height:1;
                margin:0.25rem 0;">
                {predicted_grade}
            </div>
            <div style="
                display:inline-flex; 
                align-items:center; 
                gap:0.5rem;
                margin-top:0.5rem;">
                <span class="grade-pill" style="
                    color:{text_color}; 
                    background:{bg_color};
                    border:2px solid {text_color}30;">
                    {quality_label}
                </span>
                <span style="color:#64748b; font-size:0.95rem;">• Grade {predicted_grade}</span>
            </div>
        </div>
        """,
        unsafe_allow_html=True,
    )
    
    # Visual Nutri-Score scale
    render_nutriscore_scale(predicted_grade)
    
    # Contextual message with icon
    icon = message.split()[0]
    st.markdown(
        f"""
        <div style="
            background:{bg_color}80; 
            border:1px solid {text_color}40; 
            border-radius:14px; 
            padding:1rem 1.5rem; 
            margin:1rem 0;
            display:flex; 
            align-items:flex-start; 
            gap:0.75rem;">
            <div style="font-size:1.4rem; margin-top:2px;">{icon}</div>
            <div style="color:#1e293b; font-size:1.05rem;">{message}</div>
        </div>
        """,
        unsafe_allow_html=True
    )
    
    # Explanation section
    with st.expander("💡 Why this score? Click to see factors", expanded=False):
        st.markdown("**Key factors influencing this prediction:**")
        
        # Prepare input dict for explanation
        input_dict = {
            "sugars_100g": sugars_100g, "saturated_fat_100g": saturated_fat_100g,
            "sodium_100g": sodium_100g, "nova_group": nova_group,
            "fiber_100g": fiber_100g, "proteins_100g": proteins_100g,
            "fat_100g": fat_100g, "total_allergens_present": total_allergens_present_count
        }
        
        factors = get_explanation_factors(input_dict)
        for factor_icon, factor_desc in factors:
            st.markdown(f"- **{factor_icon}** {factor_desc}")
        
        st.markdown(
            """
            <div style="
                background:#f1f5f9; 
                border-radius:12px; 
                padding:1rem; 
                margin-top:1rem; 
                font-size:0.9rem; 
                color:#475569;">
                <strong>📝 Note:</strong> This prediction is based on nutritional values per 100g. 
                Always consider portion sizes, your dietary needs, and consult a healthcare 
                professional for personalized advice.
            </div>
            """,
            unsafe_allow_html=True
        )
    
    # Show model input (debug/advanced users)
    with st.expander("🔧 Show technical input data", expanded=False):
        st.dataframe(input_df.T, use_container_width=True, height=300)
        st.caption("Features used for model prediction (transposed for readability)")
    
    # Export options
    st.markdown("### 📤 Share or Save")
    col_exp1, col_exp2, col_exp3 = st.columns(3)
    
    with col_exp1:
        # Create summary text for copying
        summary_text = (
            f"Nutri-Score Prediction\n"
            f"Grade: {predicted_grade} ({quality_label})\n"
            f"Product: Custom input\n"
            f"Key values: {energy_kcal}kcal, {sugars_100g}g sugar, {fiber_100g}g fiber\n"
            f"Generated via Nutri-Score Predictor App"
        )
        st.download_button(
            "📋 Copy Summary",
            data=summary_text,
            file_name=f"nutriscore_{predicted_grade}_summary.txt",
            mime="text/plain",
            use_container_width=True
        )
    
    with col_exp2:
        # CSV export of input
        csv = input_df.to_csv(index=False).encode('utf-8')
        st.download_button(
            "💾 Export Input CSV",
            data=csv,
            file_name="product_input_data.csv",
            mime="text/csv",
            use_container_width=True
        )
    
    with col_exp3:
        if st.button("🔗 Copy Share Link", use_container_width=True):
            st.code(st.experimental_get_query_params(), language="text")
            st.toast("🔗 Link format copied! (Add params manually)", icon="✅")
    
    # Feedback collection (Streamlit >=1.30)
    st.markdown("---")
    st.markdown("### 💬 Was this helpful?")
    feedback = st.feedback("stars")
    if feedback is not None:
        st.toast(f"Thanks for your {feedback+1}-star feedback! 🙏", icon="⭐")
        
else:
    # Default state message
    st.info(
        "👈 **Get started**: Fill in the nutrition details in the sidebar, "
        "then click **🔍 Predict Nutri-Score** to see the result.\n\n"
        "💡 *Try loading a sample product first to see how it works!*"
    )
    
    # Quick start guide
    with st.expander("🚀 Quick Start Guide", expanded=True):
        st.markdown(
            """
            1. **Choose a sample product** from the dropdown in the sidebar (optional)
            2. **Enter nutrition values** exactly as shown on the product label (per 100g)
            3. **Select allergens** that apply to the product
            4. Click **🔍 Predict Nutri-Score** to see the grade
            5. Review the explanation to understand the factors
            
            📌 **Pro Tip**: Values are case-sensitive for the model - double-check units!
            """
        )
        st.image(
            "https://www.nutriscore.fr/sites/default/files/styles/landscape_medium/public/2021-06/nutriscore-en.png",
            caption="Nutri-Score reference scale (Source: nutricore.fr)",
            use_container_width=True
        )

# --- Footer --- #
st.markdown("---")
st.markdown(
    """
    <div style="text-align:center; color:#64748b; font-size:0.9rem; padding:1rem;">
        <strong>🥗 Nutri-Score Predictor</strong> • Educational tool only • 
        Not a substitute for professional dietary advice • 
        <a href="#" style="color:#3b82f6; text-decoration:none;">Privacy Policy</a> • 
        <a href="#" style="color:#3b82f6; text-decoration:none;">Model Info</a>
    </div>
    """,
    unsafe_allow_html=True
)
