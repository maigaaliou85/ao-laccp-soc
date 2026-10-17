import streamlit as st
from PIL import Image
import os
import importlib.util
import sys

# ==============================
# DASHBOARD TITLE PAGE
# ==============================

# ------------------------------
# Logo Path
# ------------------------------
logo_path = os.path.join(os.path.dirname(__file__), "marymount_logo.png")

# ------------------------------
# Display Logo Centered
# ------------------------------
if os.path.exists(logo_path):
    try:
        img = Image.open(logo_path)
        # Center the image using st.image inside a column
        col1, col2, col3 = st.columns([1,2,1])  # middle column is wider
        with col2:
            st.image(img, width=440, use_container_width=False)
    except Exception as e:
        st.warning(f"Cannot open logo: {e}")
else:
    st.warning(f"Logo not found: {logo_path}")

# ------------------------------
# HTML Header Centered and Colored
# ------------------------------
st.markdown("""
<div style="text-align: center; padding: 10px; background-color: #f5f5f5;">
    <h1 style="color: #10069F;">AO-LACCP Dashboard</h1>
</div>
""", unsafe_allow_html=True)

# ------------------------------
# Function to Show Title Page (Centered)
# ------------------------------
def show_title_page():
    st.markdown("""
    <div style="text-align: center;">
        <h4 style="color: #2c3e50;">AI-Orchestrated Latency-Aware Cryptographic Control Plane</h4>
    
<p style="text-align: center; font-family: Arial; font-size: 11pt; color: #8B0000; text-transform: capitalize;">
    <strong><b>TOPIC: AI-Orchestrated Cyber Resilience: An Adaptive Framework For Cyber Risk Management In  Banking And Fintech Ecosystems</b></strong>
</p>
<p style="text-align: center; font-family: Arial; font-size: 14pt; color: #8B0000; text-transform: capitalize;">
    <strong><b>AO-LACCP Artifact Simulation – Doctoral Research Demo</b></strong>
</p>
  

  <p style="text-align: center; font-family: Arial; font-size: 14px; color: #2c3e50; line-height:1.5;">
    By<br>
    <strong>Aliou Maiga</strong><br><br>
    MICHELLE LIU, PhD, Committee Chair<br>
    NATHAN GREEN, PhD, Committee Member<br>
    SEYEDAMIN POURIYEH, PhD, External Reader<br>
   ANNE MAGRO, PhD, Dean<br><br>
    College of Business, Innovation, Leadership, and Technology<br><br>
    A Dissertation Proposal Presented in Partial Fulfillment of the Requirements for the Degree Doctor of Science<br><br>
    Marymount University<br>
    May 2027
</p>
    </div>
    """, unsafe_allow_html=True)

# ==============================
# Sidebar Module Selection
# ==============================
st.sidebar.header("Select Module to Run")
module_options = ["None", "Full Code 1: AO-LACCP Full Framework", 
                  "Full Code 2: AO-LACCP Multi-Agent", 
                  "Case Study Scenario: Threat Scenarios"]

if "module_choice" not in st.session_state:
    st.session_state.module_choice = "None"

st.session_state.module_choice = st.sidebar.radio(
    "Choose Module",
    module_options,
    index=module_options.index(st.session_state.module_choice)
)

if st.session_state.module_choice != "None":
    if st.sidebar.button("Close Module"):
        st.session_state.module_choice = "None"
        st.experimental_rerun = True

# ==============================
# Function to Import and Run a Module
# ==============================
def run_module(folder_name):
    module_path = os.path.join(folder_name, "main.py")
    if os.path.exists(module_path):
        spec = importlib.util.spec_from_file_location("module_main", module_path)
        module = importlib.util.module_from_spec(spec)
        sys.modules["module_main"] = module
        spec.loader.exec_module(module)
    else:
        st.error(f"Module not found: {module_path}")

# ==============================
# Run Title Page or Module
# ==============================
if st.session_state.module_choice == "None":
    show_title_page()
else:
    if st.session_state.module_choice.startswith("Full Code 1"):
        run_module("FullCode1")
    elif st.session_state.module_choice.startswith("Full Code 2"):
        run_module("FullCode2")
    elif st.session_state.module_choice.startswith("Case Study Scenario"):
        run_module("CaseStudyScenario")
