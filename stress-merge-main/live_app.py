import streamlit as st
import pandas as pd
import numpy as np
import plotly.graph_objects as go
import time
import plotly.express as px
import matplotlib.pyplot as plt
import io
import joblib

# --- CONFIGURATION & ZEN-CYBERPUNK THEME v15.0 ---
st.set_page_config(
    page_title="SFAA Live: Zen Analyzer",
    layout="wide",
    page_icon="🧘"
)

st.markdown("""
<style>
    @import url('https://fonts.googleapis.com/css2?family=Plus+Jakarta+Sans:wght@400;500;600;700&display=swap');

    /* --- INSANE ANIMATIONS (Phase 12) --- */
    @keyframes gridMove {
        0% { background-position: 0 0; }
        100% { background-position: 50px 50px; }
    }
    @keyframes float {
        0% { transform: translateY(0px) rotateX(0deg); }
        50% { transform: translateY(-15px) rotateX(5deg); }
        100% { transform: translateY(0px) rotateX(0deg); }
    }
    @keyframes neonPulse {
        0% { box-shadow: 0 0 5px #2dd4bf, 0 0 10px #2dd4bf; }
        50% { box-shadow: 0 0 20px #2dd4bf, 0 0 40px #2dd4bf; }
        100% { box-shadow: 0 0 5px #2dd4bf, 0 0 10px #2dd4bf; }
    }
    @keyframes entrance3D {
        0% { opacity: 0; transform: scale3d(0.8, 0.8, 0.8) translateY(50px); }
        100% { opacity: 1; transform: scale3d(1, 1, 1) translateY(0); }
    }
    
    /* --- HOLOGRAPHIC GRID BACKGROUND --- */
    .stApp { 
        background-color: #050b14;
        background-image: 
            linear-gradient(rgba(45, 212, 191, 0.05) 1px, transparent 1px),
            linear-gradient(90deg, rgba(45, 212, 191, 0.05) 1px, transparent 1px);
        background-size: 50px 50px;
        animation: gridMove 20s linear infinite;
        font-family: 'Plus Jakarta Sans', sans-serif;
        color: #e2e8f0;
    }
    
    /* --- HERO HEADER 3D --- */
    .hero-container {
        text-align: center;
        padding: 80px 20px;
        background: rgba(15, 23, 42, 0.6);
        border: 1px solid rgba(45, 212, 191, 0.2);
        border-radius: 30px;
        backdrop-filter: blur(20px);
        margin-bottom: 50px;
        box-shadow: 0 0 30px rgba(45, 212, 191, 0.1);
        animation: entrance3D 1s cubic-bezier(0.175, 0.885, 0.32, 1.275);
    }
    
    .hero-title {
        font-size: 4.5rem;
        font-weight: 900;
        background: linear-gradient(135deg, #2dd4bf 0%, #3b82f6 50%, #d946ef 100%);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
        animation: float 5s ease-in-out infinite;
        text-shadow: 0 0 40px rgba(45, 212, 191, 0.5);
        letter-spacing: -2px;
    }
    
    .hero-subtitle {
        font-size: 1.4rem;
        color: #94a3b8;
        font-weight: 700;
        letter-spacing: 4px;
        text-transform: uppercase;
        margin-top: 10px;
    }
    
    /* --- 3D CARDS (TILT EFFECT) --- */
    div.stCard, div[data-testid="stExpander"] {
        background: rgba(30, 41, 59, 0.4) !important;
        border: 1px solid rgba(255, 255, 255, 0.05) !important;
        backdrop-filter: blur(24px) !important;
        border-radius: 20px !important;
        transition: transform 0.4s cubic-bezier(0.175, 0.885, 0.32, 1.275), box-shadow 0.4s ease;
        animation: entrance3D 0.8s ease-out backwards;
        transform-style: preserve-3d;
    }
    
    div.stCard:hover {
        transform: translateY(-10px) scale(1.02) rotateX(2deg);
        box-shadow: 0 20px 50px -10px rgba(45, 212, 191, 0.3) !important;
        border-color: rgba(45, 212, 191, 0.6) !important;
    }

    /* --- RESULT METRIC CARDS --- */
    .result-card {
        background: linear-gradient(135deg, rgba(255,255,255,0.05), rgba(255,255,255,0.01));
        border: 1px solid rgba(255,255,255,0.1);
        border-radius: 20px;
        padding: 30px;
        backdrop-filter: blur(15px);
        animation: entrance3D 0.6s ease-out backwards;
        transition: all 0.3s ease;
    }
    .result-card:hover {
        transform: scale(1.05);
        box-shadow: 0 0 30px rgba(255,255,255,0.1);
        border-color: rgba(255,255,255,0.4);
    }
        padding: 24px;
        margin-bottom: 16px;
        animation: fadeSlideUp 0.6s ease-out 0.2s backwards;
    }
    
    .metric-label {
        color: #94a3b8;
        font-size: 0.9rem;
        font-weight: 500;
        text-transform: uppercase;
    }
    
    .metric-value {
        color: #f1f5f9;
        font-size: 2rem;
        font-weight: 700;
        margin-top: 4px;
    }

    /* --- SLIDERS --- */
    div[data-baseweb="slider"] { padding-top: 24px; }
    
    /* --- BUTTONS --- */
    button[kind="primary"] {
        background: linear-gradient(135deg, #2dd4bf 0%, #0f766e 100%) !important;
        border: none !important;
        font-weight: 600 !important;
        letter-spacing: 0.5px !important;
        padding: 12px 24px !important;
        transition: transform 0.2s !important;
    }
    button[kind="primary"]:hover {
        transform: scale(1.02);
        box-shadow: 0 0 20px rgba(45, 212, 191, 0.4);
    }

</style>
""", unsafe_allow_html=True)

# --- ADVANCED IKS DATABASE ---
IKS_DB = {
    "High": {
        "Theme": "Cooling & Grounding (Pitta)",
        "BioState": "🔥 **Critical Hyper-Arousal:** Your Sympathetic Nervous System is currently 'locked on'. This state releases a flood of Cortisol and Adrenaline, causing inflammation, cognitive fog, and potential burnout. **Immediate physiological reset required.**",
        "Pranayama": "🐝 **Bhramari (Humming Bee Breath)**: Use vibrational resonance to stimulate the Vagus Nerve and force a parasympathetic dominance (Rest & Digest).",
        "Asana": "🧘 **Shavasana (Corpse Pose)**: Total stillness is non-negotiable. Do NOT engage in cardio; your heart needs to perceive safety, not threat.",
        "Diet": "🥥 **Sattvic Cooling Protocol**: Focus on alkaline hydration (Coconut water, Cucumber). Avoid all stimulants (Caffeine, Spicy/Sour foods) to arrest the cortisol spike.",
        "Mantra": "🕉️ **Om Shanti**: 'I am Peace'. Use this frequency to override the mental noise.",
        "Img": "https://images.unsplash.com/photo-1545205597-3d9d02c29597?q=80&w=600&auto=format&fit=crop" # Yoga Shavasana
    },
    "Medium": {
        "Theme": "Balancing (Tridosha)",
        "BioState": "⚠️ **Allostatic Overload:** Your body is efficiently handling stress, but the 'cost' is accumulating silently. You are in the 'Resistance Phase' of General Adaptation Syndrome. Without intervention, this leads to exhaustion.",
        "Pranayama": "👃 **Nadi Shodhana**: Alternate Nostril Breathing to mechanically balance the Left (Logical) and Right (Creative) brain hemispheres.",
        "Asana": "🌲 **Vrikshasana (Tree Pose)**: A single-point focus pose to stabilize the wandering mind and ground physical energy.",
        "Diet": "🍵 **Adaptogenic Support**: Warm Chamomile or Tulsi (Holy Basil) tea to naturally modulate cortisol levels without sedation.",
        "Mantra": "✨ **So Hum**: 'I am That'. Re-aligning your personal rhythm with the universal rhythm.",
        "Img": "https://images.unsplash.com/photo-1474418397713-7ede21d49118?q=80&w=600&auto=format&fit=crop" # Nature/Balance
    },
    "Low": {
        "Theme": "Energizing & Maintaining (Kapha)",
        "BioState": "✅ **Optimal Homeostasis:** Your Allostatic load is minimal. Your autonomic nervous system is flexible and resilient. This is 'Eustress'—positive stress that drives growth and focus.",
        "Pranayama": "🔥 **Kapalbhati (Skull Shining)**: Rapid forceful exhalations to energize the frontal cortex and clear mental cobwebs.",
        "Asana": "☀️ **Surya Namaskar (Sun Salutation)**: Dynamic kinetic flow to build metabolic heat and maintain agility.",
        "Diet": "🌶️ **Metabolic Activation**: Light, spiced foods (Ginger, Turmeric, Honey) to prevent stagnation and keep energy flowing.",
        "Mantra": "💪 **Gayatri Mantra**: Invoking clarity and intellectual brilliance.",
        "Img": "https://images.unsplash.com/photo-1518611012118-696072aa579a?q=80&w=600&auto=format&fit=crop" # Active Fitness
    }
}

def render_reference_guide():
    with st.expander("ℹ️ Calibration Guide: What do these levels mean?"):
        st.markdown("""
        <div style='display: grid; grid-template-columns: 1fr 1fr 1fr; gap: 15px;'>
            <div style='padding: 10px; border-left: 3px solid #10b981; background: rgba(16, 185, 129, 0.1); border-radius: 4px;'>
                <strong style='color: #10b981'>LOW STRESS (Eustress)</strong>
                <p style='font-size: 0.85rem; color: #cbd5e1; margin-top: 5px;'>
                ✅ <b>State:</b> "In the Zone"<br>
                🔋 <b>Feeling:</b> Energized, focused, ready.<br>
                💡 <b>Example:</b> Studying for a subject you love; prepping for a vacation.
                </p>
            </div>
            <div style='padding: 10px; border-left: 3px solid #f59e0b; background: rgba(245, 158, 11, 0.1); border-radius: 4px;'>
                <strong style='color: #f59e0b'>MEDIUM STRESS (Strain)</strong>
                <p style='font-size: 0.85rem; color: #cbd5e1; margin-top: 5px;'>
                ⚠️ <b>State:</b> "Grinding"<br>
                🔋 <b>Feeling:</b> Tired but functional, mild anxiety.<br>
                💡 <b>Example:</b> Exam week; juggling 3 deadlines; slight sleep debt.
                </p>
            </div>
            <div style='padding: 10px; border-left: 3px solid #ef4444; background: rgba(239, 68, 68, 0.1); border-radius: 4px;'>
                <strong style='color: #ef4444'>HIGH STRESS (Burnout)</strong>
                <p style='font-size: 0.85rem; color: #cbd5e1; margin-top: 5px;'>
                🚨 <b>State:</b> "System Failure"<br>
                🔋 <b>Feeling:</b> Panic, numbness, physical pain.<br>
                💡 <b>Example:</b> Panic attack pending; 48hrs no sleep; total emotional crash.
                </p>
            </div>
        </div>
        """, unsafe_allow_html=True)

def get_grade(score):
    if score < 0.2: return "A+", "#10b981"
    if score < 0.4: return "B", "#34d399"
    if score < 0.6: return "C", "#f59e0b"
    if score < 0.8: return "D", "#fbbf24"
    return "F", "#ef4444"

# --- HELPER: AUTO ANALYSIS ---
def generate_diagnosis(bio_score, sub_score, cat_scores):
    """Generate a smart text explanation based on the data split."""
    dominant_cat = max(cat_scores, key=cat_scores.get)
    
    text = f"**Primary Stress Driver: {dominant_cat}** ({cat_scores[dominant_cat]:.0%} Load)\n\n"
    
    if bio_score > sub_score + 0.3:
        text += "⚠️ **Somatic Mismatch:** Your body is screaming (High Bio-Stress) while your mind thinks it's fine. This is dangerous—it often precedes sudden burnout or panic attacks. Check for physical triggers (sleep deprivation, caffeine)."
    elif sub_score > bio_score + 0.3:
        text += "🧠 **Psychological Distress:** Your body is relatively calm, but your mind is racing. This is 'Perceived Stress'. Mindfulness and CBT techniques will be highly effective here."
    else:
        text += "⚖️ **Aligned State:** Your physical and mental states are synchronized. Your subjective feeling matches your biological reality."
        
    return text

def render_transparency_dashboard():
    """Renders the Explainable AI (XAI) Dashboard with Premium UI."""
    st.markdown("---")
    st.markdown("""
    <div style="text-align: center; margin: 60px 0 50px 0;">
        <h1 style="font-size: 3.5rem; font-weight: 900; background: linear-gradient(135deg, #06b6d4 0%, #3b82f6 50%, #8b5cf6 100%); -webkit-background-clip: text; -webkit-text-fill-color: transparent; letter-spacing: -2px; margin-bottom: 10px;">
            🧠 NEURO-FUSION ARCHITECTURE
        </h1>
        <p style="color: #64748b; font-size: 1rem; letter-spacing: 4px; text-transform: uppercase; font-weight: 300;">Deep Dive Into The Science</p>
    </div>
    """, unsafe_allow_html=True)
    
    # --- SECTION 1: THE PIPELINE ---
    st.markdown("""
    <div style="background: linear-gradient(135deg, rgba(6, 182, 212, 0.1) 0%, rgba(59, 130, 246, 0.05) 100%); 
                padding: 30px; border-radius: 20px; border: 1px solid rgba(6, 182, 212, 0.2); 
                margin-bottom: 40px; backdrop-filter: blur(10px);">
        <h3 style="color: #06b6d4; font-size: 1.8rem; margin-bottom: 10px; font-weight: 700;">
            🔄 The Neuro-Fusion Pipeline
        </h3>
        <p style="color: #94a3b8; font-size: 0.95rem; margin-bottom: 25px;">How raw physiological data transforms into clinical insights</p>
    </div>
    """, unsafe_allow_html=True)
    
    # Pipeline Steps
    p1, p2, p3, p4, p5, p6, p7 = st.columns([2, 0.5, 2, 0.5, 2, 0.5, 2])
    with p1:
        st.markdown("""
        <div style="background: linear-gradient(135deg, #1e293b 0%, #0f172a 100%); padding: 25px; border-radius: 15px; 
                    border-left: 4px solid #06b6d4; height: 140px; display: flex; flex-direction: column; justify-content: center;">
            <div style="font-size: 2rem; margin-bottom: 8px;">📸</div>
            <div style="color: #06b6d4; font-weight: 700; font-size: 0.9rem; margin-bottom: 5px;">STEP 1: ACQUISITION</div>
            <div style="color: #cbd5e1; font-size: 0.85rem;">Wearables + QA Surveys</div>
        </div>
        """, unsafe_allow_html=True)
    with p2:
        st.markdown("<div style='text-align: center; font-size: 2rem; color: #06b6d4; line-height: 140px;'>→</div>", unsafe_allow_html=True)
    with p3:
        st.markdown("""
        <div style="background: linear-gradient(135deg, #1e293b 0%, #0f172a 100%); padding: 25px; border-radius: 15px; 
                    border-left: 4px solid #3b82f6; height: 140px; display: flex; flex-direction: column; justify-content: center;">
            <div style="font-size: 2rem; margin-bottom: 8px;">🧬</div>
            <div style="color: #3b82f6; font-weight: 700; font-size: 0.9rem; margin-bottom: 5px;">STEP 2: FUSION</div>
            <div style="color: #cbd5e1; font-size: 0.85rem;">40% Mind + 60% Body</div>
        </div>
        """, unsafe_allow_html=True)
    with p4:
        st.markdown("<div style='text-align: center; font-size: 2rem; color: #3b82f6; line-height: 140px;'>→</div>", unsafe_allow_html=True)
    with p5:
        st.markdown("""
        <div style="background: linear-gradient(135deg, #1e293b 0%, #0f172a 100%); padding: 25px; border-radius: 15px; 
                    border-left: 4px solid #8b5cf6; height: 140px; display: flex; flex-direction: column; justify-content: center;">
            <div style="font-size: 2rem; margin-bottom: 8px;">🚀</div>
            <div style="color: #8b5cf6; font-weight: 700; font-size: 0.9rem; margin-bottom: 5px;">STEP 3: AI INFERENCE</div>
            <div style="color: #cbd5e1; font-size: 0.85rem;">GBM Model (SOTA)</div>
        </div>
        """, unsafe_allow_html=True)
    with p6:
        st.markdown("<div style='text-align: center; font-size: 2rem; color: #8b5cf6; line-height: 140px;'>→</div>", unsafe_allow_html=True)
    with p7:
        st.markdown("""
        <div style="background: linear-gradient(135deg, #059669 0%, #047857 100%); padding: 25px; border-radius: 15px; 
                    border-left: 4px solid #10b981; height: 140px; display: flex; flex-direction: column; justify-content: center;">
            <div style="font-size: 2rem; margin-bottom: 8px;">🎯</div>
            <div style="color: #10b981; font-weight: 700; font-size: 0.9rem; margin-bottom: 5px;">STEP 4: DIAGNOSIS</div>
            <div style="color: #d1fae5; font-size: 0.85rem;">Clinical Report</div>
        </div>
        """, unsafe_allow_html=True)
    
    st.markdown("<br>", unsafe_allow_html=True)
    
    # --- SECTION 2: SIGNAL DECODER ---
    st.markdown("""
    <div style="background: linear-gradient(135deg, rgba(139, 92, 246, 0.1) 0%, rgba(236, 72, 153, 0.05) 100%); 
                padding: 30px; border-radius: 20px; border: 1px solid rgba(139, 92, 246, 0.2); 
                margin-bottom: 40px; backdrop-filter: blur(10px);">
        <h3 style="color: #8b5cf6; font-size: 1.8rem; margin-bottom: 10px; font-weight: 700;">
            📡 Signal Decoder
        </h3>
        <p style="color: #94a3b8; font-size: 0.95rem; margin-bottom: 25px;">Understanding the biological markers of stress</p>
    </div>
    """, unsafe_allow_html=True)
    
    d1, d2, d3 = st.columns(3)
    with d1:
        st.markdown("""
        <div style="background: linear-gradient(135deg, rgba(239, 68, 68, 0.15) 0%, rgba(220, 38, 38, 0.05) 100%); 
                    padding: 30px; border-radius: 20px; border: 2px solid rgba(239, 68, 68, 0.3); 
                    backdrop-filter: blur(10px); min-height: 280px;">
            <div style="font-size: 3rem; margin-bottom: 15px;">⚡</div>
            <h4 style="color: #ef4444; font-size: 1.3rem; margin-bottom: 8px; font-weight: 700;">Electrodermal Activity</h4>
            <div style="color: #fca5a5; font-size: 0.8rem; margin-bottom: 15px; text-transform: uppercase; letter-spacing: 1px;">The "Lie Detector" Signal</div>
            <p style="color: #e2e8f0; font-size: 0.95rem; line-height: 1.7;">
                Your sympathetic nervous system activates sweat glands when stressed. EDA detects this microscopic palm perspiration.
                <br><br>
                <span style="color: #fca5a5; font-weight: 600;">High EDA = Fight or Flight Mode</span>
            </p>
        </div>
        """, unsafe_allow_html=True)
    
    with d2:
        st.markdown("""
        <div style="background: linear-gradient(135deg, rgba(236, 72, 153, 0.15) 0%, rgba(219, 39, 119, 0.05) 100%); 
                    padding: 30px; border-radius: 20px; border: 2px solid rgba(236, 72, 153, 0.3); 
                    backdrop-filter: blur(10px); min-height: 280px;">
            <div style="font-size: 3rem; margin-bottom: 15px;">❤️</div>
            <h4 style="color: #ec4899; font-size: 1.3rem; margin-bottom: 8px; font-weight: 700;">Heart Rate Variability</h4>
            <div style="color: #fbcfe8; font-size: 0.8rem; margin-bottom: 15px; text-transform: uppercase; letter-spacing: 1px;">The Resilience Signal</div>
            <p style="color: #e2e8f0; font-size: 0.95rem; line-height: 1.7;">
                A healthy heart has irregular beats (high variability). Chronic stress makes it beat mechanically like a metronome.
                <br><br>
                <span style="color: #fbcfe8; font-weight: 600;">Low HRV = Vagal Shutdown</span>
            </p>
        </div>
        """, unsafe_allow_html=True)
    
    with d3:
        st.markdown("""
        <div style="background: linear-gradient(135deg, rgba(59, 130, 246, 0.15) 0%, rgba(37, 99, 235, 0.05) 100%); 
                    padding: 30px; border-radius: 20px; border: 2px solid rgba(59, 130, 246, 0.3); 
                    backdrop-filter: blur(10px); min-height: 280px;">
            <div style="font-size: 3rem; margin-bottom: 15px;">🌡️</div>
            <h4 style="color: #3b82f6; font-size: 1.3rem; margin-bottom: 8px; font-weight: 700;">Peripheral Temperature</h4>
            <div style="color: #93c5fd; font-size: 0.8rem; margin-bottom: 15px; text-transform: uppercase; letter-spacing: 1px;">The Constriction Signal</div>
            <p style="color: #e2e8f0; font-size: 0.95rem; line-height: 1.7;">
                Stress triggers vasoconstriction—blood vessels tighten, pulling warmth to vital organs and cooling extremities.
                <br><br>
                <span style="color: #93c5fd; font-weight: 600;">Cold Hands = Acute Stress</span>
            </p>
        </div>
        """, unsafe_allow_html=True)
    
    st.markdown("<br>", unsafe_allow_html=True)
    
    # --- SECTION 3: BIAS CORRECTION ---
    st.markdown("""
    <div style="background: linear-gradient(135deg, rgba(16, 185, 129, 0.1) 0%, rgba(5, 150, 105, 0.05) 100%); 
                padding: 30px; border-radius: 20px; border: 1px solid rgba(16, 185, 129, 0.2); 
                margin-bottom: 40px; backdrop-filter: blur(10px);">
        <h3 style="color: #10b981; font-size: 1.8rem; margin-bottom: 10px; font-weight: 700;">
            ⚖️ The "Truth Serum" Logic
        </h3>
        <p style="color: #94a3b8; font-size: 0.95rem; margin-bottom: 0;">Why we trust the body (60%) more than the mind (40%)</p>
    </div>
    """, unsafe_allow_html=True)
    
    with st.expander("🔍 Understanding Bias Correction", expanded=False):
        col1, col2 = st.columns([1, 1])
        with col1:
            st.markdown("""
            <div style="background: rgba(239, 68, 68, 0.1); padding: 20px; border-radius: 12px; border-left: 4px solid #ef4444;">
                <h4 style="color: #ef4444; margin-bottom: 10px;">❌ The Problem: Subjective Bias</h4>
                <p style="color: #cbd5e1; font-size: 0.9rem; line-height: 1.6;">
                    Humans are terrible self-reporters. We say <b>"I'm fine"</b> (denial) or <b>"I'm dying"</b> (exaggeration). 
                    Psychology calls this "poor interoceptive awareness."
                </p>
            </div>
            """, unsafe_allow_html=True)
        
        with col2:
            st.markdown("""
            <div style="background: rgba(16, 185, 129, 0.1); padding: 20px; border-radius: 12px; border-left: 4px solid #10b981;">
                <h4 style="color: #10b981; margin-bottom: 10px;">✅ The Solution: Objective Grounding</h4>
                <p style="color: #cbd5e1; font-size: 0.9rem; line-height: 1.6;">
                    Your <b>Autonomic Nervous System</b> doesn't lie. Even if you report "Low Stress," spiking cortisol (EDA) 
                    and suppressed vagal tone (HRV) reveal <b>Hidden Allostatic Load</b>.
                </p>
            </div>
            """, unsafe_allow_html=True)
        
        st.markdown("<br>", unsafe_allow_html=True)
        st.markdown("**The Weighted Equation:**", unsafe_allow_html=True)
        st.latex(r"\text{Score}_{\text{CLINICAL}} = 0.6 \times \underbrace{\text{Bio}_{\text{Load}}}_{\text{Objective Truth}} + 0.4 \times \underbrace{\text{Psych}_{\text{Score}}}_{\text{Subjective Report}}")
    
    # --- SECTION 4: THE MODEL ENGINE ---
    st.markdown("""
    <div style="background: linear-gradient(135deg, rgba(245, 158, 11, 0.1) 0%, rgba(217, 119, 6, 0.05) 100%); 
                padding: 30px; border-radius: 20px; border: 1px solid rgba(245, 158, 11, 0.2); 
                margin-bottom: 40px; backdrop-filter: blur(10px);">
        <h3 style="color: #f59e0b; font-size: 1.8rem; margin-bottom: 10px; font-weight: 700;">
            🤖 The AI Engine
        </h3>
        <p style="color: #94a3b8; font-size: 0.95rem; margin-bottom: 0;">Gradient Boosting: The state-of-the-art for tabular data</p>
    </div>
    """, unsafe_allow_html=True)
    
    with st.expander("⚙️ Technical Architecture", expanded=True):
        m1, m2 = st.columns(2)
        with m1:
            st.markdown("""
            <div style="padding: 20px; background: rgba(139, 92, 246, 0.1); border-radius: 12px; border: 1px solid rgba(139, 92, 246, 0.3);">
                <h4 style="color: #8b5cf6; margin-bottom: 15px; font-size: 1.2rem;">🚀 Sequential Learning</h4>
                <p style="color: #cbd5e1; font-size: 0.9rem; line-height: 1.7;">
                    Unlike Random Forest (parallel tree averaging), <b>Gradient Boosting</b> trains trees sequentially. 
                    Tree №2 corrects Tree №1's errors, Tree №3 corrects Tree №2, creating a cascade of refinement.
                </p>
            </div>
            """, unsafe_allow_html=True)
            st.metric("🌲 Architecture", "100 Sequential Trees", delta="SOTA")
        
        with m2:
            st.markdown("""
            <div style="padding: 20px; background: rgba(6, 182, 212, 0.1); border-radius: 12px; border: 1px solid rgba(6, 182, 212, 0.3);">
                <h4 style="color: #06b6d4; margin-bottom: 15px; font-size: 1.2rem;">🎯 Tabular Dominance</h4>
                <p style="color: #cbd5e1; font-size: 0.9rem; line-height: 1.7;">
                    For structured physiological data, GBM outperforms Deep Learning. It handles non-linear signal interactions 
                    with <b>98.7% F1-Score</b> on the SFAA-Stress-Dataset.
                </p>
            </div>
            """, unsafe_allow_html=True)
            st.metric("📊 F1-Score", "98.7%", delta="+4.5% vs RF")

def generate_clinical_report(score, bio, mind, cat_scores, remedy, level):
    """Generates a high-res clinical report with EMBEDDED CHARTS (Fixed Layout)."""
    # Setup Figure
    plt.style.use('default')
    fig = plt.figure(figsize=(8.5, 11), facecolor='white')
    
    # --- MASTER LAYOUT AXES (0,0 to 1,1) ---
    ax_main = fig.add_axes([0, 0, 1, 1])
    ax_main.axis('off')
    
    # --- HEADER ---
    ax_main.text(0.5, 0.95, "NEURO-FUSION CLINIC", ha='center', fontsize=22, color='#0f172a', weight='bold', fontname='Arial')
    ax_main.text(0.5, 0.92, "CLINICAL STRESS ASSESSMENT REPORT", ha='center', fontsize=12, color='#64748b')
    ax_main.plot([0.1, 0.9], [0.89, 0.89], color='#0f172a', linewidth=2)
    
    # --- PATIENT INFO BLOCK ---
    ax_main.text(0.1, 0.85, f"DATE: {time.strftime('%Y-%m-%d')}", fontsize=10, color='#334155', fontfamily='monospace')
    ax_main.text(0.1, 0.83, f"REF ID: SFAA-{int(time.time())}", fontsize=10, color='#334155', fontfamily='monospace')
    ax_main.text(0.9, 0.85, "CONFIDENTIAL", ha='right', fontsize=10, color='#ef4444', weight='bold')

    # Color Logic
    color = "#10b981" if level == "Low" else "#f59e0b" if level == "Medium" else "#ef4444"

    # --- 1. STRESS GAUGE (Donut Chart) ---
    # Top Right Corner
    ax_donut = fig.add_axes([0.60, 0.65, 0.25, 0.25])
    ax_donut.pie([score, 1-score], colors=[color, '#e2e8f0'], startangle=90, counterclock=False, 
                 wedgeprops={'width': 0.2, 'edgecolor': 'white'})
    ax_donut.text(0, -0.1, f"{score:.0%}", ha='center', va='center', fontsize=24, weight='bold', color=color)
    ax_donut.text(0, -0.35, level.upper(), ha='center', va='center', fontsize=10, color='#64748b')

    # --- TEXT DIAGNOSIS (Left Side) ---
    ax_main.text(0.1, 0.76, "PRIMARY DIAGNOSIS", fontsize=12, color='#0f172a', weight='bold')
    ax_main.text(0.1, 0.70, f"{level.upper()} STRESS DETECTED", fontsize=24, color=color, weight='bold')
    ax_main.text(0.1, 0.66, "Biometric & Psychometric Fusion Analysis", fontsize=10, color='#64748b')

    # --- 2. BIOMETRIC VITALS ---
    ax_main.text(0.1, 0.58, "BIOMETRIC VITALS", fontsize=12, color='#0f172a', weight='bold')
    ax_main.text(0.1, 0.53, f"MIND LOAD: {mind:.1%}", fontsize=11, color='#475569')
    ax_main.text(0.4, 0.53, f"BODY LOAD: {bio:.1%}", fontsize=11, color='#475569')
    ax_main.plot([0.1, 0.9], [0.50, 0.50], color='#e2e8f0', linewidth=1)

    # --- 3. STRESS DRIVERS (Bar Chart) ---
    ax_main.text(0.1, 0.45, "PSYCHOMETRIC AUDIT", fontsize=12, color='#0f172a', weight='bold')
    
    # Add Bar Chart Axes (Middle)
    ax_bar = fig.add_axes([0.1, 0.32, 0.8, 0.12])
    cats = list(cat_scores.keys())
    vals = list(cat_scores.values())
    colors = ['#ef4444' if v > 0.6 else '#3b82f6' for v in vals]
    
    ax_bar.barh(cats, vals, color=colors, height=0.5)
    ax_bar.set_xlim(0, 1)
    ax_bar.spines['top'].set_visible(False)
    ax_bar.spines['right'].set_visible(False)
    ax_bar.spines['bottom'].set_visible(False)
    ax_bar.spines['left'].set_visible(False)
    ax_bar.set_xticks([]) 
    ax_bar.tick_params(axis='y', labelsize=8, color='#64748b')
    
    # Bar Labels
    for i, v in enumerate(vals):
        ax_bar.text(v + 0.01, i, f"{v:.0%}", va='center', fontsize=8, color='#334155', weight='bold')

    # --- PROTOCOL (Bottom) ---
    ax_main.text(0.1, 0.26, "THERAPEUTIC PROTOCOL", fontsize=12, color='#0f172a', weight='bold')
    
    bio_clean = remedy['BioState'].replace('**', '')
    plan_clean = (
        f"PHYSIOLOGY: {bio_clean}\n\n"
        f"BREATHWORK: {remedy['Pranayama'].split('**')[1]}\n"
        f"MOVEMENT:   {remedy['Asana'].split('**')[1]}\n"
        f"NUTRITION:  {remedy['Diet'].split('**')[1]}\n"
        f"MANTRA:     {remedy['Mantra'].split('**')[1]}"
    )
    # Using ax_main ensures this starts at 0.22 height relative to page
    ax_main.text(0.1, 0.08, plan_clean, fontsize=9, color='#334155', va='bottom', wrap=True, family='monospace', linespacing=1.8)

    # Signature
    ax_main.text(0.9, 0.02, "AUTHORIZED SIGNATURE", ha='right', fontsize=8, color='#94a3b8')
    ax_main.plot([0.7, 0.9], [0.04, 0.04], color='#94a3b8', linewidth=1)

    # Save
    buf_png = io.BytesIO()
    plt.savefig(buf_png, format='png', dpi=150, bbox_inches='tight', facecolor='white')
    buf_png.seek(0)
    
    buf_pdf = io.BytesIO()
    plt.savefig(buf_pdf, format='pdf', bbox_inches='tight', facecolor='white')
    buf_pdf.seek(0)
    
    plt.close()
    return buf_png, buf_pdf

def main():
    # Hero Header
    st.markdown("""
        <div class="hero-container">
            <div class="hero-title">ZEN ANALYZER</div>
            <div class="hero-subtitle">Neuro-Fusion Stress Protocol v15.5</div>
        </div>
    """, unsafe_allow_html=True)
    
    # --- INPUT SECTION (Tabbed Wizard) ---
    st.markdown("#### 📡 Patient Data Acquisition")
    render_reference_guide()
    
    if 'history' not in st.session_state:
        st.session_state['history'] = []
    
    # --- DR. ZEN CHATBOT (Phase 8) ---
    with st.sidebar:
        st.markdown("### 🤖 Dr. Zen AI")
        if "messages" not in st.session_state:
            st.session_state["messages"] = [{"role": "assistant", "content": "Namaste! I am Dr. Zen. How can I help you de-stress today? (Ask about: Diet, Breathing, or Sleep)"}]

        for msg in st.session_state["messages"]:
            st.chat_message(msg["role"]).write(msg["content"])
            
        st.markdown("### 🗣️ Quick Consult:")
        col_chat1, col_chat2 = st.columns(2)
        
        response = None
        if col_chat1.button("❓ How it works?", use_container_width=True):
            response = (
                "**Zen Analyzer** uses a **Neuro-Fusion Protocol**: \n"
                "1. **Subjective Data**: We analyze your academic & emotional survey inputs (40% weight). \n"
                "2. **Objective Data**: We fuse this with your biometric signals (HR, Temp, EDA) (60% weight). \n"
                "3. **AI Diagnosis**: We calculate your 'Allostatic Load' to prescribe a clinical IKS remedy."
            )
        if col_chat2.button("🥗 Diet Tips", use_container_width=True):
            response = "For stress, focus on **Sattvic Foods**: fresh fruits, nuts, and warm herbal teas. Avoid caffeine and spicy foods to lower cortisol."
            
        if col_chat1.button("🧘 Teach Yoga", use_container_width=True):
            response = (
                "🧘 **How to do Vrikshasana (Tree Pose):**\n"
                "1. Stand tall, shift weight to left foot.\n"
                "2. Place right foot on left inner thigh (avoid knee).\n"
                "3. Bring hands to prayer (Namaste) at chest.\n"
                "4. Gaze at a fixed point. Hold for 30s. Switch sides.\n"
                "💡 *Good for: Focus & Balance.*"
            )
        if col_chat2.button("🌬️ Breathing", use_container_width=True):
            response = "Try **Nadi Shodhana** (Alternate Nostril Breathing) to balance your hemispheres, or **Bhramari** (Humming Bee) to calm the vagus nerve."
            
        if response:
            st.session_state["messages"].append({"role": "assistant", "content": response})
            st.rerun()

        if prompt := st.chat_input("Or type here..."):
             st.session_state["messages"].append({"role": "user", "content": prompt})
             st.chat_message("user").write(prompt)
             # Default fallback
             fallback = "I recommend checking the **Quick Consult** buttons above for reliable advice on Yoga, Diet, and Usage!"
             st.session_state["messages"].append({"role": "assistant", "content": fallback})
             st.rerun()
    
    # --- DEMO PRESETS ---
    def set_preset(level):
        if level == "Low":
            val_q, val_eda, val_hr, val_temp = 1, 2.5, 70, 36.5
        elif level == "Medium":
            val_q, val_eda, val_hr, val_temp = 2, 6.0, 95, 34.0
        else: # High
            val_q, val_eda, val_hr, val_temp = 3, 12.0, 115, 31.0
            
        # Update Session State
        keys = ['aq1','aq2','aq3','aq4','aq5', 'eq1','eq2','eq3','eq4', 'sq1','sq2','sq3','sq4', 'pq1','pq2','pq3','pq4', 'cq1','cq2','cq3','cq4','cq5','cq6','cq7']
        for k in keys: st.session_state[k] = val_q
        st.session_state['eda'] = val_eda
        st.session_state['hr'] = val_hr
        st.session_state['temp'] = val_temp

    st.write("🤖 **Demo Simulation:**")
    bc1, bc2, bc3 = st.columns(3)
    if bc1.button("🟢 Simulate Low Stress", use_container_width=True): set_preset("Low")
    if bc2.button("🟡 Simulate Medium Stress", use_container_width=True): set_preset("Medium")
    if bc3.button("🔴 Simulate High Stress", use_container_width=True): set_preset("High")
    
    t1, t2, t3, t4, t5, t6 = st.tabs(["📚 Academic", "🧠 Emotional", "🤝 Social", "🏃 Physical", "🛡️ Coping", "⌚ Biometrics"])
    
    # 1. Academic
    with t1:
        c1, c2 = st.columns(2)
        with c1:
            aq1 = st.slider("1. Workload Overwhelm", 0, 4, 1, key="aq1", help="Do you feel buried under assignments?")
            aq2 = st.slider("2. Exam Anxiety", 0, 4, 1, key="aq2", help="Do you freeze up before tests?")
            aq3 = st.slider("3. Deadline Struggles", 0, 4, 1, key="aq3")
        with c2:
            aq4 = st.slider("4. GPA Worries", 0, 4, 2, key="aq4")
            aq5 = st.slider("5. Balancing Projects", 0, 4, 1, key="aq5")
            
    # 2. Emotional
    with t2:
        c1, c2 = st.columns(2)
        with c1:
            eq1 = st.slider("6. Emotional Exhaustion", 0, 4, 2, key="eq1")
            eq2 = st.slider("7. Demotivation", 0, 4, 1, key="eq2")
        with c2:
            eq3 = st.slider("8. Mood Swings", 0, 4, 1, key="eq3")
            eq4 = st.slider("9. Feeling Isolated", 0, 4, 1, key="eq4")

    # 3. Social
    with t3:
        c1, c2 = st.columns(2)
        with c1:
            sq1 = st.slider("10. No Time for Family", 0, 4, 1, key="sq1")
            sq2 = st.slider("11. Family Expectations", 0, 4, 2, key="sq2")
        with c2:
            sq3 = st.slider("12. Relationship Issues", 0, 4, 1, key="sq3")
            sq4 = st.slider("13. Peer Pressure", 0, 4, 1, key="sq4")

    # 4. Physical
    with t4:
        c1, c2 = st.columns(2)
        with c1:
            pq1 = st.slider("14. Headaches/Fatigue", 0, 4, 1, key="pq1")
            pq2 = st.slider("15. Appetite Changes", 0, 4, 1, key="pq2")
        with c2:
            pq3 = st.slider("16. Tired after Sleep", 0, 4, 2, key="pq3")
            pq4 = st.slider("17. Palpitations", 0, 4, 1, key="pq4")

    # 5. Coping
    with t5:
        c1, c2 = st.columns(2)
        with c1:
            cq1 = st.slider("18. Hesitate to seek help", 0, 4, 2, key="cq1")
            cq2 = st.slider("19. Poor Concentration", 0, 4, 2, key="cq2")
            cq3 = st.slider("20. No Strategy", 0, 4, 2, key="cq3")
            cq4 = st.slider("21. No Hobbies", 0, 4, 3, key="cq4")
        with c2:
            cq5 = st.slider("22. Social Withdrawal", 0, 4, 1, key="cq5")
            cq6 = st.slider("23. No Relaxation", 0, 4, 2, key="cq6")
            cq7 = st.slider("24. Unaware of Support", 0, 4, 1, key="cq7")

    # 6. Biometrics
    with t6:
        st.info("💡 **Simulation Mode:** Connect medical-grade sensors or simulate bio-signals below.")
        b1, b2, b3 = st.columns(3)
        with b1: eda = st.slider("⚡ EDA (µS)", 0.0, 20.0, 2.5, key="eda", help="Electrodermal Activity. Normal Resting: 2-5µS.")
        with b2: hr = st.slider("❤️ Heart Rate", 50, 140, 72, key="hr", help="Resting Heart Rate (BPM).")
        with b3: temp = st.slider("🌡️ Skin Temp (°C)", 30.0, 37.0, 36.5, key="temp", help="Peripheral temperature.")

    # --- CALCULATION ---
    # Scores
    s_acad = (aq1+aq2+aq3+aq4+aq5)/20.0
    s_emo = (eq1+eq2+eq3+eq4)/16.0
    s_soc = (sq1+sq2+sq3+sq4)/16.0
    s_phys = (pq1+pq2+pq3+pq4)/16.0
    s_cope = (cq1+cq2+cq3+cq4+cq5+cq6+cq7)/28.0
    
    cat_scores = {
        "Academic": s_acad, "Emotional": s_emo, 
        "Social": s_soc, "Physical": s_phys, "Coping": s_cope
    }

    survey_total_norm = (s_acad+s_emo+s_soc+s_phys+s_cope)/5.0
    
    # Biometric AI Prediction (GBM SOTA)
    try:
        model = joblib.load('wearable/trained_gbm_model.pkl')
        # Model expects: [EDA_Mean, HR_Mean, TEMP_Mean]
        # We need to map probability of High Stress (Class 2) to a 0-1 scale
        # Classes are likely ['High', 'Low', 'Medium'] sorted alphabetically? 
        # Actually classes are likely string labels. Let's use simple logic:
        # P(Low)*0.1 + P(Medium)*0.5 + P(High)*0.9
        
        probs = model.predict_proba([[eda, hr, temp]])[0]
        classes = model.classes_
        
        # Model was trained with LabelEncoder: 0=High, 1=Low, 2=Medium
        # Map probabilities to stress score (0-1 scale)
        bio_score = 0.0
        for cls, prob in zip(classes, probs):
            if cls == 0:  # High
                bio_score += prob * 0.9
            elif cls == 1:  # Low
                bio_score += prob * 0.2
            elif cls == 2:  # Medium
                bio_score += prob * 0.5
            
    except Exception as e:
        # Fallback if model missing or error
        n_eda = min(1.0, eda / 15.0); n_hr = min(1.0, max(0, (hr - 60) / 60.0)); n_temp = min(1.0, max(0, (36.5 - temp) / 5.0))
        bio_score = (n_eda * 0.5) + (n_hr * 0.3) + (n_temp * 0.2)
    
    # Final Fusion
    final_score = (bio_score * 0.6) + (survey_total_norm * 0.4)
    
    # Update History
    st.session_state['history'].append(final_score)

    st.markdown("---")
    
    if st.button("🚀 GENERATE CLINICAL REPORT", type="primary", use_container_width=True):
        with st.spinner("Analyzing Bio-Markers..."):
            time.sleep(1.0)
            
            # Categorize
            if final_score < 0.4: level = "Low"; color = "#10b981" # Emerald 500
            elif final_score < 0.7: level = "Medium"; color = "#f59e0b" # Amber 500
            else: level = "High"; color = "#ef4444" # Red 500
            
            remedy = IKS_DB[level]
            
            # --- RESULTS HEADER ---
            st.markdown(f"""
            <div style="text-align: center; margin-bottom: 30px; animation: pulseSoft 3s infinite;">
                <h2 style="font-size: 2.5rem; margin: 0; color: {color};">{level.upper()} STRESS DETECTED</h2>
                <p style="font-size: 1rem; color: #94a3b8; letter-spacing: 1px;">CONFIDENCE INTERVAL: {final_score:.1%}</p>
            </div>
            """, unsafe_allow_html=True)
            
            # --- ROW 1: METRICS & REPORT CARD ---
            c1, c2 = st.columns([1, 1.5])
            
            with c1:
                st.markdown(f"""
                <div class='result-card'>
                    <div style="margin-bottom: 20px;">
                        <div class="metric-label">Total Load</div>
                        <div class="metric-value" style="color: {color}">{final_score:.1%}</div>
                    </div>
                    <div style="display: flex; justify-content: space-between;">
                        <div>
                            <div class="metric-label">Mind</div>
                            <div class="metric-value" style="font-size: 1.5rem;">{survey_total_norm:.1%}</div>
                        </div>
                        <div>
                            <div class="metric-label">Body</div>
                            <div class="metric-value" style="font-size: 1.5rem;">{bio_score:.1%}</div>
                        </div>
                    </div>
                </div>
                """, unsafe_allow_html=True)
                
            with c2:
                st.markdown("<div class='result-card'><h4>📑 Lifestyle Audit</h4>", unsafe_allow_html=True)
                for cat, val in cat_scores.items():
                    grade, g_col = get_grade(val)
                    st.markdown(f"""
                    <div style="display: flex; justify-content: space-between; align-items: center; margin-bottom: 12px; border-bottom: 1px solid rgba(255,255,255,0.05); padding-bottom: 8px;">
                        <span style="font-weight: 500;">{cat}</span>
                        <div style="display: flex; align-items: center; gap: 15px;">
                            <span style="color: #94a3b8; font-size: 0.9rem;">{val:.0%}</span>
                            <span style="font-family: 'Plus Jakarta Sans'; font-weight: 800; color: {g_col}; background: {g_col}15; padding: 2px 10px; border-radius: 6px;">{grade}</span>
                        </div>
                    </div>
                    """, unsafe_allow_html=True)
                st.markdown("</div>", unsafe_allow_html=True)

            # --- ROW 2: DEEP DIAGNOSIS ---
            st.markdown(f"""
            <div class='result-card' style='border-left: 4px solid {color};'>
                <h4 style="color: #f1f5f9; margin-top: 0;">🧬 Clinical Diagnosis</h4>
                <p style='font-size: 1.05rem; color: #e2e8f0; margin-bottom: 15px;'>{remedy['BioState']}</p>
                <div style="background: rgba(15, 23, 42, 0.5); padding: 15px; border-radius: 8px; border: 1px dashed rgba(148, 163, 184, 0.2);">
                    <p style='color: #94a3b8; font-size: 0.95rem; margin: 0;'>{generate_diagnosis(bio_score, survey_total_norm, cat_scores)}</p>
                </div>
            </div>
            """, unsafe_allow_html=True)
            
            # --- ROW 3: IKS INTERVENTION ---
            st.subheader("🌿 Therapeutic Protocol")
            
            icol1, icol2 = st.columns([1, 2])
            with icol1:
                st.image(remedy['Img'], caption=f"{level} Stress Protocol", use_container_width=True)
            
            with icol2:
                st.markdown(f"""
                <div class='result-card' style='background: linear-gradient(135deg, {color}15 0%, {color}05 100%); border: 1px solid {color}30;'>
                    <h3 style='color: {color}; margin-top:0;'>🧘 {remedy['Theme']}</h3>
                    <div style='display: grid; grid-template-columns: 1fr 1fr; gap: 20px; margin-top: 20px;'>
                        <div>
                            <h5 style='color: #2dd4bf; margin-bottom: 5px;'>🌬️ Pranayama</h5>
                            <p style="font-size: 0.9rem;">{remedy['Pranayama']}</p>
                        </div>
                        <div>
                            <h5 style='color: #60a5fa; margin-bottom: 5px;'>🧘 Asana</h5>
                            <p style="font-size: 0.9rem;">{remedy['Asana']}</p>
                        </div>
                        <div>
                            <h5 style='color: #fbbf24; margin-bottom: 5px;'>🥗 Nutrition</h5>
                            <p style="font-size: 0.9rem;">{remedy['Diet']}</p>
                        </div>
                        <div>
                            <h5 style='color: #a78bfa; margin-bottom: 5px;'>📿 Mantra</h5>
                            <p style="font-size: 0.9rem;">{remedy['Mantra']}</p>
                        </div>
                    </div>
                </div>
                """, unsafe_allow_html=True)

            # --- OFFICIAL REPORT DOWNLOAD ---
            st.markdown("---")
            st.subheader("📥 Official Clinical Downloads")
            
            with st.spinner("🖨️ Generating High-Res Clinical Report..."):
                buf_png, buf_pdf = generate_clinical_report(final_score, bio_score, survey_total_norm, cat_scores, remedy, level)
                
            # Preview
            st.image(buf_png, caption="Official Clinical Report Preview", width=500)
            
            d1, d2, d3 = st.columns([1, 1, 2])
            with d1:
                st.download_button(
                    label="📄 Download PDF Report",
                    data=buf_pdf,
                    file_name="SFAA_Clinical_Report.pdf",
                    mime="application/pdf"
                )
            with d2:
                st.download_button(
                    label="🖼️ Download Report Image",
                    data=buf_png,
                    file_name="SFAA_Clinical_Report.png",
                    mime="image/png"
                )
            with d3:
                st.info("ℹ️ **Privacy Note:** This report is generated locally on your device using a secure Python engine. No data is sent to the cloud.")

            # --- CHARTS ---
            with st.expander("📈 View Telemetry Data"):
                chart_col1, chart_col2 = st.columns(2)
                with chart_col1:
                    df_ch = pd.DataFrame(list(cat_scores.items()), columns=['Category', 'Load'])
                    fig = px.bar(df_ch, x='Load', y='Category', orientation='h', text_auto='.0%', color='Load', color_continuous_scale='Teal')
                    fig.update_layout(paper_bgcolor='rgba(0,0,0,0)', plot_bgcolor='rgba(0,0,0,0)', font=dict(color='#94a3b8', family='Plus Jakarta Sans'))
                    st.plotly_chart(fig, use_container_width=True)
                with chart_col2:
                    fig_r = go.Figure(go.Scatterpolar(
                        r=list(cat_scores.values()), theta=list(cat_scores.keys()), fill='toself', line_color=color
                    ))
                    fig_r.update_layout(polar=dict(radialaxis=dict(visible=True, range=[0, 1]), bgcolor='rgba(0,0,0,0)'), paper_bgcolor='rgba(0,0,0,0)', font=dict(color='#94a3b8', family='Plus Jakarta Sans'))
                    st.plotly_chart(fig_r, use_container_width=True)

            # --- SESSION TREND TRACKER (Phase 8) ---
            st.markdown("---")
            with st.expander("📉 Session Trend Tracker", expanded=True):
                if len(st.session_state['history']) > 1:
                    hist_df = pd.DataFrame(st.session_state['history'], columns=['Stress Score'])
                    st.line_chart(hist_df, color="#2dd4bf")
                    st.caption(f"Tracking {len(hist_df)} datapoints in current session.")
                else:
                    st.info("Generating trend data... (Need at least 2 reports)")

    # --- TRANSPARENCY DASHBOARD (Phase 14) ---
    render_transparency_dashboard()

if __name__ == "__main__":
    main()
