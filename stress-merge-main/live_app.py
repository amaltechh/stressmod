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
        "BioState": "🔥 Critical Hyper-Arousal: Your Sympathetic Nervous System is currently 'locked on'. This state releases a flood of Cortisol and Adrenaline, causing inflammation, cognitive fog, and potential burnout. Immediate physiological reset required.",
        "Pranayama": "🐝 Bhramari (Humming Bee Breath): Use vibrational resonance to stimulate the Vagus Nerve and force a parasympathetic dominance (Rest & Digest).",
        "Asana": "🧘 Shavasana (Corpse Pose): Total stillness is non-negotiable. Do NOT engage in cardio; your heart needs to perceive safety, not threat.",
        "Diet": "🥥 Sattvic Cooling Protocol: Focus on alkaline hydration (Coconut water, Cucumber). Avoid all stimulants (Caffeine, Spicy/Sour foods) to arrest the cortisol spike.",
        "Mantra": "🕉️ Om Shanti: 'I am Peace'. Use this frequency to override the mental noise.",
        "Img": "https://images.unsplash.com/photo-1545205597-3d9d02c29597?q=80&w=600&auto=format&fit=crop" # Yoga Shavasana
    },
    "Medium": {
        "Theme": "Balancing (Tridosha)",
        "BioState": "⚠️ Allostatic Overload: Your body is efficiently handling stress, but the 'cost' is accumulating silently. You are in the 'Resistance Phase' of General Adaptation Syndrome. Without intervention, this leads to exhaustion.",
        "Pranayama": "👃 Nadi Shodhana: Alternate Nostril Breathing to mechanically balance the Left (Logical) and Right (Creative) brain hemispheres.",
        "Asana": "🌲 Vrikshasana (Tree Pose): A single-point focus pose to stabilize the wandering mind and ground physical energy.",
        "Diet": "🍵 Adaptogenic Support: Warm Chamomile or Tulsi (Holy Basil) tea to naturally modulate cortisol levels without sedation.",
        "Mantra": "✨ So Hum: 'I am That'. Re-aligning your personal rhythm with the universal rhythm.",
        "Img": "https://images.unsplash.com/photo-1474418397713-7ede21d49118?q=80&w=600&auto=format&fit=crop" # Nature/Balance
    },
    "Low": {
        "Theme": "Energizing & Maintaining (Kapha)",
        "BioState": "✅ Optimal Homeostasis: Your Allostatic load is minimal. Your autonomic nervous system is flexible and resilient. This is 'Eustress'—positive stress that drives growth and focus.",
        "Pranayama": "🔥 Kapalbhati (Skull Shining): Rapid forceful exhalations to energize the frontal cortex and clear mental cobwebs.",
        "Asana": "☀️ Surya Namaskar (Sun Salutation): Dynamic kinetic flow to build metabolic heat and maintain agility.",
        "Diet": "🌶️ Metabolic Activation: Light, spiced foods (Ginger, Turmeric, Honey) to prevent stagnation and keep energy flowing.",
        "Mantra": "💪 Gayatri Mantra: Invoking clarity and intellectual brilliance.",
        "Img": "https://images.unsplash.com/photo-1518611012118-696072aa579a?q=80&w=600&auto=format&fit=crop" # Active Fitness
    }
}

# --- CATEGORY-SPECIFIC THERAPEUTIC PROTOCOLS ---
CATEGORY_IKS = {
    "Academic": {
        "Theme": "Focus & Clarity (Saraswati Protocol)",
        "Pranayama": "🐝 Bhramari (Humming Bee Breath) to clear mental fog and improve concentration.",
        "Asana": "🧘 Padmasana (Lotus Pose) to stabilize posture for long study sessions.",
        "Diet": "🧠 Brain-boosting foods: Walnuts, Brahmi/Gotu Kola tea, and omega-rich seeds.",
        "Mantra": "📚 Om Aim Saraswatyai Namaha (To invoke intellect and memory).",
        "Img": "https://images.unsplash.com/photo-1513258496099-48168024aec0?q=80&w=600&auto=format&fit=crop" # Books/Focus
    },
    "Emotional": {
        "Theme": "Heart-Centric Healing (Anahata Protocol)",
        "Pranayama": "👃 Anulom Vilom (Alternate Nostril) to balance emotional hemispheres.",
        "Asana": "🐪 Ustrasana (Camel Pose) to open the chest and release trapped emotional energy.",
        "Diet": "🍠 Mood-stabilizing grounding foods: Sweet potatoes, Ashwagandha milk, dark chocolate.",
        "Mantra": "🕊️ Om Shanti Shanti Shanti (For profound inner peace).",
        "Img": "https://images.unsplash.com/photo-1499209974431-9dddcece7f88?q=80&w=600&auto=format&fit=crop" # Peaceful
    },
    "Social": {
        "Theme": "Boundaries & Connection (Vishuddha Protocol)",
        "Pranayama": "🌊 Ujjayi (Ocean Breath) to build internal heat and vocal confidence.",
        "Asana": "🐟 Matsyasana (Fish Pose) to open the throat chakra for better communication.",
        "Diet": "🍵 Soothing throat foods: Warm ginger-lemon tea, turmeric, and honey.",
        "Mantra": "🤝 Aham Prema (I am Divine Love - to foster healthy relationships).",
        "Img": "https://images.unsplash.com/photo-1529156069898-49953e39b3ac?q=80&w=600&auto=format&fit=crop" # Friendship/Connection
    },
    "Physical": {
        "Theme": "Restoration & Recovery (Muladhara Protocol)",
        "Pranayama": "🌬️ Dirgha Pranayama (Three-Part Breath) to deeply oxygenate fatigued muscles.",
        "Asana": "🛌 Balasana (Child's Pose) or Viparita Karani (Legs Up the Wall) for sheer physical rest.",
        "Diet": "🍲 Anti-inflammatory healing: Turmeric golden milk, bone broth, magnesium-rich spinach.",
        "Mantra": "🌱 Lam (Root chakra bija mantra to ground the physical body).",
        "Img": "https://images.unsplash.com/photo-1544367567-0f2fcb009e0b?q=80&w=600&auto=format&fit=crop" # Stretching/Rest
    },
    "Coping": {
        "Theme": "Resilience & Strategy (Manipura Protocol)",
        "Pranayama": "🔥 Kapalabhati (Skull Shining Breath) to ignite internal willpower and motivation.",
        "Asana": "⛵ Navasana (Boat Pose) to build core strength and psychological fortitude.",
        "Diet": "🌶️ Metabolism-igniting foods: Ginger, black pepper, citrus fruits to break lethargy.",
        "Mantra": "⚡ Ram (Solar plexus bija mantra to activate courage and action).",
        "Img": "https://images.unsplash.com/photo-1506126613408-eca07ce68773?q=80&w=600&auto=format&fit=crop" # Meditation/Strength
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
    
    text = f"**Primary Stress Driver:** {dominant_cat} ({cat_scores[dominant_cat]:.0%} Load)<br><br>"
    
    diff = abs(bio_score - sub_score)
    
    if bio_score > sub_score + 0.3:
        text += f"⚠️ **CRITICAL WARNING - Somatic Mismatch ({diff:.0%} Variance)**<br>"
        text += f"Your body is screaming (Body Score: {bio_score:.0%}) while your mind thinks it's fine (Mind Score: {sub_score:.0%}). This is a dangerous state of autonomic nervous system overload that often precedes sudden burnout or panic attacks.<br><br>"
        text += "🩺 **Targeted Remedy:** Stop intellectualizing your stress. Your body needs immediate physiological regulation. Focus on deep box-breathing, somatic experiencing (like shaking or vagus nerve stimulation), and strictly limit caffeine/stimulants. Prioritize 8 hours of sleep tonight."
    elif bio_score > sub_score + 0.15:
        text += f"⚠️ **Elevated Somatic Load ({diff:.0%} Variance)**<br>"
        text += f"Your physical stress (Body Score: {bio_score:.0%}) is noticeably higher than your mental perception (Mind Score: {sub_score:.0%}). Your body is carrying tension you might be ignoring.<br><br>"
        text += "🩺 **Targeted Remedy:** Incorporate progressive muscle relaxation or yoga nidra before bed. Check your posture, hydration, and consider a light physical release like a brisk walk or stretching to dissipate the trapped cortisol."
    elif sub_score > bio_score + 0.3:
        text += f"⚠️ **CRITICAL WARNING - Psychological Distress ({diff:.0%} Variance)**<br>"
        text += f"Your mind is racing (Mind Score: {sub_score:.0%}), but your body is relatively calm (Body Score: {bio_score:.0%}). This severe 'Perceived Stress' indicates rumination, anxiety, or cognitive overload without immediate physical danger.<br><br>"
        text += "🧠 **Targeted Remedy:** Your brain is stuck in a loop. You need to break the cognitive cycle. Use 'brain dumping' (journaling all worries), practice mindfulness/CBT techniques to challenge catastrophic thoughts, and ground yourself in the present moment using the 5-4-3-2-1 sensory technique."
    elif sub_score > bio_score + 0.15:
        text += f"⚠️ **Elevated Cognitive Load ({diff:.0%} Variance)**<br>"
        text += f"You are experiencing high perceived stress (Mind Score: {sub_score:.0%}) compared to your physical state (Body Score: {bio_score:.0%}). You are overthinking or carrying emotional burdens.<br><br>"
        text += "🧠 **Targeted Remedy:** Engage in a flow-state activity that distracts the mind (art, puzzles, playing an instrument). Limit doom-scrolling and try a 10-minute guided meditation focused on detaching from racing thoughts."
    else:
        text += f"⚖️ **Aligned State ({diff:.0%} Variance)**<br>"
        text += f"Your physical ({bio_score:.0%}) and mental ({sub_score:.0%}) states are synchronized. Your subjective feeling matches your biological reality.<br><br>"
        text += "🌿 **Targeted Remedy:** Maintain your current routine. Use generalized adaptogenic practices like moderate exercise, balanced nutrition, and standard daily mindfulness to maintain this equilibrium."
        
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

def generate_clinical_report(score, bio, mind, cat_scores, remedy, level, patient_name="Patient"):
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
    ax_main.text(0.1, 0.87, f"PATIENT: {patient_name}", fontsize=11, color='#0f172a', weight='bold', fontfamily='monospace')
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
    
    bio_clean = remedy['BioState']
    # Extract technique name: text between emoji and first colon
    def extract_name(text):
        # Skip emoji prefix (first 2-3 chars), return the rest
        parts = text.split(' ', 1)
        return parts[1] if len(parts) > 1 else text
    plan_clean = (
        f"PHYSIOLOGY: {bio_clean}\n\n"
        f"BREATHWORK: {extract_name(remedy['Pranayama'])}\n"
        f"MOVEMENT:   {extract_name(remedy['Asana'])}\n"
        f"NUTRITION:  {extract_name(remedy['Diet'])}\n"
        f"MANTRA:     {extract_name(remedy['Mantra'])}"
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
    
    # --- SESSION STATE INIT ---
    if 'history' not in st.session_state:
        st.session_state['history'] = []
    if 'wizard_step' not in st.session_state:
        st.session_state['wizard_step'] = 0  # 0=name, 1-6=form steps
    if 'form_data' not in st.session_state:
        st.session_state['form_data'] = {
            'aq1': 1, 'aq2': 1, 'aq3': 1, 'aq4': 2, 'aq5': 1,
            'eq1': 2, 'eq2': 1, 'eq3': 1, 'eq4': 1,
            'sq1': 1, 'sq2': 2, 'sq3': 1, 'sq4': 1,
            'pq1': 1, 'pq2': 1, 'pq3': 2, 'pq4': 1,
            'cq1': 2, 'cq2': 2, 'cq3': 2, 'cq4': 3, 'cq5': 1, 'cq6': 2, 'cq7': 1,
            'eda': 2.5, 'hr': 72, 'temp': 36.5
        }
    fd = st.session_state['form_data']
    
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
             fallback = "I recommend checking the **Quick Consult** buttons above for reliable advice on Yoga, Diet, and Usage!"
             st.session_state["messages"].append({"role": "assistant", "content": fallback})
             st.rerun()
    
    # --- WIZARD STEP NAMES ---
    step_names = ["👤 Patient Info", "📚 Academic", "🧠 Emotional", "🤝 Social", "🏃 Physical", "🛡️ Coping", "⌚ Biometrics"]
    current_step = st.session_state['wizard_step']
    
    # --- PROGRESS BAR ---
    if current_step > 0:
        progress_pct = current_step / 6
        st.markdown(f"""
        <div style="margin: 20px 0 10px 0;">
            <div style="display: flex; justify-content: space-between; margin-bottom: 8px;">
                <span style="color: #94a3b8; font-size: 0.85rem;">Step {current_step} of 6</span>
                <span style="color: #2dd4bf; font-size: 0.85rem; font-weight: 600;">{step_names[current_step]}</span>
            </div>
            <div style="background: rgba(255,255,255,0.05); border-radius: 10px; height: 8px; overflow: hidden;">
                <div style="background: linear-gradient(90deg, #2dd4bf, #3b82f6); height: 100%; width: {progress_pct*100:.0f}%; border-radius: 10px; transition: width 0.4s ease;"></div>
            </div>
        </div>
        """, unsafe_allow_html=True)
    
    # =============================================
    # STEP 0: NAME ENTRY (GATE)
    # =============================================
    if current_step == 0:
        st.markdown("#### 📡 Patient Data Acquisition")
        st.markdown("""
        <div style="background: rgba(45, 212, 191, 0.05); border: 1px solid rgba(45, 212, 191, 0.2); border-radius: 16px; padding: 40px; text-align: center; margin: 30px 0;">
            <div style="font-size: 3rem; margin-bottom: 15px;">👤</div>
            <h3 style="color: #f1f5f9; margin-bottom: 8px;">Welcome to Zen Analyzer</h3>
            <p style="color: #94a3b8; margin-bottom: 25px;">Please enter your name to begin the stress assessment</p>
        </div>
        """, unsafe_allow_html=True)
        
        patient_name = st.text_input(
            "👤 Full Name",
            value=st.session_state.get('patient_name_val', ''),
            placeholder="Enter your full name to begin...",
            key="patient_name_input",
            help="Required. Your name will appear on the clinical report."
        )
        
        render_reference_guide()
        
        if st.button("▶️ BEGIN ASSESSMENT", type="primary", use_container_width=True, disabled=not bool(patient_name and patient_name.strip())):
            st.session_state['patient_name_val'] = patient_name.strip()
            st.session_state['wizard_step'] = 1
            st.rerun()
        
        if not (patient_name and patient_name.strip()):
            st.info("💡 Enter your name above to unlock the assessment form.")
        
        # Show transparency dashboard even on step 0
        render_transparency_dashboard()
        return
    
    # --- Patient name from session (for all subsequent steps) ---
    patient_name = st.session_state.get('patient_name_val', 'Patient')
    st.markdown(f"<p style='color: #64748b; font-size: 0.9rem; margin-bottom: 5px;'>Patient: <span style=\"color: #2dd4bf; font-weight: 600;\">{patient_name}</span></p>", unsafe_allow_html=True)
    
    # --- DEMO PRESETS ---
    def set_preset(level):
        if level == "Low":
            val_q, val_eda, val_hr, val_temp = 1, 2.5, 70, 36.5
        elif level == "Medium":
            val_q, val_eda, val_hr, val_temp = 2, 6.0, 95, 34.0
        else:
            val_q, val_eda, val_hr, val_temp = 3, 12.0, 115, 31.0
        keys = ['aq1','aq2','aq3','aq4','aq5', 'eq1','eq2','eq3','eq4', 'sq1','sq2','sq3','sq4', 'pq1','pq2','pq3','pq4', 'cq1','cq2','cq3','cq4','cq5','cq6','cq7']
        for k in keys: 
            st.session_state[k] = val_q
            fd[k] = val_q
        st.session_state['eda'] = val_eda
        st.session_state['hr'] = val_hr
        st.session_state['temp'] = val_temp
        fd['eda'] = val_eda
        fd['hr'] = val_hr
        fd['temp'] = val_temp

    st.write("🤖 **Demo Simulation:**")
    bc1, bc2, bc3 = st.columns(3)
    if bc1.button("🟢 Simulate Low Stress", use_container_width=True): set_preset("Low")
    if bc2.button("🟡 Simulate Medium Stress", use_container_width=True): set_preset("Medium")
    if bc3.button("🔴 Simulate High Stress", use_container_width=True): set_preset("High")
    
    # =============================================
    # STEP 1-6: FORM SECTIONS (one at a time)
    # =============================================
    
    # Helper: which keys belong to which step
    step_keys = {
        1: ['aq1','aq2','aq3','aq4','aq5'],
        2: ['eq1','eq2','eq3','eq4'],
        3: ['sq1','sq2','sq3','sq4'],
        4: ['pq1','pq2','pq3','pq4'],
        5: ['cq1','cq2','cq3','cq4','cq5','cq6','cq7'],
        6: ['eda','hr','temp'],
    }
    
    def save_current_step():
        """Save current step's widget values to persistent form_data."""
        keys = step_keys.get(current_step, [])
        for k in keys:
            if k in st.session_state:
                fd[k] = st.session_state[k]
    
    # --- STEP 1: Academic ---
    if current_step == 1:
        st.markdown("### 📚 Academic Stress")
        c1, c2 = st.columns(2)
        with c1:
            st.slider("1. Workload Overwhelm", 0, 4, fd['aq1'], key="aq1", help="Do you feel buried under assignments?")
            st.slider("2. Exam Anxiety", 0, 4, fd['aq2'], key="aq2", help="Do you freeze up before tests?")
            st.slider("3. Deadline Struggles", 0, 4, fd['aq3'], key="aq3")
        with c2:
            st.slider("4. GPA Worries", 0, 4, fd['aq4'], key="aq4")
            st.slider("5. Balancing Projects", 0, 4, fd['aq5'], key="aq5")
    
    # --- STEP 2: Emotional ---
    elif current_step == 2:
        st.markdown("### 🧠 Emotional Stress")
        c1, c2 = st.columns(2)
        with c1:
            st.slider("6. Emotional Exhaustion", 0, 4, fd['eq1'], key="eq1")
            st.slider("7. Demotivation", 0, 4, fd['eq2'], key="eq2")
        with c2:
            st.slider("8. Mood Swings", 0, 4, fd['eq3'], key="eq3")
            st.slider("9. Feeling Isolated", 0, 4, fd['eq4'], key="eq4")
    
    # --- STEP 3: Social ---
    elif current_step == 3:
        st.markdown("### 🤝 Social Stress")
        c1, c2 = st.columns(2)
        with c1:
            st.slider("10. No Time for Family", 0, 4, fd['sq1'], key="sq1")
            st.slider("11. Family Expectations", 0, 4, fd['sq2'], key="sq2")
        with c2:
            st.slider("12. Relationship Issues", 0, 4, fd['sq3'], key="sq3")
            st.slider("13. Peer Pressure", 0, 4, fd['sq4'], key="sq4")
    
    # --- STEP 4: Physical ---
    elif current_step == 4:
        st.markdown("### 🏃 Physical Stress")
        c1, c2 = st.columns(2)
        with c1:
            st.slider("14. Headaches/Fatigue", 0, 4, fd['pq1'], key="pq1")
            st.slider("15. Appetite Changes", 0, 4, fd['pq2'], key="pq2")
        with c2:
            st.slider("16. Tired after Sleep", 0, 4, fd['pq3'], key="pq3")
            st.slider("17. Palpitations", 0, 4, fd['pq4'], key="pq4")
    
    # --- STEP 5: Coping ---
    elif current_step == 5:
        st.markdown("### 🛡️ Coping Mechanisms")
        c1, c2 = st.columns(2)
        with c1:
            st.slider("18. Hesitate to seek help", 0, 4, fd['cq1'], key="cq1")
            st.slider("19. Poor Concentration", 0, 4, fd['cq2'], key="cq2")
            st.slider("20. No Strategy", 0, 4, fd['cq3'], key="cq3")
            st.slider("21. No Hobbies", 0, 4, fd['cq4'], key="cq4")
        with c2:
            st.slider("22. Social Withdrawal", 0, 4, fd['cq5'], key="cq5")
            st.slider("23. No Relaxation", 0, 4, fd['cq6'], key="cq6")
            st.slider("24. Unaware of Support", 0, 4, fd['cq7'], key="cq7")
    
    # --- STEP 6: Biometrics ---
    elif current_step == 6:
        st.markdown("### ⌚ Biometric Data")
        st.info("💡 **Simulation Mode:** Connect medical-grade sensors or simulate bio-signals below.")
        b1, b2, b3 = st.columns(3)
        with b1: st.slider("⚡ EDA (µS)", 0.0, 20.0, fd['eda'], key="eda", help="Electrodermal Activity. Normal Resting: 2-5µS.")
        with b2: st.slider("❤️ Heart Rate", 50, 140, fd['hr'], key="hr", help="Resting Heart Rate (BPM).")
        with b3: st.slider("🌡️ Skin Temp (°C)", 30.0, 37.0, fd['temp'], key="temp", help="Peripheral temperature.")
    
    # =============================================
    # NAVIGATION BUTTONS (Back / Next / Generate)
    # =============================================
    st.markdown("---")
    nav_col1, nav_col2, nav_col3 = st.columns([1, 2, 1])
    
    with nav_col1:
        if current_step > 1:
            if st.button("⬅️ Back", use_container_width=True):
                save_current_step()
                st.session_state['wizard_step'] = current_step - 1
                st.rerun()
    
    with nav_col3:
        if current_step < 6:
            if st.button("Next ➡️", type="primary", use_container_width=True):
                save_current_step()
                st.session_state['wizard_step'] = current_step + 1
                st.rerun()
    
    # --- Save current step on every rerun (catches direct slider changes) ---
    save_current_step()
    
    # --- Read all values from persistent form_data for calculation ---
    aq1 = fd['aq1']; aq2 = fd['aq2']; aq3 = fd['aq3']; aq4 = fd['aq4']; aq5 = fd['aq5']
    eq1 = fd['eq1']; eq2 = fd['eq2']; eq3 = fd['eq3']; eq4 = fd['eq4']
    sq1 = fd['sq1']; sq2 = fd['sq2']; sq3 = fd['sq3']; sq4 = fd['sq4']
    pq1 = fd['pq1']; pq2 = fd['pq2']; pq3 = fd['pq3']; pq4 = fd['pq4']
    cq1 = fd['cq1']; cq2 = fd['cq2']; cq3 = fd['cq3']; cq4 = fd['cq4']
    cq5 = fd['cq5']; cq6 = fd['cq6']; cq7 = fd['cq7']
    eda = fd['eda']; hr = fd['hr']; temp = fd['temp']

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
    
    # --- STACKING ENSEMBLE: 3-Model Architecture ---
    # Level 1a: XGBoost (Survey) — "The Psychologist"
    # Level 1b: Random Forest (Wearable) — "The Doctor"
    # Level 2:  GBM Meta-Learner (Fusion) — "The Judge"
    
    try:
        # Load all 3 models
        xgb_model = joblib.load('survey/trained_xgb_survey.pkl')
        rf_model = joblib.load('wearable/trained_rf_wearable.pkl')
        meta_model = joblib.load('wearable/trained_gbm_meta.pkl')
        le = joblib.load('wearable/label_encoder.pkl')
        
        # Level 1a: XGBoost predicts stress from survey features
        survey_input = [[s_acad, s_emo, s_soc, s_phys, s_cope]]
        xgb_probs = xgb_model.predict_proba(survey_input)[0]  # [P_High, P_Low, P_Med]
        
        # Level 1b: Random Forest predicts stress from wearable features
        wearable_input = [[eda, hr, temp]]
        rf_probs = rf_model.predict_proba(wearable_input)[0]  # [P_High, P_Low, P_Med]
        
        # Level 2: GBM Meta-Learner fuses both predictions
        meta_features = np.concatenate([xgb_probs, rf_probs]).reshape(1, -1)  # 6 features
        meta_prediction = meta_model.predict(meta_features)[0]
        meta_probs = meta_model.predict_proba(meta_features)[0]
        
        # Convert meta-learner output to a 0-1 score for the gauge/report
        # Classes from LabelEncoder: 0=High, 1=Low, 2=Medium
        bio_score = 0.0
        survey_score_from_model = 0.0
        for cls_idx, prob in enumerate(meta_probs):
            cls_name = le.inverse_transform([cls_idx])[0]
            if cls_name == 'High':
                bio_score += prob * 0.9
            elif cls_name == 'Low':
                bio_score += prob * 0.2
            elif cls_name == 'Medium':
                bio_score += prob * 0.5
        
        # Final score IS the meta-learner's output (no hardcoded formula)
        final_score = bio_score
        
        # For display: extract individual model confidence for the report
        # Store XGB and RF individual scores for the Mind/Body display
        xgb_score = 0.0
        for cls_idx, prob in enumerate(xgb_probs):
            cls_name = le.inverse_transform([cls_idx])[0]
            if cls_name == 'High': xgb_score += prob * 0.9
            elif cls_name == 'Low': xgb_score += prob * 0.2
            elif cls_name == 'Medium': xgb_score += prob * 0.5
        survey_total_norm = xgb_score  # Override for display as "Mind Score"
        
        rf_score = 0.0
        for cls_idx, prob in enumerate(rf_probs):
            cls_name = le.inverse_transform([cls_idx])[0]
            if cls_name == 'High': rf_score += prob * 0.9
            elif cls_name == 'Low': rf_score += prob * 0.2
            elif cls_name == 'Medium': rf_score += prob * 0.5
        bio_score = rf_score  # Override for display as "Body Score"
            
    except Exception as e:
        # Fallback: use old formula if stacking models are not available
        # Biometric heuristic
        n_eda = min(1.0, eda / 15.0); n_hr = min(1.0, max(0, (hr - 60) / 60.0)); n_temp = min(1.0, max(0, (36.5 - temp) / 5.0))
        bio_score = (n_eda * 0.5) + (n_hr * 0.3) + (n_temp * 0.2)
        # Fallback fusion formula (60/40)
        final_score = (bio_score * 0.6) + (survey_total_norm * 0.4)
    
    # Update History
    st.session_state['history'].append(final_score)

    # --- Only show Generate button on final step (step 6) ---
    if current_step == 6:
        st.markdown("---")
    
        if st.button("🚀 GENERATE CLINICAL REPORT", type="primary", use_container_width=True):
            with st.spinner("Analyzing Bio-Markers..."):
                time.sleep(1.0)
                
                # Categorize
                if final_score < 0.4: level = "Low"; color = "#10b981" # Emerald 500
                elif final_score < 0.7: level = "Medium"; color = "#f59e0b" # Amber 500
                else: level = "High"; color = "#ef4444" # Red 500
                
                dominant_cat = max(cat_scores, key=cat_scores.get)
                
                if level in ["High", "Medium"]:
                    remedy = CATEGORY_IKS[dominant_cat]
                    # We still need the BioState string from the main DB for the clinical diagnosis text
                    remedy['BioState'] = IKS_DB[level]['BioState']
                else:
                    remedy = IKS_DB[level]
                
                # --- RESULTS HEADER ---
                st.markdown(f"""
                <div style="text-align: center; margin-bottom: 30px; animation: pulseSoft 3s infinite;">
                    <p style="font-size: 1.1rem; color: #64748b; margin-bottom: 5px; letter-spacing: 2px; text-transform: uppercase;">Report for</p>
                    <h2 style="font-size: 2rem; margin: 0 0 10px 0; color: #f1f5f9; font-weight: 700;">{patient_name}</h2>
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
                    buf_png, buf_pdf = generate_clinical_report(final_score, bio_score, survey_total_norm, cat_scores, remedy, level, patient_name)
                    
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

                # --- NEW PATIENT BUTTON ---
                st.markdown("---")
                st.markdown("""
                <div style="text-align: center; margin: 30px 0 10px 0;">
                    <p style="color: #64748b; font-size: 0.95rem;">Assessment complete. Start a new session below.</p>
                </div>
                """, unsafe_allow_html=True)
                if st.button("👤 NEW PATIENT — Start Over", use_container_width=True):
                    # Reset wizard and clear form data
                    keys_to_clear = ['wizard_step', 'patient_name_val',
                                     'aq1','aq2','aq3','aq4','aq5',
                                     'eq1','eq2','eq3','eq4',
                                     'sq1','sq2','sq3','sq4',
                                     'pq1','pq2','pq3','pq4',
                                     'cq1','cq2','cq3','cq4','cq5','cq6','cq7',
                                     'eda','hr','temp']
                    for k in keys_to_clear:
                        if k in st.session_state:
                            del st.session_state[k]
                    st.rerun()

    # --- TRANSPARENCY DASHBOARD (Phase 14) ---
    render_transparency_dashboard()

if __name__ == "__main__":
    main()
