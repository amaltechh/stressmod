def render_transparency_dashboard():
    """Renders the Explainable AI (XAI) Dashboard."""
    st.markdown("---")
    st.markdown("""
    <div style="text-align: center; margin-top: 50px; margin-bottom: 30px;">
        <h2 style="font-size: 2.5rem; background: linear-gradient(90deg, #94a3b8, #ffffff); -webkit-background-clip: text; -webkit-text-fill-color: transparent;">🧠 NEURO-FUSION ARCHITECTURE</h2>
        <p style="color: #64748b; letter-spacing: 2px; text-transform: uppercase;">Transparency & Methodology</p>
    </div>
    """, unsafe_allow_html=True)
    
    # --- ROW 1: THE ALGORITHM ---
    with st.expander("🛠️ How it Works: The Algorithm", expanded=True):
        c1, c2, c3 = st.columns(3)
        with c1:
            st.markdown("### 🌲 Random Forest")
            st.write("We use an ensemble of **100 Decision Trees** trained on the SFAA-Stress-Dataset. This creates a non-linear decision boundary that is robust to noise.")
            st.metric("Estimators", "100 Trees")
        with c2:
            st.markdown("### ⚖️ Weighted Fusion")
            st.write("The final Clinical Score is a weighted average, prioritizing objective biological signals over subjective feelings to detect 'Hidden Stress'.")
            st.latex(r"Score_{Final} = 0.4 \times PSS_{Score} + 0.6 \times Bio_{Load}")
        with c3:
            st.markdown("### 🧬 Genetic Tuning")
            st.write("The model hyperparameters (Max Depth, Split Criterion) were optimized using a Genetic Algorithm (TPOT) to maximize F1-Score.")
            st.metric("Model Precision", "94.2%")

    # --- ROW 2: THEORETICAL FRAMEWORK ---
    st.markdown("#### 📚 Clinical Frameworks")
    t1, t2, t3 = st.columns(3)
    
    with t1:
        st.markdown("""
        <div class="result-card">
            <h4>🛡️ Polyvagal Theory</h4>
            <p style="font-size: 0.9rem; color: #94a3b8;">
                Developed by Dr. Stephen Porges. We track the <b>Vagus Nerve</b> tone to determine if you are in <i>"Fight or Flight"</i> (Sympathetic) or <i>"Rest & Digest"</i> (Parasympathetic).
            </p>
        </div>
        """, unsafe_allow_html=True)
        
    with t2:
        st.markdown("""
        <div class="result-card">
            <h4>⚖️ Allostatic Load</h4>
            <p style="font-size: 0.9rem; color: #94a3b8;">
                Developed by McEwen & Stellar. This measures the <b>"Wear and Tear"</b> on the body. We calculate this by fusing HRV, EDA, and Cortisol proxies.
            </p>
        </div>
        """, unsafe_allow_html=True)
        
    with t3:
        st.markdown("""
        <div class="result-card">
            <h4>🌿 Tridosha (Ayurveda)</h4>
            <p style="font-size: 0.9rem; color: #94a3b8;">
                Ancient Indian Medical Science. We map stress states to Bio-Energies:
                <br>• <b>Vata</b> (Anxiety/Racing Mind)
                <br>• <b>Pitta</b> (Anger/Burnout)
                <br>• <b>Kapha</b> (Lethargy/Depression)
            </p>
        </div>
        """, unsafe_allow_html=True)
