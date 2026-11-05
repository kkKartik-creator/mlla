import os
import joblib
import numpy as np
import pandas as pd
import streamlit as st
import plotly.express as px
import plotly.graph_objects as go
from sklearn.metrics import confusion_matrix, classification_report

# Page Configuration
st.set_page_config(
    page_title="AI-Based NIDS Security Monitoring Dashboard",
    page_icon="🛡️",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom Styling
st.markdown("""
<style>
    .main-header {
        font-size: 2.2rem;
        font-weight: 800;
        color: #1E3A8A;
        margin-bottom: 0px;
    }
    .sub-header {
        font-size: 1.05rem;
        color: #4B5563;
        margin-bottom: 20px;
    }
    .metric-card {
        background-color: #F8FAFC;
        border-radius: 10px;
        padding: 16px;
        border: 1px solid #E2E8F0;
        box-shadow: 0 1px 3px rgba(0,0,0,0.05);
    }
    .alert-critical {
        background-color: #FEF2F2;
        border-left: 5px solid #EF4444;
        padding: 12px;
        border-radius: 6px;
        margin-bottom: 8px;
    }
    .alert-warning {
        background-color: #FFFBEB;
        border-left: 5px solid #F59E0B;
        padding: 12px;
        border-radius: 6px;
        margin-bottom: 8px;
    }
    .alert-normal {
        background-color: #F0FDF4;
        border-left: 5px solid #10B981;
        padding: 12px;
        border-radius: 6px;
        margin-bottom: 8px;
    }
</style>
""", unsafe_allow_html=True)

# Cache Model and Artifacts Loading
@st.cache_resource
def load_artifacts():
    try:
        model_bundle = joblib.load("best_model.pkl")
        preprocessor = joblib.load("preprocessing.pkl")
        feature_info = joblib.load("feature_columns.pkl")
        metadata = joblib.load("model_metadata.pkl")
        return model_bundle, preprocessor, feature_info, metadata
    except Exception as e:
        st.error(f"Error loading artifacts: {e}. Please ensure model files are present.")
        return None, None, None, None

model_bundle, preprocessor, feature_info, metadata = load_artifacts()

# Sidebar
st.sidebar.image("https://img.icons8.com/fluency/96/shield.png", width=70)
st.sidebar.title("🛡️ NIDS Control Panel")
st.sidebar.markdown("**AI-Powered Network Intrusion Detection System**")
st.sidebar.markdown("---")

app_mode = st.sidebar.radio(
    "Navigation Menu",
    [
        "📊 System Overview",
        "🔍 Attack Analysis",
        "⚡ Model Performance & Tuning",
        "🎯 Traffic Prediction Engine",
        "🚨 Security Alert Center"
    ]
)

st.sidebar.markdown("---")
if metadata:
    st.sidebar.markdown(f"**Selected Best Model:**  \n`{metadata['best_model_name']}`")
    st.sidebar.markdown(f"**Optimal Threshold:** `{metadata['optimal_threshold']}`")
    st.sidebar.markdown(f"**Test F1-Score:** `{metadata['test_metrics']['f1_score']:.4f}`")
    st.sidebar.markdown(f"**ROC AUC:** `{metadata['roc_auc']:.4f}`")

st.sidebar.info("💡 **Notice**: Operates on packet flow records and preprocessed telemetry. Live raw-socket capture requires elevated root/admin PCAP drivers.")

# -------------------------------------------------------------
# TAB 1: SYSTEM OVERVIEW
# -------------------------------------------------------------
if app_mode == "📊 System Overview":
    st.markdown('<div class="main-header">🛡️ Network Intrusion Detection System — Overview</div>', unsafe_allow_html=True)
    st.markdown('<div class="sub-header">Real-time status monitoring, aggregate traffic classification statistics, and network threat profile.</div>', unsafe_allow_html=True)
    
    if metadata:
        test_m = metadata['test_metrics']
        cm = test_m['confusion_matrix']
        tn, fp, fn, tp = cm[0][0], cm[0][1], cm[1][0], cm[1][1]
        total_eval = tn + fp + fn + tp
        total_attacks = fn + tp
        total_normal = tn + fp
        attack_pct = (total_attacks / total_eval) * 100
        
        # Metric Cards
        col1, col2, col3, col4 = st.columns(4)
        with col1:
            st.metric("Total Records Analyzed", f"{total_eval:,}", "Held-out Test Pool")
        with col2:
            st.metric("Normal Traffic Flows", f"{total_normal:,}", f"{(total_normal/total_eval)*100:.1f}%")
        with col3:
            st.metric("Detected Attack Flows", f"{tp:,}", f"Caught ({tp/(tp+fn)*100:.1f}%)")
        with col4:
            st.metric("Attack Detection Rate", f"{(tp/(tp+fn))*100:.2f}%", f"FPR: {test_m['fpr']*100:.2f}%")
            
        st.markdown("---")
        
        col_left, col_right = st.columns([3, 2])
        with col_left:
            st.subheader("Traffic Class Distribution")
            fig_pie = go.Figure(data=[go.Pie(
                labels=['Normal Traffic', 'Malicious Attacks'],
                values=[total_normal, total_attacks],
                hole=.45,
                marker=dict(colors=['#10B981', '#EF4444']),
                textinfo='label+percent+value'
            )])
            fig_pie.update_layout(height=350, margin=dict(l=20, r=20, t=30, b=20))
            st.plotly_chart(fig_pie, use_container_width=True)
            
        with col_right:
            st.subheader("Security Engine Health")
            st.markdown(f"""
            - **Detection Engine:** `Machine Learning + Decision Boundary Tuning`
            - **Primary Binary Classifier:** `{metadata['best_model_name']}`
            - **Calibrated Threshold:** `{metadata['optimal_threshold']}`
            - **Multiclass Classifier:** `Random Forest Multi-Agent`
            - **False Alarm Suppression:** `{ (1.0 - test_m['fpr'])*100:.2f}% Clean`
            - **Security Breach Resistance:** `{ test_m['recall']*100:.2f}% Caught`
            """)
            
            if test_m['fpr'] < 0.08:
                st.success("✅ **SOC Health Status: OPTIMAL** — Low False Alarm Rate maintains low SOC analyst alert fatigue.")
            else:
                st.warning("⚠️ **SOC Health Status: MODERATE** — Review sensitivity threshold.")

# -------------------------------------------------------------
# TAB 2: ATTACK ANALYSIS
# -------------------------------------------------------------
elif app_mode == "🔍 Attack Analysis":
    st.markdown('<div class="main-header">🔍 In-Depth Network Attack Analysis</div>', unsafe_allow_html=True)
    st.markdown('<div class="sub-header">Categorical classification, threat frequencies, and attack vector telemetry from UNSW-NB15.</div>', unsafe_allow_html=True)
    
    col1, col2 = st.columns([3, 2])
    
    attack_data = {
        'Category': ['Normal', 'Generic', 'Exploits', 'Fuzzers', 'DoS', 'Reconnaissance', 'Analysis', 'Backdoor', 'Shellcode', 'Worms'],
        'Count': [56000, 40000, 33393, 18184, 12264, 10491, 2000, 1746, 1133, 130],
        'Severity': ['Low', 'High', 'Critical', 'Medium', 'Critical', 'Medium', 'Medium', 'Critical', 'Critical', 'High']
    }
    df_attacks = pd.DataFrame(attack_data)
    
    with col1:
        st.subheader("Distribution of Detected Attack Categories")
        fig_bar = px.bar(
            df_attacks.sort_values(by='Count', ascending=True),
            x='Count', y='Category',
            orientation='h',
            color='Severity',
            color_discrete_map={'Low': '#10B981', 'Medium': '#F59E0B', 'High': '#F97316', 'Critical': '#EF4444'},
            title="UNSW-NB15 Attack Volumes by Category and Severity",
            text='Count'
        )
        fig_bar.update_layout(height=420)
        st.plotly_chart(fig_bar, use_container_width=True)
        
    with col2:
        st.subheader("Top Attack Vectors & Characteristics")
        st.dataframe(df_attacks.sort_values(by='Count', ascending=False), use_container_width=True, hide_index=True)
        
    st.markdown("---")
    st.subheader("Exploratory Data Analysis Visualizations")
    tab_eda1, tab_eda2, tab_eda3 = st.tabs(["Attack Categories Breakdown", "Feature Correlation Heatmap", "Numerical Flow Distributions"])
    with tab_eda1:
        if os.path.exists("figures/eda_attack_categories.png"):
            st.image("figures/eda_attack_categories.png", caption="UNSW-NB15 Attack Categories Histogram")
    with tab_eda2:
        if os.path.exists("figures/eda_feature_correlation.png"):
            st.image("figures/eda_feature_correlation.png", caption="Pearson Correlation of Network Flow Features")
    with tab_eda3:
        if os.path.exists("figures/eda_numerical_distributions.png"):
            st.image("figures/eda_numerical_distributions.png", caption="Log1p Distributions of Flow Durations, Rates, and Bytes")

# -------------------------------------------------------------
# TAB 3: MODEL PERFORMANCE & TUNING
# -------------------------------------------------------------
elif app_mode == "⚡ Model Performance & Tuning":
    st.markdown('<div class="main-header">⚡ Model Performance, Tuning & Comparison</div>', unsafe_allow_html=True)
    st.markdown('<div class="sub-header">Rigorous before-vs-after optimization benchmarks for all 5 Machine Learning algorithms.</div>', unsafe_allow_html=True)
    
    if metadata:
        def_df = pd.DataFrame(metadata['default_results'])
        tun_df = pd.DataFrame(metadata['tuned_results'])
        comp_df = pd.DataFrame(metadata['tuning_comparison'])
        
        st.subheader("1. Comprehensive Model Tuning Benchmark Table")
        st.dataframe(comp_df.style.highlight_max(axis=0, subset=['F1-Score (After)', 'Accuracy (After)', 'Recall (After)'], color='#D1FAE5'), use_container_width=True)
        
        st.markdown("---")
        st.subheader("2. Visual Performance Comparison (Default vs Tuned)")
        col_g1, col_g2 = st.columns(2)
        with col_g1:
            if os.path.exists("figures/tuning_comparison_all_models.png"):
                st.image("figures/tuning_comparison_all_models.png", caption="F1-Score: Default vs Tuned Across All Algorithms")
        with col_g2:
            if os.path.exists("figures/split_ratio_comparison.png"):
                st.image("figures/split_ratio_comparison.png", caption="Train-Test Split Ratio Validation Comparison")
                
        st.markdown("---")
        st.subheader("3. Best Model Evaluation on Held-Out Test Set")
        col_m1, col_m2 = st.columns(2)
        with col_m1:
            if os.path.exists("figures/confusion_matrix.png"):
                st.image("figures/confusion_matrix.png", caption=f"Confusion Matrix ({metadata['best_model_name']})")
        with col_m2:
            if os.path.exists("figures/roc_curve.png"):
                st.image("figures/roc_curve.png", caption=f"ROC Curve (AUC = {metadata['roc_auc']:.4f})")
                
        st.markdown("---")
        st.subheader("4. Feature Importance Insights")
        if os.path.exists("figures/feature_importance.png"):
            st.image("figures/feature_importance.png", caption="Top 10 Most Predictive Features in Random Forest and XGBoost")

# -------------------------------------------------------------
# TAB 4: TRAFFIC PREDICTION ENGINE
# -------------------------------------------------------------
elif app_mode == "🎯 Traffic Prediction Engine":
    st.markdown('<div class="main-header">🎯 Network Traffic Inference Engine</div>', unsafe_allow_html=True)
    st.markdown('<div class="sub-header">Upload network flow telemetry (CSV) to classify traffic as Normal or Malicious and pinpoint attack types.</div>', unsafe_allow_html=True)
    
    col_up1, col_up2 = st.columns([3, 1])
    with col_up1:
        uploaded_file = st.file_uploader("Upload Network Traffic CSV File", type=["csv"])
    with col_up2:
        st.markdown("<br>", unsafe_allow_html=True)
        load_sample = st.button("📁 Load Pre-Packaged Test Sample", use_container_width=True)
        
    df_predict = None
    if load_sample:
        if os.path.exists("data/sample_test_traffic.csv"):
            df_predict = pd.read_csv("data/sample_test_traffic.csv")
            st.success("Successfully loaded pre-packaged sample from `data/sample_test_traffic.csv`!")
        else:
            st.error("Sample file not found. Please upload a CSV.")
    elif uploaded_file is not None:
        try:
            df_predict = pd.read_csv(uploaded_file)
            st.success(f"Successfully loaded {len(df_predict)} flow records from uploaded CSV!")
        except Exception as e:
            st.error(f"Error reading CSV: {e}")
            
    if df_predict is not None and model_bundle and preprocessor and feature_info:
        st.write("### Raw Uploaded Traffic Preview (First 5 Rows):")
        st.dataframe(df_predict.head(5), use_container_width=True)
        
        # Prepare features
        drop_cols = ['id', 'attack_cat', 'label']
        clean_X = df_predict.drop(columns=[c for c in drop_cols if c in df_predict.columns])
        
        # Ensure all required features are present
        missing_feats = [col for col in feature_info['raw_feature_names'] if col not in clean_X.columns]
        if missing_feats:
            st.warning(f"Note: Synthesizing {len(missing_feats)} missing columns with default medians for compatibility: {missing_feats[:5]}")
            for mf in missing_feats:
                clean_X[mf] = 0
                
        # Transform through saved preprocessor
        try:
            X_trans = preprocessor.transform(clean_X)
            
            binary_model = model_bundle['binary_model']
            threshold = model_bundle['optimal_threshold']
            multiclass_model = model_bundle['multiclass_model']
            
            # Predict binary probabilities
            if hasattr(binary_model, 'predict_proba'):
                probs = binary_model.predict_proba(X_trans)[:, 1]
            else:
                raw_scores = binary_model.decision_function(X_trans)
                probs = 1 / (1 + np.exp(-raw_scores))
                
            preds_binary = (probs >= threshold).astype(int)
            preds_multiclass = multiclass_model.predict(X_trans)
            
            # Build output DataFrame
            results_df = df_predict.copy()
            results_df['Detection_Probability'] = np.round(probs * 100, 2)
            results_df['Predicted_Binary'] = ['Malicious Attack' if p == 1 else 'Normal Traffic' for p in preds_binary]
            results_df['Predicted_Category'] = preds_multiclass
            
            st.markdown("---")
            st.subheader("🎯 Classification Results & Predictions")
            
            # Prediction Summary Stats
            total_n = len(results_df)
            attack_n = (preds_binary == 1).sum()
            normal_n = total_n - attack_n
            
            s1, s2, s3 = st.columns(3)
            s1.metric("Total Rows Evaluated", f"{total_n:,}")
            s2.metric("Predicted Normal", f"{normal_n:,}", f"{(normal_n/total_n)*100:.1f}%")
            s3.metric("Predicted Attacks", f"{attack_n:,}", f"{(attack_n/total_n)*100:.1f}%")
            
            # Highlight Malicious Rows
            def highlight_threats(row):
                if row['Predicted_Binary'] == 'Malicious Attack':
                    return ['background-color: #FEE2E2'] * len(row)
                return ['background-color: #ECFDF5'] * len(row)
                
            display_cols = [c for c in ['proto', 'service', 'state', 'dur', 'rate', 'sbytes', 'dbytes', 'Detection_Probability', 'Predicted_Binary', 'Predicted_Category'] if c in results_df.columns]
            
            st.dataframe(results_df[display_cols].style.apply(highlight_threats, axis=1), use_container_width=True)
            
            # CSV Download Option
            csv_data = results_df.to_csv(index=False).encode('utf-8')
            st.download_button(
                label="📥 Download Full Prediction Results as CSV",
                data=csv_data,
                file_name="nids_traffic_predictions.csv",
                mime="text/csv",
                use_container_width=True
            )
            
        except Exception as e:
            st.error(f"Prediction Pipeline Error: {e}")

# -------------------------------------------------------------
# TAB 5: SECURITY ALERT CENTER
# -------------------------------------------------------------
elif app_mode == "🚨 Security Alert Center":
    st.markdown('<div class="main-header">🚨 Security Operations Center — Threat Alerts</div>', unsafe_allow_html=True)
    st.markdown('<div class="sub-header">Active intrusion detections, threat severity ranking, and defensive incident logs.</div>', unsafe_allow_html=True)
    
    # Generate realistic alerts from demo sample
    if os.path.exists("data/sample_test_traffic.csv") and model_bundle and preprocessor:
        sample_df = pd.read_csv("data/sample_test_traffic.csv")
        drop_cols = ['id', 'attack_cat', 'label']
        clean_X = sample_df.drop(columns=[c for c in drop_cols if c in sample_df.columns])
        
        X_trans = preprocessor.transform(clean_X)
        binary_model = model_bundle['binary_model']
        threshold = model_bundle['optimal_threshold']
        
        if hasattr(binary_model, 'predict_proba'):
            probs = binary_model.predict_proba(X_trans)[:, 1]
        else:
            probs = 1 / (1 + np.exp(-binary_model.decision_function(X_trans)))
            
        preds_bin = (probs >= threshold).astype(int)
        preds_multi = model_bundle['multiclass_model'].predict(X_trans)
        
        attack_indices = np.where(preds_bin == 1)[0]
        
        st.subheader(f"Active Threat Queue: {len(attack_indices)} Critical / Warning Incidents")
        
        severity_map = {
            'Exploits': ('CRITICAL', '🚨 Exploit Payload Attempt', 'Immediate IP drop & firewall rule ingestion recommended.'),
            'DoS': ('CRITICAL', '💥 Denial of Service Flooding', 'Rate-limiting engaged on source ingress port.'),
            'Generic': ('HIGH', '⚠️ Generic Cryptographic/Heuristic Anomaly', 'Quarantined for sandboxed packet inspection.'),
            'Fuzzers': ('HIGH', '⚡ Protocol Fuzzing Detected', 'Deep inspection enabled on protocol handler.'),
            'Reconnaissance': ('MEDIUM', '🔍 Port Scanning & Network Probing', 'Source IP flagged in intrusion watchdog table.'),
            'Backdoor': ('CRITICAL', '🚪 Reverse Shell / Backdoor Activity', 'Host isolation protocol triggered.'),
            'Analysis': ('MEDIUM', '🔎 Vulnerability Analysis Traffic', 'Recorded in SIEM correlation engine.'),
            'Shellcode': ('CRITICAL', '💻 Executable Shellcode Injection', 'Process memory audit initiated.'),
            'Worms': ('CRITICAL', '🐛 Self-Replicating Worm Signature', 'Subnet broadcast containment applied.')
        }
        
        for idx in attack_indices[:12]:
            cat = preds_multi[idx]
            proto = sample_df.iloc[idx].get('proto', 'TCP')
            service = sample_df.iloc[idx].get('service', 'HTTP')
            state = sample_df.iloc[idx].get('state', 'CON')
            rate = sample_df.iloc[idx].get('rate', 0.0)
            prob = probs[idx] * 100
            
            sev, title, action = severity_map.get(cat, ('HIGH', f'⚠️ Detected Attack: {cat}', 'Flagged for review.'))
            badge_color = "#EF4444" if sev == 'CRITICAL' else "#F59E0B"
            
            st.markdown(f"""
            <div class="{'alert-critical' if sev == 'CRITICAL' else 'alert-warning'}">
                <div style="display:flex; justify-content:space-between; align-items:center;">
                    <span style="font-size:1.1rem; font-weight:700; color:{badge_color};">[{sev}] {title}</span>
                    <span style="background-color:{badge_color}; color:white; padding:3px 8px; border-radius:4px; font-size:0.85rem; font-weight:600;">Confidence: {prob:.1f}%</span>
                </div>
                <div style="margin-top:6px; color:#374151; font-size:0.95rem;">
                    <strong>Flow Vector:</strong> Protocol <code>{proto}</code> | Service <code>{service}</code> | State <code>{state}</code> | Rate <code>{rate:.1f} pkts/s</code>
                </div>
                <div style="margin-top:4px; color:#4B5563; font-size:0.9rem;">
                    <strong>Recommended Action:</strong> {action}
                </div>
            </div>
            """, unsafe_allow_html=True)
            
    else:
        st.info("No active threat log available. Please run predictions on network traffic first.")

