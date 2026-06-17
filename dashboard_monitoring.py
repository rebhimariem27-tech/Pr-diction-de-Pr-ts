import streamlit as st
import plotly.express as px
import plotly.graph_objects as go
import pandas as pd
from logger_decisions import charger_log
from drift_detector import detecter_drift, verifier_distribution_decisions
from session_manager import est_authentifie, deconnecter
from page_login import afficher_login

if not est_authentifie():
    afficher_login()
    st.stop()
st.set_page_config(
    page_title="FinPredict — Monitoring",
    layout="wide",
    page_icon="📊"
)

st.markdown("""
    <style>
    [data-testid="stAppViewContainer"] { background-color: #0F172A; color: #F8FAFC; }
    h1, h2, h3 { color: #F8FAFC !important; }
    .metric-card {
        background-color: #1E293B; border-radius: 12px;
        padding: 20px; margin-bottom: 15px;
        border-left: 5px solid #06B6D4;
    }
    </style>
""", unsafe_allow_html=True)

st.markdown("# 📊 Dashboard Monitoring — FinPredict AI Pro")
st.caption("Surveillance en temps réel du modèle et des données")

log = charger_log()

# ── MÉTRIQUES GLOBALES ──────────────────────────────────────────
st.markdown("## Vue globale")

if len(log) == 0:
    st.info("Aucune décision loggée. Lancez quelques analyses dans app_web.py.")
    st.stop()

col1, col2, col3, col4 = st.columns(4)

with col1:
    st.metric("Total analyses", len(log))
with col2:
    nb_approuves = (log['decision'] == 'APPROUVÉ').sum()
    st.metric("Approuvés", nb_approuves,
              delta=f"{nb_approuves/len(log)*100:.1f}%")
with col3:
    st.metric("Score ML moyen", f"{log['score_ml'].mean():.1f}%")
with col4:
    st.metric("Score final moyen", f"{log['score_final'].mean():.1f}%")

st.divider()

# ── DISTRIBUTION DES DÉCISIONS ──────────────────────────────────
st.markdown("## Distribution des décisions")

col_g1, col_g2 = st.columns(2)

with col_g1:
    counts = log['decision'].value_counts().reset_index()
    counts.columns = ['Decision', 'Count']
    colors_map = {
        'APPROUVÉ':  "#ACFFA2",
        'À RÉVISER': "#660083",
        'REJETÉ':    "#D8AFAF"
    }
    fig_pie = px.pie(
        counts, values='Count', names='Decision',
        color='Decision', color_discrete_map=colors_map,
        hole=0.6, title="Répartition des décisions"
    )
    fig_pie.update_layout(
        paper_bgcolor='rgba(0,0,0,0)',
        font_color='white', title_font_color='white'
    )
    st.plotly_chart(fig_pie, use_container_width=True)

with col_g2:
    fig_hist = px.histogram(
        log, x='score_final', nbins=20,
        color='decision', color_discrete_map=colors_map,
        title="Distribution des scores finaux",
        labels={'score_final': 'Score final (%)', 'count': 'Nombre'}
    )
    fig_hist.update_layout(
        paper_bgcolor='rgba(0,0,0,0)',
        plot_bgcolor='rgba(0,0,0,0)',
        font_color='white', title_font_color='white'
    )
    st.plotly_chart(fig_hist, use_container_width=True)

st.divider()

# ── ÉVOLUTION TEMPORELLE ────────────────────────────────────────
st.markdown("## Évolution dans le temps")

if 'timestamp' in log.columns and len(log) >= 2:
    log_sorted = log.sort_values('timestamp')
    fig_time = px.line(
        log_sorted, x='timestamp', y='score_final',
        color='decision', color_discrete_map=colors_map,
        title="Score final au fil du temps",
        markers=True
    )
    fig_time.update_layout(
        paper_bgcolor='rgba(0,0,0,0)',
        plot_bgcolor='rgba(0,0,0,0)',
        font_color='white', title_font_color='white'
    )
    st.plotly_chart(fig_time, use_container_width=True)

st.divider()

# ── DRIFT DETECTION ─────────────────────────────────────────────
st.markdown("## 🔍 Détection de Drift")

drift = detecter_drift()

if drift['statut'] == 'INSUFFISANT':
    st.warning(drift['message'])
elif drift['statut'] == 'ALERTE':
    for alerte in drift['alertes']:
        st.error(alerte)
else:
    st.success("✅ Aucun drift détecté — les données récentes sont cohérentes.")

if 'rapport' in drift and drift['rapport']:
    rapport_df = pd.DataFrame(drift['rapport']).T.reset_index()
    rapport_df.columns = [
        'Feature', 'Moy. référence',
        'Moy. récente', 'Déviation %', 'En drift'
    ]
    rapport_df['En drift'] = rapport_df['En drift'].map(
        {True: '⚠️ OUI', False: '✅ NON'}
    )
    st.dataframe(rapport_df, use_container_width=True, hide_index=True)

st.divider()

# ── DISTRIBUTION DECISIONS ──────────────────────────────────────
dist = verifier_distribution_decisions(log)
if dist.get('alerte'):
    st.error(dist['alerte'])

st.markdown("## 📋 Dernières décisions")
st.dataframe(
    log.sort_values('timestamp', ascending=False).head(20),
    use_container_width=True,
    hide_index=True
)