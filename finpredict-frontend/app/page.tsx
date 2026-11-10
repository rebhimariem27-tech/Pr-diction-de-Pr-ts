'use client';

import { useCallback, useEffect, useMemo, useState } from 'react';
import {
  Activity, AlertCircle, ArrowDownRight, ArrowLeft, ArrowRight, ArrowUpRight,
  BadgeCheck, Bell, BookOpen, BrainCircuit, CalendarDays, Check, CheckCircle2,
  ChevronDown, ChevronLeft, ChevronRight, CircleHelp, ClipboardCheck, Clock3,
  Cloud, Cpu, Database, Download, FileBarChart2, FileCheck2, FileClock, FileText,
  Fingerprint, Gauge, GaugeCircle, House, Info, LayoutDashboard, LockKeyhole,
  LogOut, Menu, MoreHorizontal, Plus, RefreshCw, Search, Settings, Shield,
  ShieldCheck, SlidersHorizontal, Sparkles, TrendingDown, TrendingUp, UserRound,
  UsersRound, WalletCards, X, Zap,
} from 'lucide-react';
import {
  CartesianGrid, Cell, Line, LineChart, Pie, PieChart, ResponsiveContainer,
  Tooltip, XAxis, YAxis,
} from 'recharts';

type View = 'dashboard' | 'analysis' | 'portfolio' | 'monitoring' | 'reports' | 'settings';
type Prediction = {
  application_id: number;
  application_ref: string;
  prediction_id: number;
  model_version: string;
  default_probability: number;
  threshold: number;
  recommendation: 'ACCEPT' | 'REFUSE';
  tracking_storage: string;
  derived_features?: Record<string, number | null>;
  explanation?: { top_risk_factors?: Factor[]; top_protective_factors?: Factor[] };
};
type Factor = { feature: string; value: string | number | null; contribution: number; effect: string };
const featureLabels: Record<string, string> = {
  person_age: 'Âge', person_income: 'Revenu annuel', loan_amnt: 'Montant du prêt', loan_int_rate: 'Taux d’intérêt',
  person_emp_length: 'Ancienneté professionnelle', loan_percent_income: 'Part du revenu consacrée au prêt',
  cb_person_cred_hist_length: 'Historique de crédit', interest_income_ratio: 'Ratio intérêts / revenu',
  person_home_ownership: 'Logement', loan_intent: 'Objet du prêt', loan_grade: 'Note du prêt', cb_person_default_on_file: 'Défaut antérieur',
};
type FormData = {
  person_age: string;
  person_income: string;
  loan_amnt: string;
  loan_int_rate: string;
  person_emp_length: string;
  cb_person_cred_hist_length: string;
  person_home_ownership: string;
  loan_intent: string;
  loan_grade: string;
  cb_person_default_on_file: string;
};

const API_BASE = (process.env.NEXT_PUBLIC_API_BASE_URL || 'http://127.0.0.1:8000').replace(/\/$/, '');
const navItems: { id: View; label: string; icon: typeof LayoutDashboard }[] = [
  { id: 'dashboard', label: 'Tableau de bord', icon: LayoutDashboard },
  { id: 'analysis', label: 'Analyse de crédit', icon: FileBarChart2 },
  { id: 'portfolio', label: 'Portefeuille', icon: WalletCards },
  { id: 'monitoring', label: 'Surveillance', icon: Activity },
  { id: 'reports', label: 'Rapports', icon: ClipboardCheck },
  { id: 'settings', label: 'Paramètres', icon: Settings },
];

const demoForm: FormData = {
  person_age: '32',
  person_income: '48000',
  loan_amnt: '10000',
  loan_int_rate: '11.14',
  person_emp_length: '5',
  cb_person_cred_hist_length: '8',
  person_home_ownership: 'RENT',
  loan_intent: 'PERSONAL',
  loan_grade: 'B',
  cb_person_default_on_file: 'N',
};

const monthlyData = [
  { month: 'Jan', rate: 10.1 }, { month: 'Fév', rate: 10.3 }, { month: 'Mar', rate: 9.0 },
  { month: 'Avr', rate: 8.8 }, { month: 'Mai', rate: 7.7 }, { month: 'Juin', rate: 7.1 },
  { month: 'Juil', rate: 6.3 }, { month: 'Août', rate: 6.2 }, { month: 'Sep', rate: 5.7 },
  { month: 'Oct', rate: 5.8 },
];
const pieData = [
  { name: 'Approuvés', value: 87.5, color: '#21A675' },
  { name: 'Rejetés', value: 12.4, color: '#F36D78' },
  { name: 'En attente', value: 0.1, color: '#F5B84D' },
];
const portfolioRows = [
  { ref: 'FP-2026-0148', name: 'Demande #0148', amount: '12 500 €', probability: 0.08, decision: 'ACCEPT', date: '10 oct. 2026' },
  { ref: 'FP-2026-0147', name: 'Demande #0147', amount: '7 800 €', probability: 0.62, decision: 'REFUSE', date: '10 oct. 2026' },
  { ref: 'FP-2026-0146', name: 'Demande #0146', amount: '21 000 €', probability: 0.31, decision: 'ACCEPT', date: '09 oct. 2026' },
  { ref: 'FP-2026-0145', name: 'Demande #0145', amount: '4 500 €', probability: 0.14, decision: 'ACCEPT', date: '09 oct. 2026' },
];

function Logo() {
  return <div className="brand"><div className="brand-mark"><span /><span /><span /></div><div><strong>FinPredict</strong><small>AI for Smarter Lending</small></div></div>;
}

function Badge({ children, kind = 'neutral' }: { children: React.ReactNode; kind?: 'neutral' | 'success' | 'danger' | 'warning' | 'blue' }) {
  return <span className={`badge badge-${kind}`}><span className="badge-dot" />{children}</span>;
}

function KpiCard({ icon: Icon, label, value, trend, sub, tone }: { icon: typeof FileText; label: string; value: string; trend: string; sub: string; tone: string }) {
  const positive = trend.trim().startsWith('+');
  return <article className="kpi-card card"><div className={`kpi-icon ${tone}`}><Icon size={19} strokeWidth={2.2} /></div><div className="kpi-content"><span className="eyebrow">{label}</span><div className="kpi-value">{value}</div><div className="kpi-foot"><span className={positive ? 'trend-up' : 'trend-down'}>{positive ? <ArrowUpRight size={14} /> : <ArrowDownRight size={14} />}{trend}</span><span className="muted">{sub}</span></div></div></article>;
}

function Sidebar({ active, onNavigate, mobileOpen, onClose }: { active: View; onNavigate: (v: View) => void; mobileOpen: boolean; onClose: () => void }) {
  return <>
    {mobileOpen && <button className="mobile-scrim" aria-label="Fermer le menu" onClick={onClose} />}
    <aside className={`sidebar ${mobileOpen ? 'sidebar-open' : ''}`}>
      <Logo />
      <div className="sidebar-section-label">ESPACE DE TRAVAIL</div>
      <nav className="primary-nav" aria-label="Navigation principale">
        {navItems.map(({ id, label, icon: Icon }) => <button key={id} onClick={() => { onNavigate(id); onClose(); }} className={`nav-item ${active === id ? 'active' : ''}`}><Icon size={18} /><span>{label}</span>{id === 'monitoring' && <span className="nav-pulse" />}</button>)}
      </nav>
      <div className="sidebar-bottom-art" aria-hidden="true"><svg viewBox="0 0 230 130" preserveAspectRatio="none"><path d="M-8 110 C30 35 52 150 96 66 S160 30 238 3"/><path d="M-8 122 C30 47 52 162 96 78 S160 42 238 15"/><path d="M-8 134 C30 59 52 174 96 90 S160 54 238 27"/><path d="M-8 98 C30 23 52 138 96 54 S160 18 238 -9"/></svg></div>
      <div className="sidebar-user"><div className="avatar avatar-user"><UserRound size={18} /></div><div className="user-info"><strong>Mariem Rebhi</strong><span>Analyste risque crédit</span></div><ChevronRight size={15} /></div>
      <div className="sidebar-footer"><span className="online-dot" />Environnement de démonstration</div>
    </aside>
  </>;
}

function Topbar({ title, onMenu }: { title: string; onMenu: () => void }) {
  const [search, setSearch] = useState('');
  return <header className="topbar"><button className="mobile-menu icon-button" onClick={onMenu} aria-label="Ouvrir le menu"><Menu size={20} /></button><div className="breadcrumb"><span>FinPredict</span><ChevronRight size={14} /><strong>{title}</strong></div><div className="topbar-actions"><div className="global-search"><Search size={16} /><input value={search} onChange={e => setSearch(e.target.value)} placeholder="Rechercher…" aria-label="Rechercher" />{search && <button onClick={() => setSearch('')} aria-label="Effacer"><X size={14} /></button>}<kbd>⌘ K</kbd></div><button className="icon-button notification" aria-label="Notifications"><Bell size={18} /><span /></button><div className="topbar-divider" /><button className="profile-button" aria-label="Profil"><div className="avatar avatar-top">MR</div><ChevronDown size={14} /></button></div></header>;
}

function PageHeader({ eyebrow, title, description, right }: { eyebrow?: string; title: string; description: string; right?: React.ReactNode }) {
  return <div className="page-header"><div>{eyebrow && <div className="page-eyebrow">{eyebrow}</div>}<h1>{title}</h1><p>{description}</p></div>{right && <div className="page-header-right">{right}</div>}</div>;
}

function DemoNotice() {
  return <div className="demo-notice"><Info size={15} /><span><strong>Mode démonstration</strong> — les indicateurs du tableau de bord sont illustratifs et ne proviennent pas encore d’Azure SQL.</span></div>;
}

function Dashboard({ onNewAnalysis, apiStatus, modelVersion }: { onNewAnalysis: () => void; apiStatus: string; modelVersion: string }) {
  return <>
    <DemoNotice />
    <PageHeader eyebrow="VUE D’ENSEMBLE" title="Bonjour Mariem 👋" description="Voici un aperçu de votre portefeuille et des analyses de risque." right={<><button className="button button-secondary"><CalendarDays size={16} /> 30 derniers jours <ChevronDown size={14} /></button><button className="button button-primary" onClick={onNewAnalysis}><Plus size={17} /> Nouvelle analyse</button></>} />
    <div className="kpi-grid">
      <KpiCard icon={FileText} label="Prêts analysés" value="12 482" trend="+12 %" sub="vs. mois dernier" tone="tone-blue" />
      <KpiCard icon={ShieldCheck} label="Taux de défaut (global)" value="5,8 %" trend="−1,2 %" sub="vs. mois dernier" tone="tone-teal" />
      <KpiCard icon={CheckCircle2} label="Prêts approuvés" value="10 932" trend="+15 %" sub="vs. mois dernier" tone="tone-green" />
      <KpiCard icon={AlertCircle} label="Prêts refusés" value="1 550" trend="−8 %" sub="vs. mois dernier" tone="tone-red" />
    </div>
    <div className="dashboard-grid main-charts">
      <article className="card chart-card trend-card"><div className="card-heading"><div><h2>Évolution du taux de défaut</h2><p>Indicateur de risque sur les 10 derniers mois</p></div><button className="select-compact">10 mois <ChevronDown size={13} /></button></div><div className="chart-area line-chart-area"><ResponsiveContainer width="100%" height="100%"><LineChart data={monthlyData} margin={{ top: 15, right: 12, bottom: 0, left: -18 }}><CartesianGrid vertical={false} stroke="#E9EEF5" strokeDasharray="3 4" /><XAxis dataKey="month" axisLine={false} tickLine={false} tick={{ fill: '#8391A7', fontSize: 11 }} dy={8} /><YAxis domain={[0, 20]} ticks={[0, 5, 10, 15, 20]} axisLine={false} tickLine={false} tick={{ fill: '#8391A7', fontSize: 11 }} tickFormatter={v => `${v}%`} /><Tooltip contentStyle={{ border: '1px solid #E6ECF4', borderRadius: 12, boxShadow: '0 8px 24px rgba(18,42,75,.08)' }} formatter={(value: number) => [`${value}%`, 'Taux de défaut']} /><Line type="monotone" dataKey="rate" stroke="#4C6FFF" strokeWidth={2.6} dot={{ r: 3.5, strokeWidth: 2, fill: '#fff', stroke: '#4C6FFF' }} activeDot={{ r: 5 }} /></LineChart></ResponsiveContainer><div className="line-chart-label">5,8 % <span>Taux actuel</span></div></div></article>
      <article className="card chart-card decisions-card"><div className="card-heading"><div><h2>Répartition des décisions</h2><p>Ventilation illustrative</p></div><button className="icon-button subtle"><MoreHorizontal size={19} /></button></div><div className="donut-layout"><div className="donut-wrap"><ResponsiveContainer width="100%" height="100%"><PieChart><Pie data={pieData} dataKey="value" nameKey="name" innerRadius="68%" outerRadius="92%" startAngle={90} endAngle={-270} paddingAngle={2} stroke="none">{pieData.map((entry, idx) => <Cell key={idx} fill={entry.color} />)}</Pie></PieChart></ResponsiveContainer><div className="donut-center"><strong>12 482</strong><span>dossiers</span></div></div><div className="legend-list">{pieData.map(row => <div className="legend-row" key={row.name}><span className="legend-dot" style={{ background: row.color }} /><span>{row.name}</span><strong>{row.value.toLocaleString('fr-FR', { maximumFractionDigits: 1 })}%</strong></div>)}</div></div></article>
      <article className="ai-feature-card"><div className="ai-feature-background" /><div className="ai-feature-content"><div className="feature-label"><Sparkles size={15} /> IA & CONFIANCE</div><h2>Des décisions plus éclairées, grâce à l’IA.</h2><p>Évaluez le risque de défaut avec un modèle calibré et des données cohérentes, tout en gardant la décision finale sous contrôle humain.</p><button className="button button-dark" onClick={onNewAnalysis}>Lancer une analyse <ArrowRight size={15} /></button></div><div className="feature-orb orb-one" /><div className="feature-orb orb-two" /></article>
    </div>
    <div className="status-grid">
      <article className="card mini-status"><div className="mini-status-icon green"><ShieldCheck size={17} /></div><div><span className="eyebrow">Modèle de production</span><strong>Version {modelVersion}</strong><small>Seuil opérationnel : 0,48</small></div><Badge kind="success">Chargé</Badge></article>
      <article className="card mini-status"><div className="mini-status-icon blue"><BrainCircuit size={17} /></div><div><span className="eyebrow">API FastAPI</span><strong>{apiStatus === 'online' ? 'Connectée' : apiStatus === 'checking' ? 'Vérification…' : 'Non connectée'}</strong><small>{apiStatus === 'online' ? 'Endpoint disponible' : 'Vérifiez le serveur local / cloud'}</small></div><Badge kind={apiStatus === 'online' ? 'success' : apiStatus === 'checking' ? 'warning' : 'danger'}>{apiStatus === 'online' ? 'En ligne' : apiStatus === 'checking' ? 'Test' : 'Hors ligne'}</Badge></article>
      <article className="card mini-status"><div className="mini-status-icon teal"><Database size={17} /></div><div><span className="eyebrow">Persistance des résultats</span><strong>Azure SQL</strong><small>Tracking des demandes et prédictions</small></div><Badge kind="blue">Cloud</Badge></article>
    </div>
  </>;
}

function Field({ label, hint, children }: { label: string; hint?: string; children: React.ReactNode }) {
  return <label className="field"><span className="field-label">{label}</span>{children}{hint && <small className="field-hint">{hint}</small>}</label>;
}

function AnalysisView({ apiStatus, modelVersion }: { apiStatus: string; modelVersion: string }) {
  const [form, setForm] = useState<FormData>(demoForm);
  const [prediction, setPrediction] = useState<Prediction | null>(null);
  const [error, setError] = useState('');
  const [loading, setLoading] = useState(false);
  const [showAdvanced, setShowAdvanced] = useState(true);
  const update = (key: keyof FormData, value: string) => {
    setForm(prev => ({ ...prev, [key]: value }));
    setPrediction(null);
    setError('');
  };
  const submit = async (event: React.FormEvent<HTMLFormElement>) => {
    event.preventDefault(); setLoading(true); setError(''); setPrediction(null);
    const numericKeys: (keyof FormData)[] = ['person_age', 'person_income', 'loan_amnt', 'loan_int_rate', 'person_emp_length', 'cb_person_cred_hist_length'];
    const payload: Record<string, string | number | null> = { application_ref: `web-${crypto.randomUUID()}` };
    for (const key of numericKeys) payload[key] = form[key].trim() === '' ? null : Number(form[key]);
    for (const key of ['person_home_ownership', 'loan_intent', 'loan_grade', 'cb_person_default_on_file'] as const) payload[key] = form[key] || null;
    try {
      const response = await fetch('/api/predict', { method: 'POST', headers: { 'Content-Type': 'application/json', Accept: 'application/json' }, body: JSON.stringify(payload) });
      const body = await response.json().catch(() => ({}));
      if (!response.ok) {
        const detail = typeof body.detail === 'string' ? body.detail : Array.isArray(body.detail) ? body.detail.map((d: { msg?: string }) => d.msg).filter(Boolean).join(' ; ') : body.detail?.message || `La requête a échoué (HTTP ${response.status}).`;
        throw new Error(detail);
      }
      setPrediction(body as Prediction);
    } catch (e) {
      const msg = e instanceof Error ? e.message : 'Une erreur inconnue est survenue.';
      if (msg.includes('Failed to fetch') || msg.includes('NetworkError')) {
        setError(`Impossible de joindre l’API à ${API_BASE}. Vérifiez qu’elle est démarrée et que CORS autorise l’origine du frontend.`);
      } else setError(msg);
    } finally { setLoading(false); }
  };
  const reset = () => { setForm(demoForm); setPrediction(null); setError(''); };
  const defaultPct = prediction ? prediction.default_probability * 100 : 0;
  return <>
    <PageHeader eyebrow="ANALYSE INDIVIDUELLE" title="Nouvelle analyse de crédit" description="Évaluez le risque de défaut à partir des caractéristiques financières du dossier." right={<Badge kind={apiStatus === 'online' ? 'success' : 'warning'}>{apiStatus === 'online' ? 'API connectée' : 'Mode local / API à vérifier'}</Badge>} />
    <div className="analysis-layout">
      <form onSubmit={submit} className="card application-card">
        <div className="form-intro"><div className="form-intro-icon"><FileCheck2 size={20} /></div><div><h2>Informations du dossier</h2><p>Les champs correspondent au schéma du modèle de production.</p></div><span className="required-note">* Obligatoire</span></div>
        <div className="form-stepper"><div className="step active"><span>1</span><div><strong>Profil</strong><small>Informations demandeur</small></div></div><div className="step-line" /><div className="step active"><span>2</span><div><strong>Financement</strong><small>Montant & revenu</small></div></div><div className="step-line" /><div className={`step ${prediction ? 'active' : ''}`}><span>3</span><div><strong>Résultat</strong><small>Analyse du risque</small></div></div></div>
        <div className="form-section-title"><span className="section-number">01</span><div><h3>Profil du demandeur</h3><p>Situation personnelle et professionnelle</p></div></div>
        <div className="form-grid">
          <Field label="Âge *"><input required type="number" min="18" max="100" value={form.person_age} onChange={e => update('person_age', e.target.value)} /></Field>
          <Field label="Revenu annuel (€) *"><input required type="number" min="1" step="any" value={form.person_income} onChange={e => update('person_income', e.target.value)} /></Field>
          <Field label="Ancienneté professionnelle (années) *"><input required type="number" min="0" step="any" value={form.person_emp_length} onChange={e => update('person_emp_length', e.target.value)} /></Field>
          <Field label="Durée de l’historique de crédit (ans) *"><input required type="number" min="0" value={form.cb_person_cred_hist_length} onChange={e => update('cb_person_cred_hist_length', e.target.value)} /></Field>
          <Field label="Logement *"><select required value={form.person_home_ownership} onChange={e => update('person_home_ownership', e.target.value)}><option value="RENT">Locataire (RENT)</option><option value="OWN">Propriétaire (OWN)</option><option value="MORTGAGE">Crédit immobilier (MORTGAGE)</option><option value="OTHER">Autre (OTHER)</option></select></Field>
          <Field label="Défaut antérieur déclaré *"><select required value={form.cb_person_default_on_file} onChange={e => update('cb_person_default_on_file', e.target.value)}><option value="N">Non (N)</option><option value="Y">Oui (Y)</option></select></Field>
        </div>
        <div className="form-section-title section-spaced"><span className="section-number">02</span><div><h3>Caractéristiques du prêt</h3><p>Paramètres financiers de la demande</p></div></div>
        <div className="form-grid">
          <Field label="Montant demandé (€) *"><input required type="number" min="1" step="any" value={form.loan_amnt} onChange={e => update('loan_amnt', e.target.value)} /></Field>
          <Field label="Taux d’intérêt (%) *"><input required type="number" min="0" step="any" value={form.loan_int_rate} onChange={e => update('loan_int_rate', e.target.value)} /></Field>
          <Field label="Objet du prêt *"><select required value={form.loan_intent} onChange={e => update('loan_intent', e.target.value)}><option value="PERSONAL">Personnel</option><option value="EDUCATION">Éducation</option><option value="MEDICAL">Médical</option><option value="VENTURE">Entrepreneuriat</option><option value="HOMEIMPROVEMENT">Amélioration du logement</option><option value="DEBTCONSOLIDATION">Regroupement de dettes</option></select></Field>
          <Field label="Grade de crédit *"><select required value={form.loan_grade} onChange={e => update('loan_grade', e.target.value)}>{['A', 'B', 'C', 'D', 'E', 'F', 'G'].map(v => <option key={v} value={v}>Grade {v}</option>)}</select></Field>
        </div>
        <button type="button" className="advanced-toggle" onClick={() => setShowAdvanced(v => !v)}><SlidersHorizontal size={15} /> Variables complémentaires du modèle <ChevronDown size={15} className={showAdvanced ? 'rotate' : ''} /></button>
        {showAdvanced && <div className="form-grid advanced-grid">
        </div>}
        <div className="privacy-note"><LockKeyhole size={15} /><span>Ne saisissez pas de nom, de numéro d’identité ou de données directement identifiantes. Les résultats fournis sont une recommandation du modèle, pas une décision finale de crédit.</span></div>
        {error && <div className="error-callout"><AlertCircle size={17} /><div><strong>Analyse non terminée</strong><p>{error}</p></div></div>}
        <div className="form-actions"><button type="button" className="button button-secondary" onClick={reset}><RefreshCw size={15} /> Réinitialiser</button><button type="submit" className="button button-primary" disabled={loading}>{loading ? <><span className="spinner" /> Analyse en cours…</> : <>Analyser le risque <ArrowRight size={16} /></>}</button></div>
      </form>
      <aside className="analysis-aside">
        {prediction ? <article className="card result-card"><div className="result-top"><div className={`result-symbol ${prediction.recommendation === 'ACCEPT' ? 'result-positive' : 'result-negative'}`}>{prediction.recommendation === 'ACCEPT' ? <CheckCircle2 size={24} /> : <AlertCircle size={24} />}</div><div><span className="eyebrow">RÉSULTAT DU MODÈLE</span><h2>{prediction.recommendation === 'ACCEPT' ? 'Risque sous le seuil' : 'Risque au-dessus du seuil'}</h2><p>Recommandation : {prediction.recommendation === 'ACCEPT' ? 'ACCEPT' : 'REFUSE'}</p></div></div><div className="risk-score"><div><span className="eyebrow">Probabilité de défaut</span><strong>{defaultPct.toLocaleString('fr-FR', { maximumFractionDigits: 2 })}<small>%</small></strong></div><div className="risk-meter"><div className="risk-meter-track"><div style={{ width: `${Math.min(defaultPct, 100)}%` }} className={defaultPct >= prediction.threshold * 100 ? 'risk-high' : 'risk-low'} /></div><div className="risk-meter-labels"><span>0 %</span><span>Seuil {Math.round(prediction.threshold * 100)} %</span><span>100 %</span></div></div></div><div className="result-details"><div><span>Référence</span><strong>{prediction.application_ref}</strong></div><div><span>Modèle</span><strong>{prediction.model_version}</strong></div><div><span>ID demande</span><strong>#{prediction.application_id}</strong></div><div><span>ID prédiction</span><strong>#{prediction.prediction_id}</strong></div>{prediction.derived_features ? <><div><span>Part du revenu / prêt</span><strong>{prediction.derived_features.loan_percent_income != null ? (prediction.derived_features.loan_percent_income * 100).toLocaleString('fr-FR', { maximumFractionDigits: 1 }) + ' %' : '—'}</strong></div><div><span>Ratio intérêts / revenu</span><strong>{prediction.derived_features.interest_income_ratio != null ? (prediction.derived_features.interest_income_ratio * 100).toLocaleString('fr-FR', { maximumFractionDigits: 2 }) + ' %' : '—'}</strong></div></> : null}<div><span>Stockage</span><strong><Database size={13} /> {prediction.tracking_storage}</strong></div></div>{prediction.explanation && (prediction.explanation.top_risk_factors?.length || prediction.explanation.top_protective_factors?.length) ? <div className="explain-block"><span className="eyebrow">POURQUOI CE RÉSULTAT</span><div className="explain-cols"><div><strong>Facteurs qui augmentent le risque</strong><ul>{(prediction.explanation.top_risk_factors || []).map(f => <li key={f.feature}>{featureLabels[f.feature] || f.feature}{f.value !== null ? ` (${f.value})` : ''}</li>)}{!(prediction.explanation.top_risk_factors || []).length && <li>Aucun</li>}</ul></div><div><strong>Facteurs qui réduisent le risque</strong><ul>{(prediction.explanation.top_protective_factors || []).map(f => <li key={f.feature}>{featureLabels[f.feature] || f.feature}{f.value !== null ? ` (${f.value})` : ''}</li>)}{!(prediction.explanation.top_protective_factors || []).length && <li>Aucun</li>}</ul></div></div></div> : null}<div className="result-disclaimer"><Info size={15} /> Décision indicative : la décision finale appartient à l’analyste ou au processus de crédit responsable.</div></article> : <article className="card aside-empty"><div className="aside-illustration"><div className="halo halo-a" /><div className="halo halo-b" /><div className="shield-big"><ShieldCheck size={42} /></div><Sparkles className="sparkle s-a" size={19} /><Sparkles className="sparkle s-b" size={14} /></div><span className="eyebrow">ANALYSE EXPLICABLE</span><h2>Votre analyse apparaîtra ici</h2><p>Le modèle estimera la probabilité de défaut, la comparera au seuil de 48 % et enregistrera le résultat dans Azure SQL via FastAPI.</p><div className="aside-checks"><div><CheckCircle2 size={15} /> Pipeline de production calibré</div><div><CheckCircle2 size={15} /> Seuil de décision versionné</div><div><CheckCircle2 size={15} /> Traçabilité du résultat</div></div></article>}
        <article className="card criteria-card"><div className="card-heading"><div><h3>Critères pris en compte</h3><p>12 variables du pipeline</p></div><CircleHelp size={16} className="icon-muted" /></div><div className="criteria-row"><span><WalletCards size={15} /> Capacité de remboursement</span><Check size={15} /></div><div className="criteria-row"><span><FileClock size={15} /> Historique de crédit</span><Check size={15} /></div><div className="criteria-row"><span><House size={15} /> Situation de logement</span><Check size={15} /></div><div className="criteria-row"><span><UsersRound size={15} /> Profil professionnel</span><Check size={15} /></div><div className="criteria-row"><span><LockKeyhole size={15} /> Données protégées</span><Check size={15} /></div><div className="criteria-note">Aucune décision automatique n’est exécutée par cette interface.</div></article>
      </aside>
    </div>
  </>;
}

function PortfolioView() {
  const [query, setQuery] = useState('');
  const filtered = portfolioRows.filter(row => `${row.ref} ${row.name}`.toLowerCase().includes(query.toLowerCase()));
  return <><DemoNotice /><PageHeader eyebrow="SUIVI DES DOSSIERS" title="Portefeuille de crédit" description="Consultez l’activité des demandes et leurs recommandations de risque." right={<button className="button button-primary"><Download size={16} /> Exporter</button>} /><div className="portfolio-stats"><div className="card portfolio-stat"><span className="eyebrow">Demandes de la période</span><strong>248</strong><small><TrendingUp size={13} /> +8,4 % vs. période précédente</small></div><div className="card portfolio-stat"><span className="eyebrow">Montant total analysé</span><strong>1,82 M€</strong><small>Valeur illustrative</small></div><div className="card portfolio-stat"><span className="eyebrow">À examiner</span><strong>14</strong><small><Clock3 size={13} /> En attente de revue humaine</small></div></div><article className="card table-card"><div className="table-header"><div><h2>Demandes récentes</h2><p>Exemples d’affichage — pas des enregistrements Azure SQL</p></div><div className="table-tools"><div className="table-search"><Search size={15} /><input placeholder="Chercher une référence…" value={query} onChange={e => setQuery(e.target.value)} /></div><button className="button button-secondary button-small"><SlidersHorizontal size={15} /> Filtres</button></div></div><div className="table-scroll"><table><thead><tr><th>Référence</th><th>Date</th><th>Montant demandé</th><th>Probabilité de défaut</th><th>Recommandation du modèle</th><th></th></tr></thead><tbody>{filtered.map(row => <tr key={row.ref}><td><strong>{row.ref}</strong><small>{row.name}</small></td><td>{row.date}</td><td className="amount-cell">{row.amount}</td><td><div className="probability-cell"><div className="probability-track"><span style={{ width: `${row.probability * 100}%` }} /></div><strong>{(row.probability * 100).toLocaleString('fr-FR', { maximumFractionDigits: 1 })}%</strong></div></td><td><Badge kind={row.decision === 'ACCEPT' ? 'success' : 'danger'}>{row.decision}</Badge></td><td><button className="icon-button subtle" title="Plus d’informations"><MoreHorizontal size={18} /></button></td></tr>)}</tbody></table>{filtered.length === 0 && <div className="empty-table">Aucun dossier ne correspond à la recherche.</div>}</div><div className="table-footer"><span>Affichage de {filtered.length} dossier(s) de démonstration</span><div className="pagination"><button disabled aria-label="Page précédente"><ChevronLeft size={16} /></button><button className="page-active">1</button><button disabled aria-label="Page suivante"><ChevronRight size={16} /></button></div></div></article></>;
}

function MonitoringView({ apiStatus, modelVersion }: { apiStatus: string; modelVersion: string }) {
  const driftData = [{ day: '04 oct.', v: .12 }, { day: '05 oct.', v: .15 }, { day: '06 oct.', v: .11 }, { day: '07 oct.', v: .18 }, { day: '08 oct.', v: .14 }, { day: '09 oct.', v: .17 }, { day: '10 oct.', v: .16 }];
  return <><DemoNotice /><PageHeader eyebrow="OBSERVABILITÉ & GOUVERNANCE" title="Surveillance du modèle" description="Suivez la disponibilité, la stabilité des données et la santé du pipeline de production." right={<button className="button button-secondary" onClick={() => window.location.reload()}><RefreshCw size={15} /> Actualiser</button>} /><div className="monitor-banner"><div className={`monitor-banner-icon ${apiStatus === 'online' ? 'ok' : 'warn'}`}><Activity size={20} /></div><div><strong>{apiStatus === 'online' ? 'API accessible' : 'API cloud / locale à vérifier'}</strong><p>{apiStatus === 'online' ? `L’API répond à ${API_BASE}.` : `Aucune connexion confirmée à ${API_BASE}. Les métriques de drift ci-dessous sont illustratives.`}</p></div><Badge kind={apiStatus === 'online' ? 'success' : 'warning'}>{apiStatus === 'online' ? 'En ligne' : 'Non vérifiée'}</Badge></div><div className="monitor-kpis"><KpiCard icon={ShieldCheck} label="Qualité des données" value="Stable" trend="0 anomalie" sub="mode démo" tone="tone-green" /><KpiCard icon={BrainCircuit} label="Version du modèle" value={modelVersion} trend="Actif" sub="Artefact chargé par API" tone="tone-blue" /><KpiCard icon={GaugeCircle} label="Taux de défaut observé" value="5,8 %" trend="−1,2 %" sub="illustratif uniquement" tone="tone-teal" /><KpiCard icon={Zap} label="Prédictions suivies" value="2 843" trend="+10 %" sub="jeu de démonstration" tone="tone-orange" /></div><div className="dashboard-grid monitor-charts"><article className="card chart-card trend-card"><div className="card-heading"><div><h2>Dérive des données (PSI)</h2><p>Indice illustratif sur les 7 derniers jours</p></div><Badge kind="success">Sous le seuil d’alerte</Badge></div><div className="chart-area"><ResponsiveContainer width="100%" height="100%"><LineChart data={driftData} margin={{ top: 15, right: 12, bottom: 0, left: -18 }}><CartesianGrid vertical={false} stroke="#E9EEF5" strokeDasharray="3 4" /><XAxis dataKey="day" axisLine={false} tickLine={false} tick={{ fill: '#8391A7', fontSize: 11 }} dy={8} /><YAxis domain={[0, .3]} ticks={[0, .1, .2, .3]} axisLine={false} tickLine={false} tick={{ fill: '#8391A7', fontSize: 11 }} /><Tooltip formatter={(value: number) => [value.toFixed(3), 'PSI']} contentStyle={{ borderRadius: 12, border: '1px solid #E6ECF4' }} /><Line type="monotone" dataKey="v" stroke="#20A892" strokeWidth={2.5} dot={{ r: 3, fill: '#fff', stroke: '#20A892', strokeWidth: 2 }} /></LineChart></ResponsiveContainer></div><div className="chart-footnote"><span className="legend-dot" style={{ background: '#20A892' }} /> PSI simulé <span className="dash-line" /> Seuil d’alerte souvent configuré à 0,25 — à confirmer dans ton monitoring</div></article><article className="card chart-card"><div className="card-heading"><div><h2>Répartition du risque</h2><p>Données illustratives</p></div><button className="icon-button subtle"><MoreHorizontal size={18} /></button></div><div className="risk-distribution"><div className="risk-ring"><div><strong>2 843</strong><span>prédictions</span></div></div><div className="risk-legend"><div><span className="risk-dot low" /><span>Faible</span><strong>72 %</strong></div><div><span className="risk-dot medium" /><span>Modéré</span><strong>18 %</strong></div><div><span className="risk-dot high" /><span>Élevé</span><strong>10 %</strong></div></div></div></article></div><article className="card model-health-card"><div><div className="mini-status-icon green"><CheckCircle2 size={18} /></div><div><strong>Pipeline ML</strong><p>Chargement de l’artefact et inférence</p></div><Badge kind="success">Chargé</Badge></div><div><div className="mini-status-icon blue"><Cloud size={18} /></div><div><strong>Base Azure SQL</strong><p>Persistance des demandes et prédictions</p></div><Badge kind={apiStatus === 'online' ? 'success' : 'warning'}>{apiStatus === 'online' ? 'Connectée' : 'À vérifier'}</Badge></div><div><div className="mini-status-icon teal"><Fingerprint size={18} /></div><div><strong>Traçabilité</strong><p>Référence et version du modèle sur les résultats d’API</p></div><Badge kind="blue">Active</Badge></div></article></>;
}

function ReportsView() {
  return <><PageHeader eyebrow="DOCUMENTATION" title="Rapports & exports" description="Préparez des synthèses auditables sur l’activité et les performances du modèle." /><div className="reports-grid"><article className="card report-card"><div className="report-icon blue"><FileBarChart2 size={24} /></div><h2>Rapport de risque crédit</h2><p>Synthèse des métriques, du seuil de décision et des caractéristiques des analyses.</p><Badge kind="warning">Données à connecter</Badge><button className="button button-secondary" disabled><Download size={15} /> Exporter PDF</button></article><article className="card report-card"><div className="report-icon green"><Activity size={24} /></div><h2>Rapport de monitoring</h2><p>Drift des caractéristiques, qualité de données, disponibilité et historique de versions.</p><Badge kind="warning">Données à connecter</Badge><button className="button button-secondary" disabled><Download size={15} /> Exporter CSV</button></article><article className="card report-card"><div className="report-icon purple"><ShieldCheck size={24} /></div><h2>Rapport de gouvernance</h2><p>Traçabilité des prédictions, décisions humaines et labels arrivés à maturité.</p><Badge kind="blue">Structure prête</Badge><button className="button button-secondary" disabled><Download size={15} /> Générer un rapport</button></article></div></>;
}

function SettingsView({ apiStatus }: { apiStatus: string }) {
  return <><PageHeader eyebrow="CONFIGURATION" title="Paramètres de la plateforme" description="Informations sur l’API, le modèle et l’architecture de stockage." /><div className="settings-layout"><article className="card settings-card"><div className="settings-card-head"><div className="report-icon blue"><Cloud size={22} /></div><div><h2>Connexion API</h2><p>Adresse de l’API configurée pour cette interface.</p></div><Badge kind={apiStatus === 'online' ? 'success' : 'warning'}>{apiStatus === 'online' ? 'Connectée' : 'Non vérifiée'}</Badge></div><div className="settings-row"><span>Base URL</span><code>{API_BASE}</code></div><div className="settings-row"><span>Endpoint de prédiction</span><code>/predict</code></div><div className="settings-row"><span>Contrôle de santé</span><code>/health</code></div><div className="settings-row"><span>Origine de configuration</span><code>NEXT_PUBLIC_API_BASE_URL</code></div><p className="settings-help"><Info size={15} /> Pour changer l’URL, modifiez `.env.local` puis redémarrez Next.js. N’exposez jamais les identifiants SQL dans une variable `NEXT_PUBLIC_*`.</p></article><article className="card settings-card"><div className="settings-card-head"><div className="report-icon green"><BrainCircuit size={22} /></div><div><h2>Modèle de risque</h2><p>Informations annoncées par `/model-info`.</p></div><Badge kind="blue">Production</Badge></div><div className="settings-row"><span>Version côté interface</span><strong>1.0.0</strong></div><div className="settings-row"><span>Seuil de décision</span><strong>0,48 (48 %)</strong></div><div className="settings-row"><span>Classe positive</span><strong>Défaut (loan_status = 1)</strong></div><div className="settings-row"><span>Décision d’interface</span><strong>Recommandation uniquement</strong></div><p className="settings-help"><Shield size={15} /> Cette interface n’accorde et ne refuse pas juridiquement un prêt. Elle affiche la recommandation retournée par le modèle.</p></article></div></>;
}

export default function HomePage() {
  const [active, setActive] = useState<View>('dashboard');
  const [mobileOpen, setMobileOpen] = useState(false);
  const [apiStatus, setApiStatus] = useState<'checking' | 'online' | 'offline'>('checking');
  const [modelVersion, setModelVersion] = useState('1.0.0');
  const [now, setNow] = useState('');
  const checkHealth = useCallback(async () => {
    setApiStatus('checking');
    try {
      const controller = new AbortController();
      const timeout = setTimeout(() => controller.abort(), 5000);
      const response = await fetch(`${API_BASE}/health`, { cache: 'no-store', signal: controller.signal });
      clearTimeout(timeout);
      if (!response.ok) throw new Error('API non disponible');
      const data = await response.json();
      setApiStatus(data.azure_sql === 'connected' ? 'online' : 'offline');
      if (data.model_version) setModelVersion(data.model_version);
    } catch { setApiStatus('offline'); }
  }, []);
  useEffect(() => {
    setNow(new Intl.DateTimeFormat('fr-FR', { weekday: 'long', day: 'numeric', month: 'long', year: 'numeric' }).format(new Date()));
    void checkHealth();
  }, [checkHealth]);
  const titles: Record<View, string> = { dashboard: 'Tableau de bord', analysis: 'Analyse de crédit', portfolio: 'Portefeuille', monitoring: 'Surveillance', reports: 'Rapports', settings: 'Paramètres' };
  const navigate = (view: View) => setActive(view);
  const renderView = () => {
    if (active === 'dashboard') return <Dashboard onNewAnalysis={() => navigate('analysis')} apiStatus={apiStatus} modelVersion={modelVersion} />;
    if (active === 'analysis') return <AnalysisView apiStatus={apiStatus} modelVersion={modelVersion} />;
    if (active === 'portfolio') return <PortfolioView />;
    if (active === 'monitoring') return <MonitoringView apiStatus={apiStatus} modelVersion={modelVersion} />;
    if (active === 'reports') return <ReportsView />;
    return <SettingsView apiStatus={apiStatus} />;
  };
  return <div className="app-shell"><Sidebar active={active} onNavigate={navigate} mobileOpen={mobileOpen} onClose={() => setMobileOpen(false)} /><div className="app-main"><Topbar title={titles[active]} onMenu={() => setMobileOpen(true)} /><main className="page-content">{renderView()}<footer className="app-footer"><span>© 2026 FinPredict AI Pro</span><span>Intelligent Credit Risk <span className="footer-sep">·</span> <span className="footer-status"><span className={`online-dot ${apiStatus !== 'online' ? 'offline-dot' : ''}`} />{apiStatus === 'online' ? 'API & SQL connectés' : apiStatus === 'checking' ? 'Vérification de connexion' : 'API non vérifiée'}</span></span><span>{now ? now.charAt(0).toUpperCase() + now.slice(1) : 'FinPredict'}</span></footer></main></div></div>;
}
