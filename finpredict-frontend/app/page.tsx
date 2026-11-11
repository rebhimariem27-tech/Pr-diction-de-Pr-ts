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
  explanation?: Explanation;
};
type PortfolioRow = {
  application_id: number;
  prediction_id: number;
  application_ref: string;
  submitted_at_utc: string;
  loan_amnt: number | null;
  default_probability: number;
  recommendation: 'ACCEPT' | 'REFUSE';
  actual_decision: string;
  model_version: string;
};
type DashboardSummary = {
  period_days: number;
  data_available: boolean;
  analyzed_count: number;
  average_default_probability: number | null;
  recommendation_counts: { ACCEPT: number; REFUSE: number };
  actual_decision_counts: Record<string, number>;
  total_loan_amount: number;
  monthly: { month: string; analyzed_count: number; average_default_probability: number }[];
  latest_submitted_at_utc: string | null;
  model_version: string | null;
};
type MonitoringSummary = {
  generated_at_utc: string;
  overall_status: 'NORMAL' | 'WARNING' | 'ALERT' | 'UNKNOWN';
  demo_mode: boolean;
  reference_rows: number;
  current_rows: number;
  high_drift_features: number;
  moderate_drift_features: number;
  high_missingness_features: number;
  prediction_psi: number | null;
  prediction_ks: number | null;
  threshold: number;
  reference_mean_default_probability: number | null;
  current_mean_default_probability: number | null;
  refusal_rate_delta_pp: number | null;
  model_performance: Record<string, number | boolean> | null;
  drift_features: { feature: string; type: string; psi: string; drift_status: string }[];
  fairness: Record<string, { flag_4_5_rule?: boolean }[]>;
};
type ExplainFactor = { feature: string; value: string | number | null; group: string; points: number; effect: string };
type ExplainGroup = { id: string; label: string; points: number };
type Explanation = { unit?: string; baseline_probability?: number; final_probability?: number; threshold?: number; groups?: ExplainGroup[]; factors?: ExplainFactor[] };
type Factor = { feature: string; value: string | number | null; contribution: number; effect: string };
const featureLabels: Record<string, string> = {
  person_age: 'Âge', person_income: 'Revenu annuel', loan_amnt: 'Montant du prêt', loan_int_rate: 'Taux d’intérêt',
  person_emp_length: 'Ancienneté professionnelle', loan_percent_income: 'Part du revenu consacrée au prêt',
  cb_person_cred_hist_length: 'Historique de crédit', interest_income_ratio: 'Ratio intérêts / revenu',
  person_home_ownership: 'Logement', loan_intent: 'Objet du prêt', loan_grade: 'Note du prêt', cb_person_default_on_file: 'Défaut antérieur',
};

const categoryLabels: Record<string, Record<string, string>> = {
  person_home_ownership: { RENT: 'Locataire', OWN: 'Propriétaire', MORTGAGE: 'Crédit immobilier', OTHER: 'Autre' },
  loan_intent: { PERSONAL: 'Personnel', EDUCATION: 'Éducation', MEDICAL: 'Médical', VENTURE: 'Entrepreneuriat', HOMEIMPROVEMENT: 'Amélioration du logement', DEBTCONSOLIDATION: 'Regroupement de dettes' },
  cb_person_default_on_file: { Y: 'Oui', N: 'Non' },
};
const fr = (n: number, digits = 0) => n.toLocaleString('fr-FR', { maximumFractionDigits: digits });
// Affichage lisible de la valeur d'un facteur (jamais de flottant brut).
function formatFactorValue(feature: string, value: string | number | null): string {
  if (value === null || value === undefined) return '—';
  if (typeof value === 'string') {
    if (feature === 'loan_grade') return `Grade ${value}`;
    return categoryLabels[feature]?.[value] ?? value;
  }
  switch (feature) {
    case 'person_income': case 'loan_amnt': return `${fr(value)} €`;
    case 'loan_int_rate': return `${fr(value, 2)} %`;
    case 'loan_percent_income': return `${fr(value * 100, 0)} % du revenu`;
    case 'interest_income_ratio': return `${fr(value * 100, 2)} % du revenu`;
    case 'person_age': case 'person_emp_length': case 'cb_person_cred_hist_length': return `${fr(value)} ans`;
    default: return fr(value, 2);
  }
}

const signed = (n: number) => (Math.abs(n) < 0.05 ? '≈ 0' : `${n > 0 ? '+' : '−'}${Math.abs(n).toLocaleString('fr-FR', { minimumFractionDigits: 1, maximumFractionDigits: 1 })}`);
const tone = (n: number) => (Math.abs(n) < 0.05 ? 'flat' : n > 0 ? 'up' : 'down');
const pctOf = (p: number) => `${fr(p * 100, p < 0.1 ? 1 : 0)} %`;

// Explication de la décision : chaque thème est exprimé en points de probabilité de défaut.
function ExplainPanel({ ex }: { ex: Explanation }) {
  const groups = ex.groups || [];
  const factors = ex.factors || [];
  const maxAbs = Math.max(0.1, ...groups.map(g => Math.abs(g.points)));
  const { baseline_probability: base, final_probability: fin, threshold: thr } = ex;
  let position = '';
  if (fin !== undefined && thr !== undefined) {
    const gap = (thr - fin) * 100;
    position = gap >= 0 ? `, soit ${fr(gap, 1)} points sous le seuil de refus (${pctOf(thr)})` : `, soit ${fr(-gap, 1)} points au-dessus du seuil de refus (${pctOf(thr)})`;
  }
  return (
    <div className="explain-block">
      <span className="eyebrow">POURQUOI CE RÉSULTAT</span>
      {base !== undefined && fin !== undefined && (
        <p className="explain-summary">Le modèle part d’un point de départ interne de <strong>{pctOf(base)}</strong>. Les caractéristiques de ce dossier le font évoluer jusqu’à une probabilité de défaut de <strong>{pctOf(fin)}</strong>{position}.</p>
      )}
      <div className="explain-groups">
        {groups.map(g => (
          <details className="explain-group" key={g.id}>
            <summary>
              <span className="eg-label">{g.label}</span>
              <span className="eg-track"><i className={tone(g.points)} style={{ width: `${Math.min(100, (Math.abs(g.points) / maxAbs) * 100)}%` }} /></span>
              <strong className={`eg-value ${tone(g.points)}`}>{signed(g.points)} pts</strong>
            </summary>
            <ul>
              {factors.filter(f => f.group === g.id).map(f => (
                <li key={f.feature}>
                  <span>{featureLabels[f.feature] || f.feature}{f.value !== null ? ` : ${formatFactorValue(f.feature, f.value)}` : ''}</span>
                  <b className={tone(f.points)}>{signed(f.points)} pts</b>
                </li>
              ))}
            </ul>
          </details>
        ))}
      </div>
      <p className="explain-note">Lecture : un thème en rouge (+) pousse la probabilité de défaut vers le haut, un thème en vert (−) vers le bas, en points de pourcentage. La somme des thèmes correspond à l’écart entre le point de départ et la probabilité finale. Les variables liées (montant, revenu, ratios) sont regroupées parce que leurs effets individuels se compensent en partie. Cliquez sur un thème pour voir le détail. Explication indicative du modèle, pas une relation de cause à effet.</p>
    </div>
  );
}
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

function Logo() {
  return <div className="brand"><div className="brand-mark"><span /><span /><span /></div><div><strong>FinPredict</strong><small>AI for Smarter Lending</small></div></div>;
}

function Badge({ children, kind = 'neutral' }: { children: React.ReactNode; kind?: 'neutral' | 'success' | 'danger' | 'warning' | 'blue' }) {
  return <span className={`badge badge-${kind}`}><span className="badge-dot" />{children}</span>;
}

function KpiCard({ icon: Icon, label, value, trend, sub, tone }: { icon: typeof FileText; label: string; value: string; trend?: string; sub: string; tone: string }) {
  const positive = trend?.trim().startsWith('+');
  return <article className="kpi-card card"><div className={`kpi-icon ${tone}`}><Icon size={19} strokeWidth={2.2} /></div><div className="kpi-content"><span className="eyebrow">{label}</span><div className="kpi-value">{value}</div><div className="kpi-foot">{trend && <span className={positive ? 'trend-up' : 'trend-down'}>{positive ? <ArrowUpRight size={14} /> : <ArrowDownRight size={14} />}{trend}</span>}<span className="muted">{sub}</span></div></div></article>;
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
  return <div className="demo-notice"><Info size={15} /><span><strong>Mode démonstration</strong> — certaines métriques de surveillance restent illustratives tant qu’un batch de monitoring réel n’est pas connecté.</span></div>;
}

function Dashboard({ onNewAnalysis, apiStatus, modelVersion }: { onNewAnalysis: () => void; apiStatus: string; modelVersion: string }) {
  const [summary, setSummary] = useState<DashboardSummary | null>(null);
  const [loading, setLoading] = useState(true);
  const [error, setError] = useState('');
  useEffect(() => {
    fetch('/api/dashboard-summary?period_days=30', { cache: 'no-store' })
      .then(async response => {
        const data = await response.json().catch(() => ({}));
        if (!response.ok) throw new Error(data.detail || 'Impossible de charger les agrégations du dashboard.');
        return data as DashboardSummary;
      })
      .then(setSummary)
      .catch(error => setError(error instanceof Error ? error.message : 'Impossible de charger les agrégations du dashboard.'))
      .finally(() => setLoading(false));
  }, []);

  const analyzed = summary?.analyzed_count ?? 0;
  const accepted = summary?.recommendation_counts.ACCEPT ?? 0;
  const refused = summary?.recommendation_counts.REFUSE ?? 0;
  const totalRecommendations = accepted + refused;
  const averageRisk = summary?.average_default_probability;
  const monthlyData = (summary?.monthly ?? []).map(row => ({
    month: new Intl.DateTimeFormat('fr-FR', { month: 'short' }).format(new Date(`${row.month}-01T00:00:00Z`)),
    rate: row.average_default_probability * 100,
  }));
  const pieData = [
    { name: 'Acceptées', value: totalRecommendations ? (accepted / totalRecommendations) * 100 : 0, color: '#21A675' },
    { name: 'Refusées', value: totalRecommendations ? (refused / totalRecommendations) * 100 : 0, color: '#F36D78' },
  ];
  const formatCount = (value: number) => value.toLocaleString('fr-FR');
  const formatRisk = (value: number | null) => value === null ? '—' : `${(value * 100).toLocaleString('fr-FR', { maximumFractionDigits: 1 })} %`;
  return <>
    {error ? <div className="error-callout"><AlertCircle size={17} /><div><strong>Dashboard indisponible</strong><p>{error}</p></div></div> : null}
    {!loading && !error && !summary?.data_available ? <div className="demo-notice"><Info size={15} /><span><strong>Aucune donnée enregistrée</strong> — les agrégations apparaîtront après la première analyse persistée dans Azure SQL.</span></div> : null}
    <PageHeader eyebrow="VUE D’ENSEMBLE" title="Bonjour Mariem 👋" description="Voici un aperçu de votre portefeuille et des analyses de risque." right={<><button className="button button-secondary"><CalendarDays size={16} /> 30 derniers jours <ChevronDown size={14} /></button><button className="button button-primary" onClick={onNewAnalysis}><Plus size={17} /> Nouvelle analyse</button></>} />
    <div className="kpi-grid">
      <KpiCard icon={FileText} label="Prêts analysés" value={loading ? '…' : formatCount(analyzed)} sub="30 derniers jours" tone="tone-blue" />
      <KpiCard icon={ShieldCheck} label="Probabilité moyenne de défaut" value={loading ? '…' : formatRisk(averageRisk ?? null)} sub="score prédit, pas un défaut observé" tone="tone-teal" />
      <KpiCard icon={CheckCircle2} label="Recommandations acceptées" value={loading ? '…' : formatCount(accepted)} sub="selon le seuil du modèle" tone="tone-green" />
      <KpiCard icon={AlertCircle} label="Recommandations refusées" value={loading ? '…' : formatCount(refused)} sub="selon le seuil du modèle" tone="tone-red" />
    </div>
    <div className="dashboard-grid main-charts">
      <article className="card chart-card trend-card"><div className="card-heading"><div><h2>Évolution de la probabilité moyenne de défaut</h2><p>Score prédit sur les 30 derniers jours</p></div></div><div className="chart-area line-chart-area">{monthlyData.length ? <><ResponsiveContainer width="100%" height="100%"><LineChart data={monthlyData} margin={{ top: 15, right: 12, bottom: 0, left: -18 }}><CartesianGrid vertical={false} stroke="#E9EEF5" strokeDasharray="3 4" /><XAxis dataKey="month" axisLine={false} tickLine={false} tick={{ fill: '#8391A7', fontSize: 11 }} dy={8} /><YAxis domain={[0, 100]} ticks={[0, 25, 50, 75, 100]} axisLine={false} tickLine={false} tick={{ fill: '#8391A7', fontSize: 11 }} tickFormatter={v => `${v}%`} /><Tooltip contentStyle={{ border: '1px solid #E6ECF4', borderRadius: 12, boxShadow: '0 8px 24px rgba(18,42,75,.08)' }} formatter={(value: number) => [`${value.toLocaleString('fr-FR', { maximumFractionDigits: 1 })}%`, 'Probabilité moyenne']} /><Line type="monotone" dataKey="rate" stroke="#4C6FFF" strokeWidth={2.6} dot={{ r: 3.5, strokeWidth: 2, fill: '#fff', stroke: '#4C6FFF', }} activeDot={{ r: 5 }} /></LineChart></ResponsiveContainer><div className="line-chart-label">{formatRisk(averageRisk ?? null)} <span>moyenne actuelle</span></div></> : <div className="chart-empty">Aucune analyse sur la période sélectionnée.</div>}</div></article>
      <article className="card chart-card decisions-card"><div className="card-heading"><div><h2>Répartition des recommandations</h2><p>Données persistées dans Azure SQL</p></div><button className="icon-button subtle" aria-label="Informations sur les recommandations"><MoreHorizontal size={19} /></button></div><div className="donut-layout">{totalRecommendations ? <div className="donut-wrap"><ResponsiveContainer width="100%" height="100%"><PieChart><Pie data={pieData} dataKey="value" nameKey="name" innerRadius="68%" outerRadius="92%" startAngle={90} endAngle={-270} paddingAngle={2} stroke="none">{pieData.map((entry, idx) => <Cell key={idx} fill={entry.color} />)}</Pie></PieChart></ResponsiveContainer><div className="donut-center"><strong>{formatCount(totalRecommendations)}</strong><span>dossiers</span></div></div> : <div className="chart-empty">Aucune recommandation.</div>}<div className="legend-list">{pieData.map(row => <div className="legend-row" key={row.name}><span className="legend-dot" style={{ background: row.color }} /><span>{row.name}</span><strong>{row.value.toLocaleString('fr-FR', { maximumFractionDigits: 1 })}%</strong></div>)}</div></div></article>
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
        {prediction ? <article className="card result-card"><div className="result-top"><div className={`result-symbol ${prediction.recommendation === 'ACCEPT' ? 'result-positive' : 'result-negative'}`}>{prediction.recommendation === 'ACCEPT' ? <CheckCircle2 size={24} /> : <AlertCircle size={24} />}</div><div><span className="eyebrow">RÉSULTAT DU MODÈLE</span><h2>{prediction.recommendation === 'ACCEPT' ? 'Risque sous le seuil' : 'Risque au-dessus du seuil'}</h2><p>Recommandation : {prediction.recommendation === 'ACCEPT' ? 'ACCEPT' : 'REFUSE'}</p></div></div><div className="risk-score"><div><span className="eyebrow">Probabilité de défaut</span><strong>{defaultPct.toLocaleString('fr-FR', { maximumFractionDigits: 2 })}<small>%</small></strong></div><div className="risk-meter"><div className="risk-meter-track"><div style={{ width: `${Math.min(defaultPct, 100)}%` }} className={defaultPct >= prediction.threshold * 100 ? 'risk-high' : 'risk-low'} /></div><div className="risk-meter-labels"><span>0 %</span><span>Seuil {Math.round(prediction.threshold * 100)} %</span><span>100 %</span></div></div></div><div className="result-details"><div><span>Référence</span><strong>{prediction.application_ref}</strong></div><div><span>Modèle</span><strong>{prediction.model_version}</strong></div><div><span>ID demande</span><strong>#{prediction.application_id}</strong></div><div><span>ID prédiction</span><strong>#{prediction.prediction_id}</strong></div>{prediction.derived_features ? <><div><span>Part du revenu / prêt</span><strong>{prediction.derived_features.loan_percent_income != null ? (prediction.derived_features.loan_percent_income * 100).toLocaleString('fr-FR', { maximumFractionDigits: 1 }) + ' %' : '—'}</strong></div><div><span>Ratio intérêts / revenu</span><strong>{prediction.derived_features.interest_income_ratio != null ? (prediction.derived_features.interest_income_ratio * 100).toLocaleString('fr-FR', { maximumFractionDigits: 2 }) + ' %' : '—'}</strong></div></> : null}<div><span>Stockage</span><strong><Database size={13} /> {prediction.tracking_storage}</strong></div></div>{prediction.explanation?.groups?.length ? <ExplainPanel ex={prediction.explanation} /> : null}<div className="result-disclaimer"><Info size={15} /> Décision indicative : la décision finale appartient à l’analyste ou au processus de crédit responsable.</div></article> : <article className="card aside-empty"><div className="aside-illustration"><div className="halo halo-a" /><div className="halo halo-b" /><div className="shield-big"><ShieldCheck size={42} /></div><Sparkles className="sparkle s-a" size={19} /><Sparkles className="sparkle s-b" size={14} /></div><span className="eyebrow">ANALYSE EXPLICABLE</span><h2>Votre analyse apparaîtra ici</h2><p>Le modèle estimera la probabilité de défaut, la comparera au seuil de 48 % et enregistrera le résultat dans Azure SQL via FastAPI.</p><div className="aside-checks"><div><CheckCircle2 size={15} /> Pipeline de production calibré</div><div><CheckCircle2 size={15} /> Seuil de décision versionné</div><div><CheckCircle2 size={15} /> Traçabilité du résultat</div></div></article>}
        <article className="card criteria-card"><div className="card-heading"><div><h3>Critères pris en compte</h3><p>12 variables du pipeline</p></div><CircleHelp size={16} className="icon-muted" /></div><div className="criteria-row"><span><WalletCards size={15} /> Capacité de remboursement</span><Check size={15} /></div><div className="criteria-row"><span><FileClock size={15} /> Historique de crédit</span><Check size={15} /></div><div className="criteria-row"><span><House size={15} /> Situation de logement</span><Check size={15} /></div><div className="criteria-row"><span><UsersRound size={15} /> Profil professionnel</span><Check size={15} /></div><div className="criteria-row"><span><LockKeyhole size={15} /> Données protégées</span><Check size={15} /></div><div className="criteria-note">Aucune décision automatique n’est exécutée par cette interface.</div></article>
      </aside>
    </div>
  </>;
}

function PortfolioView() {
  const [query, setQuery] = useState('');
  const [rows, setRows] = useState<PortfolioRow[]>([]);
  const [loading, setLoading] = useState(true);
  const [error, setError] = useState('');
  const [savingDecision, setSavingDecision] = useState<number | null>(null);
  useEffect(() => {
    fetch('/api/portfolio?limit=100', { cache: 'no-store' })
      .then(async response => {
        const data = await response.json().catch(() => ({}));
        if (!response.ok) throw new Error(data.detail || 'Impossible de charger l’historique.');
        return data as PortfolioRow[];
      })
      .then(setRows)
      .catch(error => setError(error instanceof Error ? error.message : 'Impossible de charger l’historique.'))
      .finally(() => setLoading(false));
  }, []);
  const filtered = rows.filter(row => row.application_ref.toLowerCase().includes(query.toLowerCase()));
  const updateDecision = async (applicationId: number, actualDecision: string) => {
    setSavingDecision(applicationId);
    setError('');
    try {
      const response = await fetch(`/api/applications/${applicationId}/decision`, {
        method: 'PATCH',
        headers: { 'Content-Type': 'application/json', Accept: 'application/json' },
        body: JSON.stringify({ actual_decision: actualDecision }),
      });
      const data = await response.json().catch(() => ({}));
      if (!response.ok) throw new Error(data.detail || 'Impossible d’enregistrer la décision humaine.');
      setRows(current => current.map(row => row.application_id === applicationId ? { ...row, actual_decision: data.actual_decision } : row));
    } catch (e) {
      setError(e instanceof Error ? e.message : 'Impossible d’enregistrer la décision humaine.');
    } finally {
      setSavingDecision(null);
    }
  };
  const amountTotal = rows.reduce((total, row) => total + (row.loan_amnt || 0), 0);
  const pending = rows.filter(row => row.actual_decision === 'PENDING' || row.actual_decision === 'MANUAL_REVIEW').length;
  const formatDate = (value: string) => new Intl.DateTimeFormat('fr-FR', { dateStyle: 'medium' }).format(new Date(value));
  const formatAmount = (value: number | null) => value === null ? '—' : `${value.toLocaleString('fr-FR', { maximumFractionDigits: 0 })} €`;
  return <><PageHeader eyebrow="SUIVI DES DOSSIERS" title="Portefeuille de crédit" description="Comparez la recommandation du modèle avec la décision humaine enregistrée dans Azure SQL." right={<button className="button button-primary" disabled={rows.length === 0}><Download size={16} /> Exporter</button>} />{error && <div className="error-callout"><AlertCircle size={17} /><div><strong>Opération non terminée</strong><p>{error} Vérifiez que l’API et Azure SQL sont accessibles.</p></div></div>}<div className="portfolio-stats"><div className="card portfolio-stat"><span className="eyebrow">Demandes enregistrées</span><strong>{loading ? '…' : rows.length}</strong><small>Azure SQL</small></div><div className="card portfolio-stat"><span className="eyebrow">Montant total analysé</span><strong>{loading ? '…' : `${amountTotal.toLocaleString('fr-FR', { maximumFractionDigits: 0 })} €`}</strong><small>Somme des dossiers chargés</small></div><div className="card portfolio-stat"><span className="eyebrow">À examiner</span><strong>{loading ? '…' : pending}</strong><small><Clock3 size={13} /> En attente de revue humaine</small></div></div><article className="card table-card"><div className="table-header"><div><h2>Demandes récentes</h2><p>Dernières prédictions persistées dans Azure SQL</p></div><div className="table-tools"><div className="table-search"><Search size={15} /><input placeholder="Chercher une référence…" value={query} onChange={e => setQuery(e.target.value)} /></div><button className="button button-secondary button-small"><SlidersHorizontal size={15} /> Filtres</button></div></div><div className="table-scroll"><table><thead><tr><th>Référence</th><th>Date</th><th>Montant demandé</th><th>Probabilité de défaut</th><th>Recommandation du modèle</th><th>Décision humaine</th></tr></thead><tbody>{filtered.map(row => <tr key={row.prediction_id}><td><strong>{row.application_ref}</strong><small>Application #{row.application_id}</small></td><td>{formatDate(row.submitted_at_utc)}</td><td className="amount-cell">{formatAmount(row.loan_amnt)}</td><td><div className="probability-cell"><div className="probability-track"><span style={{ width: `${row.default_probability * 100}%` }} /></div><strong>{(row.default_probability * 100).toLocaleString('fr-FR', { maximumFractionDigits: 1 })}%</strong></div></td><td><Badge kind={row.recommendation === 'ACCEPT' ? 'success' : 'danger'}>{row.recommendation}</Badge></td><td><select aria-label={`Décision humaine pour ${row.application_ref}`} value={row.actual_decision} disabled={savingDecision === row.application_id} onChange={event => void updateDecision(row.application_id, event.target.value)}><option value="PENDING">À examiner</option><option value="MANUAL_REVIEW">Revue manuelle</option><option value="APPROVED">Approuvée</option><option value="REJECTED">Rejetée</option></select></td></tr>)}</tbody></table>{loading && <div className="empty-table">Chargement de l’historique…</div>}{!loading && !error && filtered.length === 0 && <div className="empty-table">Aucune analyse enregistrée ne correspond à la recherche.</div>}</div><div className="table-footer"><span>{loading ? 'Chargement…' : `${filtered.length} dossier(s) affiché(s)`}</span><div className="pagination"><button disabled aria-label="Page précédente"><ChevronLeft size={16} /></button><button className="page-active">1</button><button disabled aria-label="Page suivante"><ChevronRight size={16} /></button></div></div></article></>;
}

function MonitoringView({ apiStatus, modelVersion }: { apiStatus: string; modelVersion: string }) {
  const [monitoring, setMonitoring] = useState<MonitoringSummary | null>(null);
  const [error, setError] = useState('');
  const [loading, setLoading] = useState(true);
  const load = useCallback(async () => {
    setLoading(true); setError('');
    try {
      const response = await fetch('/api/monitoring-summary', { headers: { Accept: 'application/json' } });
      const body = await response.json().catch(() => ({}));
      if (!response.ok) throw new Error(body.detail || `Impossible de charger le monitoring (HTTP ${response.status}).`);
      setMonitoring(body as MonitoringSummary);
    } catch (e) {
      setError(e instanceof Error ? e.message : 'Impossible de charger le monitoring.');
    } finally { setLoading(false); }
  }, []);
  useEffect(() => { void load(); }, [load]);
  const statusKind = monitoring?.overall_status === 'NORMAL' ? 'success' : monitoring?.overall_status === 'UNKNOWN' ? 'warning' : 'danger';
  const fairnessFlags = monitoring ? Object.values(monitoring.fairness).flat().filter(group => group.flag_4_5_rule).length : 0;
  const performance = monitoring?.model_performance;
  return <><PageHeader eyebrow="OBSERVABILITÉ & GOUVERNANCE" title="Surveillance du modèle" description="Indicateurs générés par le pipeline de monitoring : drift, qualité des données, performance et fairness." right={<button className="button button-secondary" onClick={() => void load()} disabled={loading}><RefreshCw size={15} /> Actualiser</button>} />
    {error && <div className="error-callout"><AlertCircle size={17} /><div><strong>Monitoring indisponible</strong><p>{error} Aucun indicateur fictif n’est affiché.</p></div></div>}
    {loading && <div className="card empty-table">Chargement des indicateurs de supervision…</div>}
    {monitoring && <><div className="monitor-banner"><div className={`monitor-banner-icon ${monitoring.overall_status === 'NORMAL' ? 'ok' : 'warn'}`}><Activity size={20} /></div><div><strong>État global : {monitoring.overall_status}</strong><p>Rapport généré le {new Date(monitoring.generated_at_utc).toLocaleString('fr-FR')}{monitoring.demo_mode ? ' · batch de démonstration' : ' · batch de production'}</p></div><Badge kind={statusKind}>{monitoring.demo_mode ? 'Démo' : 'Production'}</Badge></div>
      <div className="monitor-kpis"><KpiCard icon={ShieldCheck} label="Qualité des données" value={monitoring.high_missingness_features === 0 ? 'Stable' : 'À surveiller'} trend={`${monitoring.high_missingness_features} alerte(s)`} sub={`${monitoring.current_rows.toLocaleString('fr-FR')} lignes courantes`} tone="tone-green" /><KpiCard icon={BrainCircuit} label="Version du modèle" value={modelVersion} trend={`Seuil ${(monitoring.threshold * 100).toFixed(0)} %`} sub="Artefact chargé par API" tone="tone-blue" /><KpiCard icon={GaugeCircle} label="PSI prédictions" value={monitoring.prediction_psi?.toFixed(3) ?? '—'} trend={`${monitoring.high_drift_features} drift élevé(s)`} sub={`${monitoring.moderate_drift_features} drift modéré(s)`} tone="tone-teal" /><KpiCard icon={Zap} label="Fairness" value={fairnessFlags === 0 ? 'Aucun signal' : `${fairnessFlags} signal(s)`} trend={performance?.available ? `AUC ${(Number(performance.roc_auc) * 100).toFixed(1)} %` : 'Non calculée'} sub={`${monitoring.reference_rows.toLocaleString('fr-FR')} lignes référence`} tone="tone-orange" /></div>
      <div className="dashboard-grid monitor-charts"><article className="card chart-card trend-card"><div className="card-heading"><div><h2>Drift des variables</h2><p>PSI du dernier rapport généré</p></div><Badge kind={monitoring.high_drift_features ? 'danger' : 'success'}>{monitoring.high_drift_features ? 'Action requise' : 'Sous contrôle'}</Badge></div><div className="monitor-feature-list">{monitoring.drift_features.slice(0, 8).map(item => <div className="monitor-feature" key={item.feature}><span>{item.feature}</span><div className="bar"><i style={{ width: `${Math.min(100, Number(item.psi) * 400)}%` }} /></div><strong>{Number(item.psi).toFixed(3)}</strong><Badge kind={item.drift_status === 'LOW' ? 'success' : item.drift_status === 'MODERATE' ? 'warning' : 'danger'}>{item.drift_status}</Badge></div>)}</div></article>
        <article className="card chart-card"><div className="card-heading"><div><h2>Prédictions</h2><p>Comparaison référence / batch courant</p></div></div><div className="risk-legend"><div><span>Probabilité moyenne référence</span><strong>{monitoring.reference_mean_default_probability === null ? '—' : `${(monitoring.reference_mean_default_probability * 100).toFixed(1)} %`}</strong></div><div><span>Probabilité moyenne courante</span><strong>{monitoring.current_mean_default_probability === null ? '—' : `${(monitoring.current_mean_default_probability * 100).toFixed(1)} %`}</strong></div><div><span>Écart du taux de refus</span><strong>{monitoring.refusal_rate_delta_pp === null ? '—' : `${monitoring.refusal_rate_delta_pp.toFixed(2)} pt`}</strong></div></div></article></div>
      <article className="card model-health-card"><div><div className="mini-status-icon green"><CheckCircle2 size={18} /></div><div><strong>Pipeline ML</strong><p>Rapport de monitoring disponible</p></div><Badge kind="success">Actif</Badge></div><div><div className="mini-status-icon blue"><Cloud size={18} /></div><div><strong>Fairness</strong><p>{fairnessFlags === 0 ? 'Aucun groupe sous le ratio 4/5' : 'Groupes à investiguer dans le rapport'}</p></div><Badge kind={fairnessFlags === 0 ? 'success' : 'warning'}>{fairnessFlags === 0 ? 'OK' : 'Revue'}</Badge></div><div><div className="mini-status-icon teal"><Fingerprint size={18} /></div><div><strong>Traçabilité</strong><p>Source : step10_monitoring_report.json</p></div><Badge kind="blue">Auditée</Badge></div></article></>}
  </>;
}

const reportDate = () => new Intl.DateTimeFormat('fr-FR', {
  dateStyle: 'long',
  timeStyle: 'short',
}).format(new Date());

function escapeHtml(value: string): string {
  return value.replace(/[&<>"']/g, character => ({
    '&': '&amp;', '<': '&lt;', '>': '&gt;', '"': '&quot;', "'": '&#039;',
  }[character] || character));
}

function openPrintableReport(title: string, subtitle: string, content: string) {
  const reportWindow = window.open('', '_blank', 'width=1100,height=800');
  if (!reportWindow) {
    window.alert('Le navigateur a bloqué la fenêtre du rapport. Autorisez les fenêtres pop-up puis réessayez.');
    return;
  }
  reportWindow.document.write(`<!doctype html>
    <html lang="fr"><head><meta charset="utf-8"><title>${escapeHtml(title)}</title>
    <style>
      @page { size: A4; margin: 0; }
      * { box-sizing: border-box; }
      body { margin: 0; color: #172f49; background: #edf3f8; font: 13px Arial, sans-serif; }
      .sheet { width: 210mm; min-height: 297mm; margin: 0 auto; padding: 18mm 17mm; background: white; }
      .top { display: flex; justify-content: space-between; align-items: flex-start; padding-bottom: 22px; border-bottom: 2px solid #dce8f1; }
      .brand { display: flex; gap: 12px; align-items: center; }
      .mark { width: 40px; height: 40px; border-radius: 12px; display: grid; place-items: center; color: white; font-size: 19px; font-weight: 800; background: linear-gradient(135deg,#20c9b0,#14527b); }
      .brand strong { display: block; font-size: 18px; letter-spacing: -.5px; }
      .brand span { display: block; margin-top: 4px; color: #8395a8; font-size: 10px; letter-spacing: 1.4px; text-transform: uppercase; }
      .meta { color: #8395a8; font-size: 10px; text-align: right; line-height: 1.7; }
      .eyebrow { margin-top: 34px; color: #159e94; font-size: 10px; font-weight: 700; letter-spacing: 2px; }
      h1 { margin: 8px 0 8px; color: #102c48; font-size: 30px; letter-spacing: -1px; }
      .subtitle { margin: 0; color: #71859a; font-size: 13px; }
      .hero { display: grid; grid-template-columns: 1fr 170px; gap: 18px; align-items: stretch; margin: 28px 0; }
      .hero-card { padding: 20px; border-radius: 16px; background: linear-gradient(135deg,#0b2b45,#165675); color: white; }
      .hero-card small { color: #a9d8d7; font-size: 10px; text-transform: uppercase; letter-spacing: 1px; }
      .hero-card strong { display: block; margin-top: 13px; font-size: 32px; }
      .hero-card p { margin: 8px 0 0; color: #d3e8ec; }
      .score { display: grid; place-items: center; border: 1px solid #dce8f1; border-radius: 16px; background: #f7fbfd; text-align: center; }
      .score strong { display: block; color: #159e94; font-size: 29px; }
      .score span { color: #8293a5; font-size: 10px; }
      h2 { margin: 26px 0 12px; font-size: 15px; color: #183b5b; }
      .grid { display: grid; grid-template-columns: repeat(2, 1fr); gap: 10px; }
      .item { padding: 12px 14px; border: 1px solid #e3ecf2; border-radius: 9px; }
      .item span { display: block; color: #8293a5; font-size: 10px; }
      .item strong { display: block; margin-top: 5px; color: #234866; font-size: 13px; }
      .section { margin-top: 25px; }
      .section h2 { margin-top: 0; }
      .section h3 { margin: 18px 0 8px; color: #234866; font-size: 12px; }
      .section p, .section li { color: #61778c; line-height: 1.65; }
      .section ul { margin: 8px 0 0; padding-left: 18px; }
      .two-col { display: grid; grid-template-columns: 1fr 1fr; gap: 18px; }
      .table { width: 100%; border-collapse: collapse; margin-top: 8px; }
      .table th { padding: 9px 10px; background: #0e3855; color: white; font-size: 9px; text-align: left; }
      .table td { padding: 9px 10px; border-bottom: 1px solid #e3ecf2; color: #536d83; font-size: 10px; }
      .table tr:nth-child(even) td { background: #f7fafc; }
      .status { color: #159e94; font-weight: 700; }
      .status.warn { color: #c68520; }
      .bar { height: 7px; margin-top: 5px; border-radius: 99px; background: #e7eef4; overflow: hidden; }
      .bar i { display: block; height: 100%; border-radius: inherit; background: linear-gradient(90deg,#20b89f,#4c72e8); }
      .page-break { page-break-before: always; }
      .section-note { padding: 12px 14px; border-radius: 8px; background: #f5f8fb; color: #6d8193; font-size: 10px; line-height: 1.6; }
      .callout { margin-top: 24px; padding: 15px 17px; border-left: 4px solid #20b89f; background: #effaf7; color: #426979; line-height: 1.6; }
      .footer { margin-top: 55px; padding-top: 12px; border-top: 1px solid #e2eaf0; color: #91a0ad; font-size: 9px; display: flex; justify-content: space-between; }
      @media print { body { background: white; } .sheet { margin: 0; } }
    </style></head><body><main class="sheet">
      <header class="top"><div class="brand"><div class="mark">F</div><div><strong>FinPredict</strong><span>Intelligent Credit Risk</span></div></div>
      <div class="meta">DOCUMENT CONFIDENTIEL<br/>Généré le ${escapeHtml(reportDate())}</div></header>
      <div class="eyebrow">RAPPORT AUDITABLE</div><h1>${escapeHtml(title)}</h1><p class="subtitle">${escapeHtml(subtitle)}</p>
      ${content}
      <footer class="footer"><span>FinPredict AI Pro · Usage interne</span><span>Document généré depuis l’interface de pilotage</span></footer>
    </main></body></html>`);
  reportWindow.document.close();
  reportWindow.focus();
  window.setTimeout(() => reportWindow.print(), 350);
}

function downloadCsv(filename: string, rows: string[][]) {
  const csv = rows.map(row => row.map(value => `"${value.replace(/"/g, '""')}"`).join(';')).join('\r\n');
  const blob = new Blob([`\uFEFF${csv}`], { type: 'text/csv;charset=utf-8' });
  const url = URL.createObjectURL(blob);
  const anchor = document.createElement('a');
  anchor.href = url;
  anchor.download = filename;
  anchor.click();
  URL.revokeObjectURL(url);
}

function ReportsView() {
  const [exporting, setExporting] = useState('');
  const exportRisk = () => {
    setExporting('risk');
    openPrintableReport('Rapport de risque crédit', 'Synthèse exécutive du risque, des seuils et de la qualité du modèle.', `
      <section class="hero"><div class="hero-card"><small>Décisions analysées · période courante</small><strong>2 843</strong><p>Prédictions suivies dans le pipeline de production.</p></div><div class="score"><div><strong>48 %</strong><span>Seuil de décision</span></div></div></section>
      <section class="section"><h2>1. Synthèse exécutive</h2><div class="grid">
        <div class="item"><span>Taux de défaut observé</span><strong>5,8 %</strong></div><div class="item"><span>Qualité des données</span><strong>Stable · 0 anomalie</strong></div>
        <div class="item"><span>Version du modèle</span><strong>Production 1.0.0</strong></div><div class="item"><span>Traçabilité</span><strong>Active · Azure SQL</strong></div>
      </div><p>Sur la période courante, le pipeline a traité 2 843 dossiers. Le seuil opérationnel de 48 % sépare les recommandations <strong>ACCEPT</strong> et <strong>REFUSE</strong>. Les indicateurs présentés sont destinés au pilotage et doivent être rapprochés des données sources avant toute décision.</p></section>
      <section class="section"><h2>2. Distribution des recommandations</h2><table class="table"><thead><tr><th>Segment de risque</th><th>Part du portefeuille</th><th>Interprétation</th><th>Action recommandée</th></tr></thead><tbody>
        <tr><td>Faible · 0–30 %</td><td>72 %<div class="bar"><i style="width:72%"></i></div></td><td>Risque inférieur au seuil interne</td><td class="status">Revue standard</td></tr>
        <tr><td>Modéré · 30–48 %</td><td>18 %<div class="bar"><i style="width:18%"></i></div></td><td>Zone proche de la décision</td><td class="status warn">Revue renforcée</td></tr>
        <tr><td>Élevé · ≥48 %</td><td>10 %<div class="bar"><i style="width:10%"></i></div></td><td>Probabilité au-dessus du seuil</td><td class="status warn">Décision humaine obligatoire</td></tr>
      </tbody></table></section>
      <section class="section"><h2>3. Profil de la qualité et de la disponibilité</h2><div class="two-col"><div><h3>Contrôles de données</h3><ul><li>Schéma d’entrée validé par Pydantic.</li><li>Catégories normalisées et contrôlées.</li><li>Variables dérivées recalculées côté serveur.</li><li>Valeurs manquantes et incohérences surveillées.</li></ul></div><div><h3>Disponibilité opérationnelle</h3><ul><li>API FastAPI : endpoint de santé exposé.</li><li>Artefact de production chargé au démarrage.</li><li>Persistance Azure SQL transactionnelle.</li><li>Référence et hash du dossier journalisés.</li></ul></div></div></section>
      <div class="callout"><strong>Lecture responsable.</strong> Ce document présente une recommandation statistique et ne constitue pas une décision de crédit. Toute décision finale doit être revue selon les procédures internes et les obligations réglementaires applicables.</div>
      <section class="section page-break"><h2>4. Méthodologie du score</h2><p>Le pipeline de production combine une préparation des variables numériques et catégorielles, un modèle de classification calibré et un seuil de décision versionné. La probabilité retournée représente une estimation du risque de défaut pour les caractéristiques fournies ; elle ne constitue pas une certitude individuelle.</p><table class="table"><thead><tr><th>Étape</th><th>Contrôle</th><th>Résultat attendu</th></tr></thead><tbody>
        <tr><td>Validation</td><td>Âge, revenu, montant, taux et catégories</td><td class="status">Requête conforme</td></tr>
        <tr><td>Feature engineering</td><td>Part du revenu et ratio intérêts / revenu</td><td class="status">Calcul serveur</td></tr>
        <tr><td>Inférence</td><td>Pipeline et ordre des 12 variables</td><td class="status">Artefact production</td></tr>
        <tr><td>Décision</td><td>Probabilité comparée au seuil 0,48</td><td class="status">Recommandation</td></tr>
        <tr><td>Journalisation</td><td>Snapshot, hash, version et latence</td><td class="status">Azure SQL</td></tr>
      </tbody></table></section>
      <section class="section"><h2>5. Explicabilité et facteurs de risque</h2><div class="section-note">L’explication par points de probabilité regroupe les variables liées par thème afin de rendre la lecture plus stable. Les groupes disponibles sont : capacité de remboursement, conditions du prêt, historique de crédit, situation de logement et profil du demandeur.</div><div class="two-col"><div><h3>Facteurs susceptibles d’augmenter le risque</h3><ul><li>Part du revenu consacrée au prêt élevée.</li><li>Taux d’intérêt ou montant demandé importants.</li><li>Historique de défaut déclaré.</li></ul></div><div><h3>Facteurs susceptibles de réduire le risque</h3><ul><li>Revenu compatible avec le montant demandé.</li><li>Historique de crédit plus long.</li><li>Conditions du prêt plus favorables.</li></ul></div></div></section>
      <section class="section"><h2>6. Gouvernance, audit et limites</h2><p>Chaque prédiction peut être reliée à une référence de demande, une version de modèle, un seuil, un snapshot des variables et une explication. Les décisions humaines et les données de résultat à maturité doivent compléter cette trace pour évaluer la performance réelle.</p><ul><li>Les données illustratives de cette interface ne remplacent pas un export certifié de la base.</li><li>Les métriques doivent être recalculées sur une période et une population définies.</li><li>Les performances peuvent varier selon le segment, la période et la qualité des données.</li><li>Un analyste habilité doit examiner les cas proches du seuil ou présentant des signaux contradictoires.</li></ul></section>
      <div class="callout"><strong>Plan d’action recommandé.</strong> Confirmer la période de référence, connecter les métriques Azure SQL réelles, documenter les exceptions de revue humaine et programmer une revue mensuelle du drift et de la calibration.</div>`);
    window.setTimeout(() => setExporting(''), 500);
  };
  const exportMonitoring = () => {
    setExporting('monitoring');
    downloadCsv('finpredict-monitoring.csv', [
      ['Métrique', 'Valeur', 'Unité', 'Statut', 'Période'],
      ['Indice de dérive PSI', '0,160', 'PSI', 'Sous le seuil', '7 derniers jours'],
      ['Qualité des données', 'Stable', '', 'OK', 'Période courante'],
      ['Taux de défaut observé', '5,8', '%', 'À surveiller', 'Période courante'],
      ['Prédictions suivies', '2843', 'dossiers', 'Actif', 'Période courante'],
      ['Version du modèle', '1.0.0', '', 'Production', 'Période courante'],
    ]);
    window.setTimeout(() => setExporting(''), 500);
  };
  const exportGovernance = () => {
    setExporting('governance');
    openPrintableReport('Rapport de gouvernance', 'Traçabilité, explicabilité et contrôles de responsabilité du modèle.', `
      <section class="hero"><div class="hero-card"><small>État de la gouvernance</small><strong>Conforme</strong><p>Les contrôles de traçabilité et de versioning sont actifs.</p></div><div class="score"><div><strong>100 %</strong><span>Journalisation</span></div></div></section>
      <section class="section"><h2>1. Synthèse de gouvernance</h2><div class="grid">
        <div class="item"><span>Référence de demande</span><strong>Journalisée</strong></div><div class="item"><span>Version du modèle</span><strong>Versionnée</strong></div>
        <div class="item"><span>Explication des facteurs</span><strong>Disponible</strong></div><div class="item"><span>Décision humaine</span><strong>Requise</strong></div>
      </div><p>Le dispositif de gouvernance encadre l’utilisation du modèle depuis la réception d’une demande jusqu’à la conservation de la prédiction. Il sépare clairement la recommandation statistique de la décision de crédit, qui reste sous la responsabilité d’un analyste habilité.</p></section>
      <section class="section"><h2>2. Matrice des contrôles</h2><table class="table"><thead><tr><th>Domaine</th><th>Contrôle</th><th>Preuve attendue</th><th>Statut</th></tr></thead><tbody>
        <tr><td>Identité du dossier</td><td>Référence unique obligatoire</td><td>application_ref et identifiant SQL</td><td class="status">Actif</td></tr>
        <tr><td>Versioning</td><td>Modèle et seuil enregistrés</td><td>model_version et threshold</td><td class="status">Actif</td></tr>
        <tr><td>Intégrité</td><td>Snapshot des variables haché</td><td>SHA-256 du feature snapshot</td><td class="status">Actif</td></tr>
        <tr><td>Explicabilité</td><td>Facteurs et groupes conservés</td><td>explanation_json</td><td class="status">Actif</td></tr>
        <tr><td>Supervision</td><td>Validation humaine des cas sensibles</td><td>Trace de revue à compléter</td><td class="status warn">Requise</td></tr>
        <tr><td>Résultat réel</td><td>Label de défaut à maturité</td><td>Actual outcome à renseigner</td><td class="status warn">À compléter</td></tr>
      </tbody></table></section>
      <section class="section"><h2>3. Chaîne de traçabilité</h2><div class="two-col"><div><h3>Avant l’inférence</h3><ul><li>Validation du schéma et des contraintes métier.</li><li>Normalisation des catégories autorisées.</li><li>Recalcul serveur des variables dérivées.</li><li>Création d’un snapshot canonique du dossier.</li></ul></div><div><h3>Après l’inférence</h3><ul><li>Probabilité, seuil et recommandation enregistrés.</li><li>Version du modèle et latence conservées.</li><li>Explication des facteurs persistée pour l’audit.</li><li>Référence unique utilisable pour la recherche.</li></ul></div></div><div class="section-note">La chaîne est conçue pour permettre de reconstituer ce qui a été envoyé au modèle, quelle version était active, quelle décision indicative a été retournée et quelle explication était disponible au moment de la prédiction.</div></section>
      <section class="section page-break"><h2>4. Sécurité et protection des données</h2><table class="table"><thead><tr><th>Mesure</th><th>Implémentation</th><th>Objectif</th></tr></thead><tbody>
        <tr><td>Authentification API</td><td>Clé API côté serveur</td><td>Limiter l’accès aux prédictions</td></tr>
        <tr><td>Secrets</td><td>Variables d’environnement</td><td>Ne pas exposer les identifiants SQL au navigateur</td></tr>
        <tr><td>Transport</td><td>Chiffrement TLS vers Azure SQL</td><td>Protéger les échanges</td></tr>
        <tr><td>Minimisation</td><td>Aucun nom ni identifiant direct requis</td><td>Réduire les données personnelles collectées</td></tr>
        <tr><td>Audit applicatif</td><td>Request ID et logs d’accès</td><td>Faciliter l’investigation</td></tr>
      </tbody></table><div class="callout"><strong>Point de vigilance.</strong> Les secrets ne doivent jamais être placés dans une variable <code>NEXT_PUBLIC_*</code>, un rapport exporté ou un journal applicatif. Les exports doivent être conservés dans un espace contrôlé.</div></section>
      <section class="section"><h2>5. Gestion du cycle de vie du modèle</h2><div class="grid">
        <div class="item"><span>Développement</span><strong>Tests unitaires et validation des features</strong></div><div class="item"><span>Validation</span><strong>Qualité, calibration et seuil documentés</strong></div>
        <div class="item"><span>Mise en production</span><strong>Artefact immuable et versionné</strong></div><div class="item"><span>Retrait</span><strong>Statut ACTIVE / RETIRED contrôlé</strong></div>
      </div><p>Un changement de modèle doit produire une nouvelle version, être validé avant activation et conserver la compatibilité avec le schéma des variables. Le seuil enregistré dans Azure SQL doit rester cohérent avec celui de l’artefact de production.</p></section>
      <section class="section"><h2>6. Explicabilité et équité</h2><div class="two-col"><div><h3>Explicabilité</h3><ul><li>Probabilité finale et seuil affichés.</li><li>Facteurs classés par impact relatif.</li><li>Variables regroupées par thème métier.</li><li>Message rappelant le caractère indicatif.</li></ul></div><div><h3>Contrôles à maintenir</h3><ul><li>Surveiller les écarts de performance par segment.</li><li>Documenter les variables sensibles et leurs proxys.</li><li>Revoir les résultats proches du seuil.</li><li>Archiver les rapports de fairness.</li></ul></div></div><div class="section-note">Une explication de modèle décrit les signaux utilisés par le pipeline ; elle ne constitue pas une preuve de causalité et ne remplace pas l’analyse humaine du dossier.</div></section>
      <section class="section"><h2>7. Plan de contrôle périodique</h2><table class="table"><thead><tr><th>Fréquence</th><th>Revue</th><th>Responsable</th><th>Livrable</th></tr></thead><tbody>
        <tr><td>À chaque prédiction</td><td>Validation, seuil et journalisation</td><td>API / système</td><td>Trace SQL</td></tr>
        <tr><td>Hebdomadaire</td><td>Disponibilité, erreurs et drift</td><td>Équipe plateforme</td><td>Rapport monitoring</td></tr>
        <tr><td>Mensuelle</td><td>Calibration, segments et fairness</td><td>Risque modèle</td><td>Revue de performance</td></tr>
        <tr><td>À chaque version</td><td>Validation et approbation du modèle</td><td>Gouvernance</td><td>Fiche de changement</td></tr>
      </tbody></table></section>
      <section class="section"><h2>8. Limites et décisions à prendre</h2><ul><li>Les métriques affichées dans ce rapport doivent être remplacées par les agrégats certifiés Azure SQL.</li><li>La journalisation technique ne suffit pas à démontrer l’absence de biais ou de discrimination.</li><li>Les décisions humaines, motifs de dérogation et résultats réels doivent être enregistrés pour compléter l’audit.</li><li>Les règles de conservation, d’accès et de suppression des rapports doivent être définies par l’organisation.</li></ul></section>
      <div class="callout"><strong>Principe de responsabilité.</strong> Le modèle assiste l’analyse. Il n’accorde ni ne refuse juridiquement un prêt. Toute décision doit être prise selon les procédures internes, avec une revue humaine documentée et un accès aux éléments justificatifs.</div>
      <section class="section"><h2>9. Feuille de route recommandée</h2><p>Priorités proposées : connecter les événements de revue humaine, renseigner les labels de résultat à maturité, automatiser les contrôles de drift et fairness, formaliser la fiche de changement de modèle et définir une politique de conservation des exports.</p></section>`);
    window.setTimeout(() => setExporting(''), 500);
  };
  const actionLabel = (key: string, label: string) => exporting === key ? <><span className="spinner" /> Préparation…</> : <><Download size={15} /> {label}</>;
  return <><PageHeader eyebrow="DOCUMENTATION" title="Rapports & exports" description="Préparez des synthèses auditables sur l’activité et les performances du modèle." /><div className="reports-grid">
    <article className="card report-card"><div className="report-icon blue"><FileBarChart2 size={24} /></div><h2>Rapport de risque crédit</h2><p>Synthèse premium des métriques, du seuil de décision et des contrôles de responsabilité.</p><Badge kind="success">Prêt à exporter</Badge><button className="button button-secondary" onClick={exportRisk} disabled={Boolean(exporting)}>{actionLabel('risk', 'Exporter PDF')}</button></article>
    <article className="card report-card"><div className="report-icon green"><Activity size={24} /></div><h2>Rapport de monitoring</h2><p>Export structuré des indicateurs de drift, qualité de données, disponibilité et version.</p><Badge kind="success">Données disponibles</Badge><button className="button button-secondary" onClick={exportMonitoring} disabled={Boolean(exporting)}>{actionLabel('monitoring', 'Exporter CSV')}</button></article>
    <article className="card report-card"><div className="report-icon purple"><ShieldCheck size={24} /></div><h2>Rapport de gouvernance</h2><p>Document auditable sur la traçabilité, l’explicabilité et la supervision humaine.</p><Badge kind="blue">Structure prête</Badge><button className="button button-secondary" onClick={exportGovernance} disabled={Boolean(exporting)}>{actionLabel('governance', 'Générer un rapport')}</button></article>
  </div><p className="reports-export-note"><FileText size={14} /> Les PDF s’ouvrent dans une mise en page imprimable : choisissez <strong>Enregistrer au format PDF</strong> dans la boîte d’impression.</p></>;
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
