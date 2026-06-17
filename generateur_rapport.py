from reportlab.lib.pagesizes import A4
from reportlab.lib import colors
from reportlab.lib.styles import getSampleStyleSheet, ParagraphStyle
from reportlab.lib.units import cm
from reportlab.platypus import (SimpleDocTemplate, Paragraph, Spacer,
                                 Table, TableStyle, HRFlowable)
from reportlab.lib.enums import TA_CENTER, TA_LEFT
from datetime import datetime
import io

# Palette couleurs du projet
BLEU_FONCE  = colors.HexColor('#0F172A')
BLEU_CLAIR  = colors.HexColor('#06B6D4')
SLATE       = colors.HexColor('#1E293B')
GRIS_TEXTE  = colors.HexColor('#94A3B8')
VERT        = colors.HexColor('#10B981')
ROUGE       = colors.HexColor('#F87171')
JAUNE       = colors.HexColor('#FBBF24')
BLANC       = colors.white

def couleur_decision(decision: str):
    if "APPROUVÉ" in decision:  return VERT
    if "REJETÉ"   in decision:  return ROUGE
    return JAUNE

def generer_rapport_pdf(
    # Infos client
    salaire:     int,
    montant:     int,
    grade:       str,
    intention:   str,
    # Résultats
    score_ml:    float,
    score_qdrant: float,
    score_final: float,
    decision:    str,
    # SHAP
    favorables:   list,   # [{'feature': str, 'impact': float}, ...]
    defavorables: list,
    explication:  str,
    # Qdrant
    nb_similaires: int,
    ratio_rentable: float,
) -> bytes:
    """
    Génère un rapport PDF complet et retourne les bytes.
    Usage : st.download_button(..., data=generer_rapport_pdf(...))
    """

    buffer = io.BytesIO()
    doc    = SimpleDocTemplate(
        buffer, pagesize=A4,
        leftMargin=2*cm, rightMargin=2*cm,
        topMargin=2*cm,  bottomMargin=2*cm
    )

    styles = getSampleStyleSheet()

    # --- STYLES PERSONNALISÉS ---
    titre_style = ParagraphStyle(
        'Titre', fontSize=20, textColor=BLANC,
        backColor=BLEU_FONCE, alignment=TA_CENTER,
        spaceAfter=6, spaceBefore=6, leading=28,
        leftIndent=-1*cm, rightIndent=-1*cm,
        borderPad=12
    )
    sous_titre_style = ParagraphStyle(
        'SousTitre', fontSize=11, textColor=GRIS_TEXTE,
        alignment=TA_CENTER, spaceAfter=16
    )
    section_style = ParagraphStyle(
        'Section', fontSize=12, textColor=BLEU_CLAIR,
        spaceBefore=14, spaceAfter=6, fontName='Helvetica-Bold'
    )
    body_style = ParagraphStyle(
        'Body', fontSize=10, textColor=colors.black,
        spaceAfter=4, leading=16
    )
    bold_style = ParagraphStyle(
        'Bold', fontSize=10, fontName='Helvetica-Bold',
        textColor=colors.black
    )

    elements = []
    noms_lisibles = {
        'taux':              'Taux d\'intérêt',
        'montant':           'Montant du prêt',
        'revenu':            'Revenu annuel',
        'ratio_endettement': 'Ratio d\'endettement',
        'effort_mensuel':    'Effort mensuel',
        'grade_enc':         'Grade de crédit',
        'defaut_passe_enc':  'Défaut passé',
        'ratio_revenu':      'Part prêt/revenu',
        'historique_credit': 'Historique crédit',
    }

    # ─── EN-TÊTE ───────────────────────────────────────────────
    elements.append(Paragraph("🏦 FinPredict AI Pro", titre_style))
    elements.append(Paragraph(
        f"Rapport de décision de prêt — {datetime.now().strftime('%d/%m/%Y à %H:%M')}",
        sous_titre_style
    ))
    elements.append(HRFlowable(width="100%", thickness=1,
                                color=BLEU_CLAIR, spaceAfter=10))

    # ─── DÉCISION PRINCIPALE ───────────────────────────────────
    couleur = couleur_decision(decision)
    decision_data = [[
        Paragraph("Décision finale", ParagraphStyle(
            'DC', fontSize=11, textColor=GRIS_TEXTE, alignment=TA_CENTER)),
        Paragraph("Score de confiance", ParagraphStyle(
            'DC', fontSize=11, textColor=GRIS_TEXTE, alignment=TA_CENTER)),
        Paragraph("Score ML", ParagraphStyle(
            'DC', fontSize=11, textColor=GRIS_TEXTE, alignment=TA_CENTER)),
    ],[
        Paragraph(f"<b>{decision}</b>", ParagraphStyle(
            'DV', fontSize=16, textColor=couleur, alignment=TA_CENTER)),
        Paragraph(f"<b>{score_final:.1f}%</b>", ParagraphStyle(
            'DV', fontSize=16, textColor=BLEU_CLAIR, alignment=TA_CENTER)),
        Paragraph(f"<b>{score_ml:.1f}%</b>", ParagraphStyle(
            'DV', fontSize=16, textColor=colors.black, alignment=TA_CENTER)),
    ]]

    t = Table(decision_data, colWidths=[5.5*cm, 5.5*cm, 5.5*cm])
    t.setStyle(TableStyle([
        ('BACKGROUND', (0,0), (-1,0), SLATE),
        ('BACKGROUND', (0,1), (-1,1), colors.HexColor('#F8FAFC')),
        ('GRID',       (0,0), (-1,-1), 0.5, GRIS_TEXTE),
        ('ROUNDEDCORNERS', [6]),
        ('TOPPADDING',    (0,0), (-1,-1), 10),
        ('BOTTOMPADDING', (0,0), (-1,-1), 10),
    ]))
    elements.append(t)
    elements.append(Spacer(1, 14))

    # ─── PROFIL CLIENT ─────────────────────────────────────────
    elements.append(Paragraph("📋 Profil du demandeur", section_style))
    client_data = [
        ["Revenu annuel",   f"{salaire:,} €".replace(',', ' ')],
        ["Montant demandé", f"{montant:,} €".replace(',', ' ')],
        ["Grade de crédit", grade],
        ["Intention",       intention],
        ["Ratio d'endettement", f"{montant/(salaire+1)*100:.1f}%"],
    ]
    t2 = Table(client_data, colWidths=[7*cm, 9.5*cm])
    t2.setStyle(TableStyle([
        ('FONTNAME',      (0,0), (0,-1), 'Helvetica-Bold'),
        ('TEXTCOLOR',     (0,0), (0,-1), colors.HexColor('#334155')),
        ('BACKGROUND',    (0,0), (-1,-1), colors.HexColor('#F8FAFC')),
        ('ROWBACKGROUNDS',(0,0), (-1,-1),
         [colors.white, colors.HexColor('#F1F5F9')]),
        ('GRID',          (0,0), (-1,-1), 0.3, GRIS_TEXTE),
        ('TOPPADDING',    (0,0), (-1,-1), 7),
        ('BOTTOMPADDING', (0,0), (-1,-1), 7),
        ('LEFTPADDING',   (0,0), (-1,-1), 10),
    ]))
    elements.append(t2)
    elements.append(Spacer(1, 14))

    # ─── ANALYSE SHAP ──────────────────────────────────────────
    elements.append(Paragraph("🔍 Facteurs de décision (analyse SHAP)", section_style))
    elements.append(Paragraph(explication, body_style))
    elements.append(Spacer(1, 8))

    shap_header = [
        Paragraph("<b>Facteur</b>", bold_style),
        Paragraph("<b>Direction</b>", bold_style),
        Paragraph("<b>Impact</b>", bold_style),
    ]
    shap_rows = [shap_header]

    for f in (favorables + defavorables):
        nom    = noms_lisibles.get(f['feature'], f['feature'])
        impact = f['impact']
        dir_   = "✅ Favorable" if impact > 0 else "⚠️ Limitant"
        couleur_txt = VERT if impact > 0 else ROUGE
        shap_rows.append([
            Paragraph(nom, body_style),
            Paragraph(dir_, ParagraphStyle(
                'D', fontSize=10, textColor=couleur_txt)),
            Paragraph(f"{abs(impact):.4f}", body_style),
        ])

    t3 = Table(shap_rows, colWidths=[7*cm, 5*cm, 4.5*cm])
    t3.setStyle(TableStyle([
        ('BACKGROUND',    (0,0), (-1,0), SLATE),
        ('TEXTCOLOR',     (0,0), (-1,0), BLANC),
        ('ROWBACKGROUNDS',(0,1), (-1,-1),
         [colors.white, colors.HexColor('#F1F5F9')]),
        ('GRID',          (0,0), (-1,-1), 0.3, GRIS_TEXTE),
        ('TOPPADDING',    (0,0), (-1,-1), 7),
        ('BOTTOMPADDING', (0,0), (-1,-1), 7),
        ('LEFTPADDING',   (0,0), (-1,-1), 10),
    ]))
    elements.append(t3)
    elements.append(Spacer(1, 14))

    # ─── CONTEXTE QDRANT ───────────────────────────────────────
    elements.append(Paragraph("📊 Contexte historique (Qdrant)", section_style))
    qdrant_data = [
        ["Dossiers similaires analysés", str(nb_similaires)],
        ["Taux de rentabilité historique", f"{ratio_rentable:.1f}%"],
        ["Score Qdrant",                  f"{score_qdrant:.1f}%"],
    ]
    t4 = Table(qdrant_data, colWidths=[9*cm, 7.5*cm])
    t4.setStyle(TableStyle([
        ('FONTNAME',      (0,0), (0,-1), 'Helvetica-Bold'),
        ('ROWBACKGROUNDS',(0,0), (-1,-1),
         [colors.white, colors.HexColor('#F1F5F9')]),
        ('GRID',          (0,0), (-1,-1), 0.3, GRIS_TEXTE),
        ('TOPPADDING',    (0,0), (-1,-1), 7),
        ('BOTTOMPADDING', (0,0), (-1,-1), 7),
        ('LEFTPADDING',   (0,0), (-1,-1), 10),
    ]))
    elements.append(t4)
    elements.append(Spacer(1, 20))

    # ─── PIED DE PAGE ──────────────────────────────────────────
    elements.append(HRFlowable(width="100%", thickness=0.5,
                                color=GRIS_TEXTE))
    elements.append(Paragraph(
        "Document généré automatiquement par FinPredict AI Pro — "
        "Usage interne uniquement — Confidentiel",
        ParagraphStyle('Footer', fontSize=8,
                       textColor=GRIS_TEXTE, alignment=TA_CENTER)
    ))

    doc.build(elements)
    return buffer.getvalue()