"""
Backend de rétrosynthèse — FastAPI + RDKit (v3).

Améliorations vs v2 :
  - mémoïsation des disconnections (lru_cache) -> recherches profondes viables ;
  - VALIDATION des règles au démarrage : une SMIRKS invalide est LOGUÉE, pas
    avalée en silence ;
  - scoring réel par route (fiabilité x résolu x longueur) ;
  - stock branchable depuis un fichier .smi (sinon repli sur la taille) ;
  - déduplication des routes + logs explicites ;
  - conserve l'arbre ET/OU et l'anti-cycle par chemin.

Lancer :  py -m uvicorn backend:app --reload --host 0.0.0.0 --port 8000
Stock   :  définir la variable d'env  RETRO_STOCK_FILE=chemin/vers/stock.smi  (optionnel)
"""

from __future__ import annotations

import json
import logging
import os
import re
import time
import urllib.error
import urllib.parse
import urllib.request
from functools import lru_cache
from itertools import product as iproduct

from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field
from rdkit import Chem, RDLogger
from rdkit.Chem import AllChem
from rdkit.Chem.Draw import rdMolDraw2D

RDLogger.DisableLog("rdApp.*")
logging.basicConfig(level=logging.INFO, format="%(levelname)s %(message)s")
logger = logging.getLogger("retro")

app = FastAPI(title="RetroSynthesis API", version="3.0.0")
app.add_middleware(CORSMiddleware, allow_origins=["*"], allow_methods=["*"],
                   allow_headers=["*"])

# ---------------------------------------------------------------------------
# RÈGLES : (SMIRKS, catégorie, fiabilité 0-1, conditions indicatives).
# Familles ester/amide éclatées en voies réalistes distinctes (Fischer,
# chlorure d'acyle, anhydride acétique) pour faire remonter toutes les voies
# non exotiques plutôt qu'une seule disconnection générique. Les règles
# "Williamson / Suzuki / amination réductrice / rétro-aldol" restent des
# POINTS DE DÉPART à valider sur tes propres molécules (non exécutées avec
# RDKit dans cet environnement). Toute SMIRKS qui ne compile pas est signalée
# au démarrage, jamais avalée en silence.
# ---------------------------------------------------------------------------
_RAW_RULES: dict[str, tuple[str, str, float, str]] = {
    # --- Famille ester : 4 voies réalistes distinctes, fiabilité reflétant
    # l'usage réel en synthèse (pas seulement la faisabilité théorique). ---
    "Estérification de Fischer (ester alkylique)":
        ("[C:1](=[O:2])[O:3][CX4:4]>>[C:1](=[O:2])[OH].[OH:3][CX4:4]",
         "Acylation", 0.85,
         "acide carboxylique + alcool, H2SO4 (cat.), reflux, Dean-Stark (élim. H2O)"),
    "Estérification de Fischer (ester aryle)":
        ("[C:1](=[O:2])[O:3][c:4]>>[C:1](=[O:2])[OH].[OH:3][c:4]",
         "Acylation", 0.40,
         "acide + phénol, H2SO4 conc., reflux prolongé — équilibre défavorable, "
         "rarement utilisé en pratique sur un phénol"),
    "Estérification de Steglich (acide + alcool/phénol, DCC)":
        ("[C:1](=[O:2])[O:3][#6:4]>>[C:1](=[O:2])[OH].[OH:3][#6:4]",
         "Acylation", 0.82,
         "acide carboxylique + alcool OU phénol, DCC + DMAP (cat.), CH2Cl2, T.A. — "
         "réussit là où Fischer échoue sur les phénols/alcools encombrés"),
    "Estérification via chlorure d'acyle (Schotten-Baumann)":
        ("[C:1](=[O:2])[O:3][#6:4]>>[C:1](=[O:2])[Cl].[OH:3][#6:4]",
         "Acylation", 0.90,
         "chlorure d'acyle + alcool/phénol, base (pyridine ou Et3N), 0 °C → T.A."),
    "Acétylation par anhydride acétique (ester)":
        ("[#6:5][O:3][C:1](=[O:2])[CH3:4]>>[#6:5][O:3][H].[CH3:4][C:1](=[O:2])OC(C)=O",
         "Acylation", 0.95,
         "anhydride acétique, catalyseur acide (H2SO4/H3PO4) ou base (pyridine), 80–90 °C"),
    # --- Famille amide : mêmes 3 voies (Fischer non applicable aux amides). ---
    "Couplage peptidique (acide + amine, agent activant)":
        ("[C:1](=[O:2])[N;!$([NX3]=*):3][#6:4]>>[C:1](=[O:2])[OH].[N:3][#6:4]",
         "Acylation", 0.80,
         "acide carboxylique + amine, DCC ou EDC, DMAP (cat.), CH2Cl2, T.A."),
    "Amidation via chlorure d'acyle (Schotten-Baumann)":
        ("[C:1](=[O:2])[N;!$([NX3]=*):3][#6:4]>>[C:1](=[O:2])[Cl].[N:3][#6:4]",
         "Acylation", 0.90,
         "chlorure d'acyle + amine, base (NaOH aq. ou Et3N), 0 °C"),
    "Acétylation par anhydride acétique (amide)":
        ("[#6:5][N;!$([NX3]=*):3][C:1](=[O:2])[CH3:4]"
         ">>[#6:5][N:3][H].[CH3:4][C:1](=[O:2])OC(C)=O",
         "Acylation", 0.92,
         "anhydride acétique, T.A. → reflux (ex. paracétamol, acétanilide)"),
    # --- Autres familles (inchangées, conditions ajoutées). ---
    "Élimination de Hofmann":
        ("[N+:1]([C:2])([C:3])([C:4])[C:5][C:6]>>[N:1]([C:2])([C:3])[C:4].[C:5]=[C:6]",
         "Alcaloïde", 0.60,
         "sel d'ammonium quaternaire, Ag2O puis chauffage — alcène le moins substitué favorisé"),
    "Rétro-Grignard (alcool sec.)":
        ("[OH:1][C:2]([C:3])[C:4]>>[C:2](=O)[C:4].[C:3][Mg]Br",
         "C–C", 0.70,
         "organomagnésien R-MgBr, THF anhydre, 0 °C → T.A., puis hydrolyse acide (NH4Cl aq.)"),
    # --- À VALIDER ---
    "Éther de Williamson":
        ("[CX4:1][O:2][CX4:3]>>[CX4:1][OH:2].[CX4:3][Cl]",
         "Substitution", 0.75,
         "alcoolate (NaH + R-OH) + halogénure R'-X, DMF/THF, T.A. → reflux (SN2)"),
    "Couplage de Suzuki (biaryle)":
        ("[c:1]!@[c:2]>>[c:1][Br].[c:2]B(O)O",
         "Couplage C–C", 0.90,
         "acide boronique + halogénure d'aryle, Pd(PPh3)4 (cat.), K2CO3, toluène/EtOH/H2O, reflux"),
    "Amination réductrice":
        ("[CX4;!$([CX4]([#7])[#7]):1][NX3;!$([NX3]C=O):2]>>[C:1]=O.[N:2]",
         "C–N", 0.80,
         "amine + aldéhyde/cétone, NaBH(OAc)3 ou NaBH3CN, T.A. (AcOH cat. parfois)"),
    "Rétro-aldol (β-hydroxy carbonyle)":
        ("[C:1](=[O:2])[CH2:3][CH:4][OX2H:5]>>[C:1](=[O:2])[CH3:3].[CH:4]=[O:5]",
         "C–C", 0.65,
         "aldolisation : base (LDA, NaOH) ou catalyse acide — sensible à la régiosélectivité"),
    # --- Chaînon manquant : sans cette règle, TOUT chlorure d'acyle généré par
    # une disconnection Schotten-Baumann (ester ou amide) était un cul-de-sac
    # définitif (aucune des règles précédentes ne sait re-décomposer un
    # C(=O)Cl), même si l'acide parent est trivial/en stock. Reconnecte
    # systématiquement ces branches à l'acide carboxylique correspondant. ---
    "Chloration d'acide carboxylique (accès au chlorure d'acyle)":
        ("[CX3:1](=[O:2])[Cl:3]>>[CX3:1](=[O:2])[OH]",
         "Activation", 0.92,
         "SOCl2 (reflux, dégagement HCl/SO2) ou (COCl)2 + DMF cat., CH2Cl2, T.A. — "
         "quasi quantitatif, réaction standard d'activation d'acide"),

    # === CLIVAGES BASIQUES (hydrolyse / aminolyse) ===
    # NB : seules les bases FORTES (NaOH/KOH) et les nucléophiles azotés
    # (amines, NH3) clivent réellement. Les carbonates (K2CO3, Na2CO3) et
    # bicarbonates (NaHCO3) ne clivent PAS : ce sont des bases de réaction
    # (déprotonation pour Williamson) ou de lavage — mentionnées en conditions
    # des règles concernées, pas comme règles de clivage à part entière.

    # Amide → acide + amine : hydrolyse (retour aux deux précurseurs séparés,
    # plus petits). Conditions dures car l'amide est le dérivé d'acide le plus
    # stable.
    "Amide → acide + amine (hydrolyse)":
        ("[C:1](=[O:2])[NX3:3]>>[C:1](=[O:2])[OH].[N:3]",
         "Clivage", 0.60,
         "amide + NaOH/KOH aqueux à reflux prolongé, OU H3O+ (H2SO4/HCl, Δ) — "
         "hydrolyse dure ; l'amide est le carbonyle le plus résistant"),

    # Amide ← aminolyse d'un ester : voie DOUCE de formation d'amide, un ester
    # réagit avec une amine (ex. éthanolamine) -> amide + alcool. En rétro,
    # l'amide peut donc venir d'un ester + amine.
    "Amide ← aminolyse d'ester (amine + ester)":
        ("[C:1](=[O:2])[NX3:3]>>[C:1](=[O:2])OC.[N:3]",
         "Clivage", 0.58,
         "ester + amine (ex. éthanolamine, NH3, amine 1aire/2aire), chauffage "
         "modéré, sans catalyseur — aminolyse ; plus doux que via chlorure d'acyle"),

    # === VAGUE 1 : familles de fonctions supplémentaires (chaque SMIRKS testé
    # avec RDKit sur molécules réelles avant intégration : compile + précurseurs
    # chimiquement corrects). Toutes ont une synthèse inverse réelle. ===

    # Nitrile : un acide (ou amide) peut provenir de l'hydrolyse d'un nitrile,
    # lui-même souvent obtenu par substitution (R-X + CN⁻). Ouvre une voie vers
    # des précurseurs plus courts d'un carbone.
    "Hydrolyse de nitrile → acide carboxylique":
        ("[C:1](=[O:2])[OH:3]>>[C:1]#[N:2]",
         "Fonction C≡N", 0.70,
         "nitrile + H2O, hydrolyse acide (H2SO4, reflux) ou basique (NaOH) — "
         "le nitrile provient typiquement de R–X + NaCN"),

    # Friedel-Crafts : une cétone aryle-alkyle se déconnecte en arène + chlorure
    # d'acyle (acylation, pas d'alkylation — évite les réarrangements).
    "Acylation de Friedel-Crafts (cétone aromatique)":
        ("[c:1][C:2](=[O:3])[#6:4]>>[c:1][H].[Cl][C:2](=[O:3])[#6:4]",
         "Aromatique", 0.78,
         "arène + chlorure d'acyle, AlCl3 (≥1 équiv.), CH2Cl2 ou sans solvant, "
         "0 °C → T.A. — acylation (pas d'alkylation, évite les réarrangements)"),

    # Halogénure d'alkyle ← alcool : brique de substitution/Grignard, ramène à
    # l'alcool correspondant (souvent en stock ou réductible).
    "Halogénure d'alkyle ← alcool":
        ("[CX4:1][Cl,Br,I:2]>>[CX4:1][OH]",
         "Substitution", 0.80,
         "alcool + SOCl2/PBr3 (ou HX) — accès à l'halogénure depuis l'alcool"),

    # Nitration aromatique : Ar-NO2 vient de la nitration de l'arène nu.
    "Nitration aromatique":
        ("[c:1][N+:2](=[O:3])[O-:4]>>[c:1][H]",
         "Aromatique", 0.85,
         "arène + HNO3/H2SO4 (mélange sulfonitrique), 0–50 °C — "
         "orientation dictée par les substituants présents"),

    # Aniline ← réduction du nitro : voie d'accès classique à une aniline.
    "Aniline ← réduction du groupe nitro":
        ("[c:1][NH2:2]>>[c:1][N+](=O)[O-]",
         "Aromatique", 0.88,
         "nitroarène + H2/Pd-C, ou Fe/HCl, ou SnCl2 — réduction du nitro en amine"),

    # Alcool primaire ← réduction d'acide/ester (alkyle et benzylique).
    "Alcool primaire ← réduction d'acide/ester":
        ("[#6:1][CH2:2][OH:3]>>[#6:1][C:2](=O)[OH]",
         "Réduction", 0.75,
         "acide carboxylique ou ester + LiAlH4 (THF, reflux) — réduction en "
         "alcool primaire"),

    # Alcool secondaire ← réduction de cétone (voie alternative au Grignard).
    "Alcool secondaire ← réduction de cétone":
        ("[#6:1][CH:2]([#6:4])[OH:3]>>[#6:1][C:2]([#6:4])=O",
         "Réduction", 0.78,
         "cétone + NaBH4 (MeOH, 0 °C → T.A. ; doux, sélectif du carbonyle) "
         "ou LiAlH4 — réduction en alcool secondaire"),

    # === RÉDUCTIONS DIFFÉRENCIÉES PAR SÉLECTIVITÉ DU RÉDUCTEUR ===
    # Chaque source d'hydrure ne touche que certaines fonctions ; on encode
    # la fonction ET le réducteur adapté, testé RDKit avant intégration.

    # Amine ← réduction d'amide par LiAlH4 (NaBH4 ne réduit PAS l'amide).
    # C=O de l'amide entièrement retiré -> CH2-N.
    "Amine ← réduction d'amide (LiAlH4)":
        ("[#6:1][CH2:2][NX3:3]>>[#6:1][C:2](=O)[N:3]",
         "Réduction", 0.70,
         "amide + LiAlH4 (THF, reflux) — réduction complète C=O → CH2 ; "
         "NaBH4 est INEFFICACE sur les amides"),

    # Amine primaire ← réduction de nitrile par LiAlH4 (homologation : R-CN
    # donne R-CH2-NH2, +1 carbone). Voie complémentaire de l'hydrolyse du nitrile.
    "Amine primaire ← réduction de nitrile (LiAlH4)":
        ("[#6:1][CH2:2][NH2:3]>>[#6:1][C:2]#[N:3]",
         "Réduction", 0.70,
         "nitrile + LiAlH4 (THF) ou H2/Ni Raney — réduction en amine primaire "
         "(+1 C par rapport au substrat du nitrile)"),

    # Aldéhyde ← réduction PARTIELLE d'ester par DIBAL-H à froid (s'arrête à
    # l'aldéhyde, ne va pas jusqu'à l'alcool si 1 équiv à -78 °C).
    "Aldéhyde ← réduction partielle d'ester (DIBAL-H)":
        ("[#6:1][CH:2]=[O:3]>>[#6:1][C:2](=[O:3])OC",
         "Réduction", 0.60,
         "ester + DIBAL-H (1 équiv, toluène, -78 °C) — s'arrête à l'aldéhyde ; "
         "contrôle strict de T° et stœchiométrie requis"),

    # Méthylène ← désoxygénation totale d'une cétone aryle (Clemmensen en
    # milieu acide OU Wolff-Kishner en milieu basique). C=O -> CH2.
    "Méthylène ← désoxygénation de cétone (Clemmensen / Wolff-Kishner)":
        ("[c:1][CH2:2][#6:3]>>[c:1][C:2](=O)[#6:3]",
         "Réduction", 0.66,
         "cétone → CH2 : Clemmensen (Zn-Hg, HCl conc. — substrats stables en "
         "acide) OU Wolff-Kishner (N2H4 puis KOH, Δ — substrats stables en base)"),

    # Alcane ← hydrogénation d'un alcène, RESTREINTE au cas benzylique
    # (styrène-like) : la version générique [CX4][CX4] matcherait toute liaison
    # C–C et générerait un bruit d'alcènes absurdes (vérifié sur l'ibuprofène).
    "Alcène benzylique ← (précède hydrogénation H2/Pd)":
        ("[c:0][CX4:1][CX4H2,CX4H3:2]>>[c:0][C:1]=[C:2]",
         "Réduction", 0.55,
         "alcène (styrénique) + H2/Pd-C, 1 atm → alcane ; en rétro, ce carbone "
         "benzylique peut provenir d'un alcène conjugué à l'arène"),

    # Cétone/aldéhyde ← oxydation d'alcool (voie d'accès au carbonyle).
    "Cétone/aldéhyde ← oxydation d'alcool":
        ("[#6:1][C:2](=[O:3])[#6:4]>>[#6:1][CH:2]([OH])[#6:4]",
         "Oxydation", 0.72,
         "alcool + PCC/PDC (CH2Cl2) ou Swern ou Dess-Martin — oxydation ménagée "
         "en carbonyle sans sur-oxydation"),

    # === VAGUE 2 : substitutions, aromatiques électrophiles, redox complémentaires. ===

    # Nitrile aliphatique ← substitution : R-CH2-CN vient de R-CH2-X + CN⁻.
    # (Ne matche pas Ar-CN : correct, la SN2 ne s'applique pas sur aryle.)
    "Nitrile ← substitution (R–X + cyanure)":
        ("[CX4:1][C:2]#[N:3]>>[CX4:1][Cl].[C:2]#[N:3]",
         "Substitution", 0.72,
         "halogénure d'alkyle + NaCN/KCN, DMSO ou DMF, chauffage — homologation +1 C"),

    # Éther aryl-alkyle : Williamson sur phénolate + halogénure d'alkyle.
    "Éther aryl-alkyle (Williamson sur phénol)":
        ("[c:1][O:2][CX4:3]>>[c:1][OH:2].[CX4:3][Br]",
         "Substitution", 0.80,
         "phénol + base (K2CO3) + halogénure d'alkyle, DMF/acétone, reflux (SN2)"),

    # Halogénation aromatique électrophile : Ar-X ← Ar-H.
    "Halogénation aromatique":
        ("[c:1][Cl,Br:2]>>[c:1][H]",
         "Aromatique", 0.82,
         "arène + Cl2/Br2, catalyse FeX3 ou AlX3 — SEAr, orientation par les "
         "substituants"),

    # Sulfonation aromatique : Ar-SO3H ← Ar-H (réversible, utile comme groupe bloquant).
    "Sulfonation aromatique":
        ("[c:1][S:2](=[O:3])(=[O:4])[OH:5]>>[c:1][H]",
         "Aromatique", 0.75,
         "arène + H2SO4 fumant (SO3), chaud — réaction réversible, souvent "
         "employée comme groupe bloquant temporaire"),

    # Acide ← oxydation d'un alcool primaire (voie complémentaire à l'hydrolyse
    # de nitrile et à la réduction).
    "Acide carboxylique ← oxydation d'alcool primaire":
        ("[#6:1][C:2](=[O:3])[OH:4]>>[#6:1][CH2:2][OH]",
         "Oxydation", 0.74,
         "alcool primaire + KMnO4 ou Jones (CrO3/H2SO4) — oxydation poussée "
         "jusqu'à l'acide"),

    # Alcène ← déshydratation d'un alcool (Zaitsev).
    "Alcène ← déshydratation d'alcool":
        ("[C:1]=[C:2]>>[C:1][C:2][OH]",
         "Élimination", 0.68,
         "alcool + H2SO4/H3PO4 conc., chaud (ou POCl3/pyridine) — élimination "
         "E1, alcène le plus substitué (Zaitsev) favorisé"),

    # Amine ← alkylation directe : fiabilité VOLONTAIREMENT basse — la
    # mono-alkylation d'une amine par un halogénure est difficile à arrêter
    # (suralkylation jusqu'à l'ammonium). Reste proposée mais jamais en tête ;
    # l'amination réductrice (déjà présente) est la vraie voie recommandée.
    "Amine ← alkylation directe (suralkylation probable)":
        ("[#6:1][NH:2][CX4:3]>>[#6:1][NH2:2].[CX4:3][Br]",
         "Substitution", 0.35,
         "amine + halogénure d'alkyle — ATTENTION suralkylation difficile à "
         "contrôler ; préférer l'amination réductrice quand c'est possible"),

    # === VAGUE 3 : chimie du nitro (voies aldéhyde → nitroalcène → amine). ===

    # Amine primaire ← réduction d'un nitroalcène. Ouvre la voie "aldéhyde +
    # nitroalcane" classique (ex. amphétamine ← phényl-2-nitropropène). La
    # réduction (LiAlH4, ou H2/cat., ou Zn/HCl, ou Al-Hg) réduit à la fois la
    # double liaison C=C et le groupe NO2 en NH2. Fiabilité modérée : la
    # position de la double liaison régénérée n'est pas unique (le moteur peut
    # proposer 2 régiochimies de nitroalcène ; le chimiste tranche).
    "Amine primaire ← réduction de nitroalcène":
        ("[CX4:1][CX4:2][NH2:3]>>[C:1]=[C:2][N+](=O)[O-]",
         "Réduction", 0.62,
         "nitroalcène + réducteur (LiAlH4 THF ; ou H2/Pd ; ou Zn ou Al-Hg) — "
         "réduit simultanément C=C et NO2→NH2"),

    # Nitroalcène ← condensation de Henry (nitroaldol déshydraté) : aldéhyde +
    # nitroalcane. Les nitroalcanes courts (nitrométhane, nitroéthane) sont des
    # briques de base commerciales.
    "Nitroalcène ← condensation de Henry (aldéhyde + nitroalcane)":
        ("[c,C:1][CH:2]=[C:3][N+:4](=[O:5])[O-:6]>>[c,C:1][CH:2]=[O].[CH2:3][N+:4](=[O:5])[O-:6]",
         "C–C", 0.68,
         "aldéhyde + nitroalcane, base catalytique (amine, NH4OAc), puis "
         "déshydratation — condensation de Henry / nitroaldol"),
}

RULES: dict[str, AllChem.ChemicalReaction] = {}
RULE_RELIABILITY: dict[str, float] = {}
RULE_CATEGORY: dict[str, str] = {}
RULE_CONDITIONS: dict[str, str] = {}

for _name, (_smk, _cat, _rel, _cond) in _RAW_RULES.items():
    try:
        _rxn = AllChem.ReactionFromSmarts(_smk)
        _rxn.Initialize()
    except Exception as exc:  # SMIRKS invalide -> signalée, PAS silencieuse
        logger.warning("Règle IGNORÉE (SMIRKS invalide) : %r -> %s", _name, exc)
        continue
    RULES[_name] = _rxn
    RULE_RELIABILITY[_name] = _rel
    RULE_CATEGORY[_name] = _cat
    RULE_CONDITIONS[_name] = _cond

logger.info("Règles compilées : %d / %d", len(RULES), len(_RAW_RULES))

# ---------------------------------------------------------------------------
# Stock / briques de base
# ---------------------------------------------------------------------------
DEFAULT_MAX_HEAVY = 6
STOCK: set[str] = set()


def canonical(smiles: str) -> str | None:
    mol = Chem.MolFromSmiles(smiles)
    return Chem.MolToSmiles(mol) if mol is not None else None


def load_stock(path: str) -> None:
    n = 0
    with open(path, encoding="utf-8") as fh:
        for line in fh:
            token = line.split()
            if not token:
                continue
            can = canonical(token[0])
            if can:
                STOCK.add(can)
                n += 1
    logger.info("Stock chargé depuis %s : %d molécules", path, n)


_stock_file = os.environ.get("RETRO_STOCK_FILE")
if _stock_file and os.path.exists(_stock_file):
    load_stock(_stock_file)

# Catalogue intégré de briques de base commerciales courantes.
# Canonicalisées au démarrage pour correspondre aux SMILES RDKit.
# Pour la prod : pointer RETRO_STOCK_FILE vers un vrai catalogue (eMolecules/ZINC),
# il s'ajoute à celui-ci.
_DEFAULT_STOCK = [
    "O", "CO", "CCO", "CCCO", "CC(C)O", "OCCO",              # eau, alcools usuels, glycol
    "CC(=O)O", "CC(=O)OC(C)=O", "CC(=O)Cl",                  # ac. acétique, anhydride/chlorure d'acétyle
    "C=O", "CC=O", "CC(C)=O", "CCC=O",                       # formaldéhyde, acétaldéhyde, acétone, propanal
    "N", "CN", "CNC", "CCN", "OCCN",                         # ammoniac, méthyl/diméthyl/éthylamine, éthanolamine
    "c1ccccc1", "Cc1ccccc1", "Oc1ccccc1", "Nc1ccccc1",      # benzène, toluène, phénol, aniline
    "Clc1ccccc1", "Brc1ccccc1",                             # chloro/bromobenzène
    "O=C(O)c1ccccc1", "O=C(O)c1ccccc1O", "O=Cc1ccccc1",     # ac. benzoïque, ac. salicylique, benzaldéhyde
    "OCc1ccccc1", "O=C(Cl)c1ccccc1",                        # alcool benzylique, chlorure de benzoyle
    "OC(=O)C(=O)O", "OC(=O)CC(=O)O", "OC(=O)CCC(=O)O",      # ac. oxalique, malonique, succinique
    "Cl", "Br", "I", "OS(=O)(=O)O", "O=[N+]([O-])O",        # HCl, HBr, HI, ac. sulfurique, ac. nitrique
    "ClCCl", "ClC(Cl)Cl", "CCOCC", "CCOC(C)=O",             # DCM, chloroforme, éther, acétate d'éthyle
    # Précurseurs aromatiques disubstitués commerciaux (>6 atomes lourds, donc
    # ajoutés explicitement sinon non reconnus comme briques de base).
    "Nc1ccc(O)cc1", "Nc1ccccc1O",                           # 4- et 2-aminophénol
    "Nc1ccc(C(=O)O)cc1", "Nc1ccccc1C(=O)O",                 # ac. 4-aminobenzoïque (PABA) et anthranilique
    "Oc1ccccc1C(=O)O",                                      # ac. salicylique (forme explicite)
    "O=C(O)c1ccc(O)cc1",                                    # ac. 4-hydroxybenzoïque
    "O=C(O)c1ccc(N)cc1",                                    # ac. 4-aminobenzoïque (forme explicite, doublon volontaire)
    "O=C(O)c1ccc(C)cc1", "O=C(O)c1ccc(Cl)cc1",              # ac. 4-méthyl/4-chlorobenzoïque
    "CCOc1ccccc1", "COc1ccccc1", "Nc1ccc(N)cc1",            # phénétole, anisole, p-phénylènediamine
    "C[N+](=O)[O-]", "CC[N+](=O)[O-]",                      # nitrométhane, nitroéthane (briques de Henry)
    "O=Cc1ccccc1", "O=CCc1ccccc1",                          # benzaldéhyde, phénylacétaldéhyde
]
for _s in _DEFAULT_STOCK:
    _c = canonical(_s)
    if _c:
        STOCK.add(_c)
logger.info("Catalogue de briques de base : %d molécules", len(STOCK))


# ---------------------------------------------------------------------------
# Extension PubChem du stock.
#
# Le catalogue local (_DEFAULT_STOCK) reste la voie rapide, sans réseau, pour
# les réactifs usuels. Pour une molécule plus grosse qui n'y figure pas, on
# interroge PubChem en complément, en DEUX temps :
#   1) compound/smiles/<smiles>/cids/JSON -> CID si la structure est connue.
#   2) pug_view/categories/compound/<cid>/JSON -> liste des dépôts (SID) par
#      catégorie de source. On exige explicitement qu'AU MOINS UNE source
#      soit catégorisée "Chemical Vendors".
# Important : "a un CID PubChem" =/= "achetable". PubChem référence ~110
# millions de structures (littérature, calculs, brevets...), la plupart
# jamais vendues. Sans ce filtre vendeur, presque toute molécule organique
# raisonnable matcherait, ce qui viderait le concept de "stock" de son sens
# (le moteur s'arrêterait n'importe où au lieu de pousser jusqu'à de vraies
# briques de départ). D'où l'appel en 2 étapes, pas juste un test d'existence.
#
# Résilience réseau OBLIGATOIRE : timeout court, throttle ~5 req/s (politique
# d'usage NCBI), jamais d'exception qui remonte à l'appelant. Une panne ou
# indisponibilité de PubChem ne doit JAMAIS faire planter ni geler une
# recherche -> repli silencieux sur l'heuristique locale (taille de molécule).
# ---------------------------------------------------------------------------
PUBCHEM_PUG = "https://pubchem.ncbi.nlm.nih.gov/rest/pug"
PUBCHEM_VIEW = "https://pubchem.ncbi.nlm.nih.gov/rest/pug_view"
PUBCHEM_TIMEOUT = 4.0             # secondes par requête HTTP individuelle
PUBCHEM_MIN_INTERVAL = 0.21       # ~5 req/s, throttle recommandé par NCBI
ENABLE_PUBCHEM_STOCK = os.environ.get("RETRO_PUBCHEM_STOCK", "1") != "0"
PUBCHEM_BUDGET_PER_REQUEST = int(os.environ.get("RETRO_PUBCHEM_BUDGET", "30"))

_pubchem_last_call = 0.0
_pubchem_budget = PUBCHEM_BUDGET_PER_REQUEST  # remis à zéro au début de _solve()


def _pubchem_throttle() -> None:
    global _pubchem_last_call
    wait = PUBCHEM_MIN_INTERVAL - (time.monotonic() - _pubchem_last_call)
    if wait > 0:
        time.sleep(wait)
    _pubchem_last_call = time.monotonic()


def _pubchem_get_json(url: str) -> dict | None:
    """None = échec réseau/HTTP/JSON quelconque -> jamais d'exception
    propagée. Le code appelant doit alors retomber sur l'heuristique locale."""
    try:
        _pubchem_throttle()
        req = urllib.request.Request(url, headers={"User-Agent": "SynthBench-retro/1.0"})
        with urllib.request.urlopen(req, timeout=PUBCHEM_TIMEOUT) as resp:
            return json.loads(resp.read().decode("utf-8"))
    except Exception as exc:  # network, timeout, HTTP 4xx/5xx, JSON malformé...
        logger.info("PubChem indisponible (%s) : %s", url, exc)
        return None


@lru_cache(maxsize=4096)
def pubchem_is_vendor_listed(can_smiles: str) -> bool | None:
    """True  : au moins un fournisseur commercial référencé sur PubChem.
    False : structure résolue par PubChem mais SANS source "Chemical Vendors".
    None  : indéterminé (réseau coupé, budget épuisé, fonctionnalité
            désactivée) -> NE PAS interpréter comme False, juste "on ne sait
            pas", l'appelant doit retomber sur l'heuristique locale."""
    global _pubchem_budget
    if not ENABLE_PUBCHEM_STOCK or _pubchem_budget <= 0:
        return None
    _pubchem_budget -= 1

    smi_q = urllib.parse.quote(can_smiles, safe="")
    cid_data = _pubchem_get_json(f"{PUBCHEM_PUG}/compound/smiles/{smi_q}/cids/JSON")
    if cid_data is None:
        return None
    cids = cid_data.get("IdentifierList", {}).get("CID", [])
    if not cids:
        return False  # structure inconnue de PubChem

    cat_data = _pubchem_get_json(f"{PUBCHEM_VIEW}/categories/compound/{cids[0]}/JSON")
    if cat_data is None:
        return None
    for cat in cat_data.get("SourceCategories", {}).get("Categories", []):
        for source in cat.get("Sources", []):
            if "Chemical Vendors" in source.get("SourceCategories", []):
                return True
    return False


def _reset_pubchem_budget() -> None:
    """Appelé en tête de chaque résolution (_solve) : limite le nombre
    d'appels PubChem par requête /retro ou /selftest, pour borner la latence
    pire-cas même sur une cible complexe à beaucoup de précurseurs distincts."""
    global _pubchem_budget
    _pubchem_budget = PUBCHEM_BUDGET_PER_REQUEST


def is_building_block(can_smiles: str) -> tuple[bool, str | None]:
    """Retourne (in_stock, source). source vaut "catalogue local" ou
    "PubChem (fournisseur commercial)" quand identifié explicitement ; None
    sinon (molécule triviale par taille, ou non résolue en stock du tout)."""
    if can_smiles in STOCK:
        return True, "catalogue local"
    mol = Chem.MolFromSmiles(can_smiles)
    if mol is None:
        return True, None  # SMILES non interprétable : feuille, pas la peine d'insister
    if mol.GetNumHeavyAtoms() <= DEFAULT_MAX_HEAVY:
        return True, None
    if pubchem_is_vendor_listed(can_smiles):
        return True, "PubChem (fournisseur commercial)"
    return False, None


# ---------------------------------------------------------------------------
# Une étape (mémoïsée) : applique chaque règle, ne garde que les fragments
# chimiquement valides, déduplique les doublons de symétrie.
# ---------------------------------------------------------------------------
@lru_cache(maxsize=8192)
def one_step(can_smiles: str) -> tuple[tuple[str, tuple[str, ...]], ...]:
    mol = Chem.MolFromSmiles(can_smiles)
    if mol is None:
        return ()

    seen: set[tuple[str, frozenset[str]]] = set()
    out: list[tuple[str, tuple[str, ...]]] = []

    for name, rxn in RULES.items():
        for product_set in rxn.RunReactants((mol,)):
            precursors: list[str] = []
            valid = True
            for frag in product_set:
                try:
                    Chem.SanitizeMol(frag)  # rejet LÉGITIME des valences impossibles
                except Chem.rdchem.MolSanitizeException:
                    valid = False
                    break
                # IMPORTANT : round-trip de canonicalisation IDENTIQUE à celui du
                # STOCK (canonical() = MolFromSmiles∘MolToSmiles). Sans ça, un H
                # explicite issu d'un template (ex. anhydride "[O:3][H]" / "[N:3][H]")
                # sortait en "[H]O..." et ne matchait JAMAIS le stock canonique,
                # pénalisant à tort la voie anhydride (pourtant la meilleure).
                can_frag = canonical(Chem.MolToSmiles(frag))
                if can_frag is None:
                    valid = False
                    break
                precursors.append(can_frag)

            if not valid or not precursors:
                continue
            if any(p == can_smiles for p in precursors):
                continue
            key = (name, frozenset(precursors))
            if key in seen:
                continue
            seen.add(key)
            out.append((name, tuple(precursors)))

    return tuple(out)


# ---------------------------------------------------------------------------
# Arbre de synthèse
# ---------------------------------------------------------------------------
class Node(BaseModel):
    smiles: str
    in_stock: bool
    stock_source: str | None = None  # "catalogue local" / "PubChem (fournisseur commercial)" / None
    reaction: str | None = None
    category: str | None = None
    conditions: str | None = None
    children: list["Node"] = []
    score: float | None = None  # rempli sur le noeud racine de chaque route
    svg: str | None = None      # dépiction RDKit (structure dessinée) du noeud


Node.model_rebuild()


@lru_cache(maxsize=4096)
def mol_svg(smiles: str) -> str | None:
    """Dépiction SVG d'une molécule (rendu par RDKit, le moteur qui fait la chimie).
    SVG rendu responsive : on retire width/height fixes, on garde le viewBox."""
    mol = Chem.MolFromSmiles(smiles)
    if mol is None:
        return None
    drawer = rdMolDraw2D.MolDraw2DSVG(220, 160)
    drawer.drawOptions().padding = 0.08
    try:
        rdMolDraw2D.PrepareAndDrawMolecule(drawer, mol)
    except Exception:
        return None
    drawer.FinishDrawing()
    svg = drawer.GetDrawingText()
    i = svg.find("<svg")
    if i > 0:
        svg = svg[i:]
    svg = re.sub(r"(<svg[^>]*?)\s+width=(['\"])[^'\"]*\2", r"\1", svg, count=1)
    svg = re.sub(r"(<svg[^>]*?)\s+height=(['\"])[^'\"]*\2", r"\1", svg, count=1)
    return svg


def attach_svg(node: "Node") -> None:
    node.svg = mol_svg(node.smiles)
    for child in node.children:
        attach_svg(child)


def _cartesian(lists: list[list[Node]], cap: int) -> list[tuple[Node, ...]]:
    out: list[tuple[Node, ...]] = []
    for combo in iproduct(*lists):
        out.append(combo)
        if len(out) >= cap:
            break
    return out


def build_routes(can_smiles: str, max_depth: int, beam: int,
                 ancestors: frozenset[str] = frozenset(),
                 is_root: bool = True) -> list[Node]:
    # La CIBLE (racine) ne doit jamais être court-circuitée comme brique de
    # base, même si elle est achetable (l'aspirine EST vendue, mais on veut
    # quand même montrer sa rétrosynthèse). Le test stock ne s'applique donc
    # qu'aux précurseurs (is_root=False). Sans ce garde, toute cible listée
    # chez un fournisseur PubChem renvoyait "déjà une brique de base" et
    # aucune voie -> exactement le bug observé sur aspirine/paracétamol/etc.
    if not is_root:
        in_stock, stock_source = is_building_block(can_smiles)
        if in_stock:
            return [Node(smiles=can_smiles, in_stock=True, stock_source=stock_source)]
    if max_depth <= 0 or can_smiles in ancestors:
        return [Node(smiles=can_smiles, in_stock=False)]

    new_anc = ancestors | {can_smiles}
    routes: list[Node] = []

    # Tri par fiabilité décroissante AVANT troncature : sans ça, le slice
    # [:beam] gardait les règles dans l'ordre du dict, pas les plus probables,
    # et pouvait faire disparaître une voie réaliste à fiabilité élevée.
    candidates = sorted(one_step(can_smiles),
                        key=lambda nc: -RULE_RELIABILITY.get(nc[0], 0.5))

    for name, precursors in candidates[:beam]:
        per_precursor = [build_routes(p, max_depth - 1, beam, new_anc, is_root=False)
                         for p in precursors]
        for combo in _cartesian(per_precursor, cap=beam):
            routes.append(Node(smiles=can_smiles, in_stock=False,
                               reaction=name, category=RULE_CATEGORY.get(name),
                               conditions=RULE_CONDITIONS.get(name),
                               children=list(combo)))
    # On génère PLUS large que beam (jusqu'à un plafond), sans couper sur la
    # simple position : une voie réaliste mais issue d'une règle tardive (ex.
    # condensation de Henry) ne doit pas être éliminée juste parce que des
    # règles plus prioritaires ont déjà rempli le quota. On assemble un vivier,
    # puis on garde les `beam` MEILLEURES au score.
    gen_cap = max(beam * 3, 12)  # vivier plus large que le beam final
    for name, precursors in candidates:
        per_precursor = [build_routes(p, max_depth - 1, beam, new_anc, is_root=False)
                         for p in precursors]
        for combo in _cartesian(per_precursor, cap=beam):
            routes.append(Node(smiles=can_smiles, in_stock=False,
                               reaction=name, category=RULE_CATEGORY.get(name),
                               conditions=RULE_CONDITIONS.get(name),
                               children=list(combo)))
            if len(routes) >= gen_cap:
                break
        if len(routes) >= gen_cap:
            break

    if routes:
        # Garder les beam meilleures au score (résolu + en stock + court), pas
        # les premières générées. C'est ce qui fait remonter les voies qui
        # aboutissent réellement à des briques, où qu'elles soient dans l'ordre.
        routes.sort(key=score_route, reverse=True)
        return routes[:beam]
    # Aucune décomposition trouvée pour la cible. Si elle est elle-même une
    # brique de base (petite molécule, ou achetable), on l'étiquette comme
    # telle pour qu'elle reste "résolue" ; sinon feuille non résolue.
    in_stock, stock_source = is_building_block(can_smiles)
    return [Node(smiles=can_smiles, in_stock=in_stock, stock_source=stock_source)]


def is_solved(node: Node) -> bool:
    if not node.children:
        return node.in_stock
    return all(is_solved(c) for c in node.children)


def route_depth(node: Node) -> int:
    if not node.children:
        return 0
    return 1 + max(route_depth(c) for c in node.children)


def route_signature(node: Node):
    return (node.smiles, node.reaction,
            tuple(route_signature(c) for c in node.children))


def count_leaves_in_stock(node: Node) -> tuple[int, int]:
    """(feuilles en stock, feuilles totales) — mesure à quel point une route
    aboutit réellement à des briques disponibles."""
    if not node.children:
        return (1 if node.in_stock else 0, 1)
    si = ti = 0
    for c in node.children:
        s, t = count_leaves_in_stock(c)
        si += s; ti += t
    return si, ti


def score_route(node: Node) -> float:
    rel = 1.0
    stack = [node]
    while stack:
        n = stack.pop()
        if n.reaction:
            rel *= RULE_RELIABILITY.get(n.reaction, 0.5)
        stack.extend(n.children)
    solved = is_solved(node)
    # Une route entièrement résolue vaut BEAUCOUP plus qu'une route partielle :
    # c'est le critère n°1 (on veut des voies qui aboutissent à des briques
    # réelles). Une route partielle est fortement pénalisée.
    solved_factor = 1.0 if solved else 0.25
    # Bonus supplémentaire proportionnel à la fraction de feuilles réellement
    # en stock : départage deux routes résolues en faveur de celle dont les
    # précurseurs sont plus concrètement disponibles.
    si, ti = count_leaves_in_stock(node)
    stock_frac = si / ti if ti else 0.0
    stock_bonus = 0.7 + 0.3 * stock_frac
    # Pénalité de longueur douce (les voies courtes sont préférées, mais une
    # voie longue entièrement résolue reste meilleure qu'une courte en impasse).
    length_factor = 0.92 ** route_depth(node)
    return round(max(0.0, min(1.0, rel * solved_factor * stock_bonus * length_factor)), 3)


# Les conditions affichées (champ `conditions` de chaque Node) sont des
# indications théoriques de littérature générale, non validées
# expérimentalement par ce backend (RDKit fait la chimie structurale, pas la
# faisabilité réactionnelle réelle). Exposé dans CHAQUE réponse API, pas
# seulement en commentaire, pour que le frontend puisse l'afficher à l'utilisateur.
CONDITIONS_DISCLAIMER = (
    "Conditions à titre indicatif (littérature générale) — non vérifiées "
    "expérimentalement. À valider avant toute mise en œuvre réelle."
)

STOCK_DISCLAIMER = (
    "Statut \"en stock\" = référencé soit dans le catalogue local, soit chez "
    "au moins un fournisseur commercial sur PubChem à un instant donné — "
    "PAS une garantie de disponibilité, prix ou pureté actuels."
)


# ---------------------------------------------------------------------------
# API
# ---------------------------------------------------------------------------
class RetroRequest(BaseModel):
    smiles: str = Field(..., min_length=1, examples=["CC(=O)Oc1ccccc1C(=O)O"])
    max_depth: int = Field(3, ge=1, le=6)
    beam: int = Field(8, ge=1, le=20)


class RetroResponse(BaseModel):
    target: str
    solved: bool
    routes: list[Node]
    disclaimer: str = CONDITIONS_DISCLAIMER
    stock_disclaimer: str = STOCK_DISCLAIMER


def _solve(can: str, max_depth: int, beam: int) -> list[Node]:
    """Coeur commun : génère, déduplique, score et trie les routes pour une
    cible déjà canonicalisée. Partagé par /retro et /selftest pour qu'ils
    testent EXACTEMENT le même chemin de code (pas de divergence silencieuse)."""
    _reset_pubchem_budget()  # budget d'appels PubChem borné PAR requête
    raw = build_routes(can, max_depth, beam)
    unique: dict[tuple, Node] = {}
    for r in raw:
        unique.setdefault(route_signature(r), r)
    routes = list(unique.values())
    for r in routes:
        r.score = score_route(r)
    routes.sort(key=lambda r: (-(r.score or 0.0), route_depth(r)))
    return routes[:beam]


@app.post("/retro", response_model=RetroResponse)
def analyze_retro(req: RetroRequest) -> RetroResponse:
    can = canonical(req.smiles)
    if can is None:
        raise HTTPException(status_code=422, detail="SMILES invalide")

    final = _solve(can, req.max_depth, req.beam)
    logger.info("Cible %s : %d route(s), résolu=%s",
                can, len(final), any(is_solved(r) for r in final))

    for r in final:
        attach_svg(r)  # dessine chaque molécule (structure) avant l'envoi
    return RetroResponse(target=can, solved=any(is_solved(r) for r in final),
                         routes=final)


# Cas de référence : permet de VÉRIFIER en un appel que les voies non exotiques
# attendues sortent bien (l'aspirine est l'exemple de référence du projet).
_SELFTEST_CASES = [
    ("Aspirine", "CC(=O)Oc1ccccc1C(=O)O"),
    ("Acétate d'éthyle", "CCOC(C)=O"),
    ("Acétanilide", "CC(=O)Nc1ccccc1"),
    ("Paracétamol", "CC(=O)Nc1ccc(O)cc1"),
    ("Benzocaïne", "CCOC(=O)c1ccc(N)cc1"),
    ("Benzoate de méthyle", "COC(=O)c1ccccc1"),
]


@app.get("/selftest")
def selftest(max_depth: int = 3, beam: int = 8) -> dict[str, object]:
    """Vérification empirique des voies trouvées sur des molécules connues.
    À visiter après déploiement : /selftest — confirme combien de voies non
    exotiques sortent pour chaque cas et lesquelles, sans dépiction SVG."""
    cases: list[dict[str, object]] = []
    for label, smi in _SELFTEST_CASES:
        can = canonical(smi)
        if can is None:
            cases.append({"label": label, "smiles": smi, "error": "SMILES invalide"})
            continue
        routes = _solve(can, max_depth, beam)
        cases.append({
            "label": label,
            "smiles": can,
            "solved": any(is_solved(r) for r in routes),
            "n_routes": len(routes),
            "routes": [
                {
                    "reaction": r.reaction,
                    "score": r.score,
                    "conditions": r.conditions,
                    "precursors": [c.smiles for c in r.children],
                    "all_in_stock": all(c.in_stock for c in r.children),
                }
                for r in routes if r.reaction
            ],
        })
    return {"rules_loaded": len(RULES), "stock_size": len(STOCK), "cases": cases}


@app.get("/health")
def health() -> dict[str, object]:
    return {"status": "ok", "rules_loaded": len(RULES),
            "rules_total": len(_RAW_RULES), "stock_size": len(STOCK),
            "pubchem_stock_enabled": ENABLE_PUBCHEM_STOCK,
            "pubchem_budget_per_request": PUBCHEM_BUDGET_PER_REQUEST}
