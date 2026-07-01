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
                 ancestors: frozenset[str] = frozenset()) -> list[Node]:
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
        per_precursor = [build_routes(p, max_depth - 1, beam, new_anc)
                         for p in precursors]
        for combo in _cartesian(per_precursor, cap=beam):
            routes.append(Node(smiles=can_smiles, in_stock=False,
                               reaction=name, category=RULE_CATEGORY.get(name),
                               conditions=RULE_CONDITIONS.get(name),
                               children=list(combo)))
            if len(routes) >= beam:
                break
        if len(routes) >= beam:
            break

    return routes or [Node(smiles=can_smiles, in_stock=False)]


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


def score_route(node: Node) -> float:
    rel = 1.0
    stack = [node]
    while stack:
        n = stack.pop()
        if n.reaction:
            rel *= RULE_RELIABILITY.get(n.reaction, 0.5)
        stack.extend(n.children)
    solved_factor = 1.0 if is_solved(node) else 0.3
    length_factor = 0.9 ** route_depth(node)
    return round(max(0.0, min(1.0, rel * solved_factor * length_factor)), 3)


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
