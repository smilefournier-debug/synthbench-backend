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

import logging
import os
from functools import lru_cache
from itertools import product as iproduct

from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field
from rdkit import Chem, RDLogger
from rdkit.Chem import AllChem

RDLogger.DisableLog("rdApp.*")
logging.basicConfig(level=logging.INFO, format="%(levelname)s %(message)s")
logger = logging.getLogger("retro")

app = FastAPI(title="RetroSynthesis API", version="3.0.0")
app.add_middleware(CORSMiddleware, allow_origins=["*"], allow_methods=["*"],
                   allow_headers=["*"])

# ---------------------------------------------------------------------------
# RÈGLES : (SMIRKS, catégorie, fiabilité 0-1).
# Les 4 premières sont conceptuellement vérifiées ; les 4 suivantes sont des
# POINTS DE DÉPART à valider sur tes propres molécules (je n'ai pas pu exécuter
# RDKit ici). Toute SMIRKS qui ne compile pas est signalée au démarrage.
# ---------------------------------------------------------------------------
_RAW_RULES: dict[str, tuple[str, str, float]] = {
    "Clivage d'ester":
        ("[C:1](=[O:2])[O:3][C:4]>>[C:1](=[O:2])[OH].[OH:3][C:4]",
         "Acylation", 0.90),
    "Hydrolyse amide / peptidique":
        ("[C:1](=[O:2])[N;!$([NX3]=*):3][C:4]>>[C:1](=[O:2])[OH].[N:3][C:4]",
         "Acylation", 0.85),
    "Élimination de Hofmann":
        ("[N+:1]([C:2])([C:3])([C:4])[C:5][C:6]>>[N:1]([C:2])([C:3])[C:4].[C:5]=[C:6]",
         "Alcaloïde", 0.60),
    "Rétro-Grignard (alcool sec.)":
        ("[OH:1][C:2]([C:3])[C:4]>>[C:2](=O)[C:4].[C:3][Mg]Br",
         "C–C", 0.70),
    # --- À VALIDER ---
    "Éther de Williamson":
        ("[CX4:1][O:2][CX4:3]>>[CX4:1][OH:2].[CX4:3][Cl]",
         "Substitution", 0.75),
    "Couplage de Suzuki (biaryle)":
        ("[c:1]!@[c:2]>>[c:1][Br].[c:2]B(O)O",
         "Couplage C–C", 0.90),
    "Amination réductrice":
        ("[CX4;!$([CX4]([#7])[#7]):1][NX3;!$([NX3]C=O):2]>>[C:1]=O.[N:2]",
         "C–N", 0.80),
    "Rétro-aldol (β-hydroxy carbonyle)":
        ("[C:1](=[O:2])[CH2:3][CH:4][OX2H:5]>>[C:1](=[O:2])[CH3:3].[CH:4]=[O:5]",
         "C–C", 0.65),
}

RULES: dict[str, AllChem.ChemicalReaction] = {}
RULE_RELIABILITY: dict[str, float] = {}
RULE_CATEGORY: dict[str, str] = {}

for _name, (_smk, _cat, _rel) in _RAW_RULES.items():
    try:
        _rxn = AllChem.ReactionFromSmarts(_smk)
        _rxn.Initialize()
    except Exception as exc:  # SMIRKS invalide -> signalée, PAS silencieuse
        logger.warning("Règle IGNORÉE (SMIRKS invalide) : %r -> %s", _name, exc)
        continue
    RULES[_name] = _rxn
    RULE_RELIABILITY[_name] = _rel
    RULE_CATEGORY[_name] = _cat

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


def is_building_block(can_smiles: str) -> bool:
    if can_smiles in STOCK:
        return True
    mol = Chem.MolFromSmiles(can_smiles)
    if mol is None:
        return True
    return mol.GetNumHeavyAtoms() <= DEFAULT_MAX_HEAVY


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
                precursors.append(Chem.MolToSmiles(frag))

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
    reaction: str | None = None
    category: str | None = None
    children: list["Node"] = []
    score: float | None = None  # rempli sur le noeud racine de chaque route


Node.model_rebuild()


def _cartesian(lists: list[list[Node]], cap: int) -> list[tuple[Node, ...]]:
    out: list[tuple[Node, ...]] = []
    for combo in iproduct(*lists):
        out.append(combo)
        if len(out) >= cap:
            break
    return out


def build_routes(can_smiles: str, max_depth: int, beam: int,
                 ancestors: frozenset[str] = frozenset()) -> list[Node]:
    if is_building_block(can_smiles):
        return [Node(smiles=can_smiles, in_stock=True)]
    if max_depth <= 0 or can_smiles in ancestors:
        return [Node(smiles=can_smiles, in_stock=False)]

    new_anc = ancestors | {can_smiles}
    routes: list[Node] = []

    for name, precursors in one_step(can_smiles)[:beam]:
        per_precursor = [build_routes(p, max_depth - 1, beam, new_anc)
                         for p in precursors]
        for combo in _cartesian(per_precursor, cap=beam):
            routes.append(Node(smiles=can_smiles, in_stock=False,
                               reaction=name, category=RULE_CATEGORY.get(name),
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


# ---------------------------------------------------------------------------
# API
# ---------------------------------------------------------------------------
class RetroRequest(BaseModel):
    smiles: str = Field(..., min_length=1, examples=["CC(=O)Oc1ccccc1C(=O)O"])
    max_depth: int = Field(3, ge=1, le=6)
    beam: int = Field(5, ge=1, le=20)


class RetroResponse(BaseModel):
    target: str
    solved: bool
    routes: list[Node]


@app.post("/retro", response_model=RetroResponse)
def analyze_retro(req: RetroRequest) -> RetroResponse:
    can = canonical(req.smiles)
    if can is None:
        raise HTTPException(status_code=422, detail="SMILES invalide")

    raw = build_routes(can, req.max_depth, req.beam)

    # dédup + scoring
    unique: dict[tuple, Node] = {}
    for r in raw:
        unique.setdefault(route_signature(r), r)
    routes = list(unique.values())
    for r in routes:
        r.score = score_route(r)

    routes.sort(key=lambda r: (-(r.score or 0.0), route_depth(r)))
    logger.info("Cible %s : %d route(s), résolu=%s",
                can, len(routes), any(is_solved(r) for r in routes))

    return RetroResponse(target=can, solved=any(is_solved(r) for r in routes),
                         routes=routes[:req.beam])


@app.get("/health")
def health() -> dict[str, object]:
    return {"status": "ok", "rules_loaded": len(RULES),
            "rules_total": len(_RAW_RULES), "stock_size": len(STOCK)}
