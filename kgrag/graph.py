import json
import re

import networkx as nx
from openai import OpenAI

from kgrag.config import CHAT_MODEL, EMBED_MODEL, GRAPH_PATH

# scraped field -> (entity type, relation)
FIELD_RELATIONS = {
    "Concerned Department": ("Department", "OFFERED_BY"),
    "Sponsered By": ("Sponsor", "SPONSORED_BY"),
    "Beneficiaries": ("Beneficiary", "BENEFITS"),
    "Types of Benefits": ("BenefitType", "PROVIDES"),
}

# LLM-extracted key -> (entity type, relation)
EXTRACTED_RELATIONS = {
    "crops": ("Crop", "COVERS_CROP"),
    "target_groups": ("TargetGroup", "PRIORITISES"),
    "districts": ("District", "AVAILABLE_IN"),
    "offices": ("Office", "APPLY_AT"),
}

EXTRACT_PROMPT = """Extract entities from this Tamil Nadu government scheme. Return JSON with these keys,
each a list of short canonical names in Title Case (empty list if none):
- crops: crops or farm commodities covered (e.g. "Groundnut", "Paddy", "Maize")
- target_groups: specific eligible or prioritised groups (e.g. "Women Farmers", "SC/ST Farmers", "Small/Marginal Farmers", "Micro Enterprises")
- districts: Tamil Nadu districts named
- offices: officials or offices where people apply (e.g. "Assistant Agricultural Officer", "Joint Director of Agriculture")

Scheme:
{text}"""

client = OpenAI()


def embed(texts: list[str]) -> list[list[float]]:
    resp = client.embeddings.create(model=EMBED_MODEL, input=texts)
    return [d.embedding for d in resp.data]


def scheme_text(s: dict) -> str:
    return "\n".join(f"{k}: {v}" for k, v in s.items() if k != "url")


def extract_entities(s: dict) -> dict:
    resp = client.chat.completions.create(
        model=CHAT_MODEL,
        temperature=0,
        response_format={"type": "json_object"},
        messages=[{"role": "user", "content": EXTRACT_PROMPT.format(text=scheme_text(s))}],
    )
    return json.loads(resp.choices[0].message.content)


def _link(g: nx.MultiDiGraph, scheme_id: str, etype: str, rel: str, name: str):
    name = re.sub(r"\s+", " ", name).strip()
    if not name:
        return
    node_id = f"{etype}:{name.lower()}"
    if node_id not in g:
        g.add_node(node_id, type=etype, name=name)
    g.add_edge(scheme_id, node_id, relation=rel)


def build_graph(schemes: list[dict]) -> nx.MultiDiGraph:
    g = nx.MultiDiGraph()
    embeddings = embed([scheme_text(s) for s in schemes])
    for s, emb in zip(schemes, embeddings):
        sid = "Scheme:" + s["url"].split("id=")[-1]
        g.add_node(
            sid,
            type="Scheme",
            name=s["Scheme Title/Name"],
            url=s["url"],
            description=s.get("Description", ""),
            funding=s.get("Funding Pattern", ""),
            how_to_avail=s.get("How To avail", ""),
            embedding=emb,
        )
        for field, (etype, rel) in FIELD_RELATIONS.items():
            if s.get(field):
                _link(g, sid, etype, rel, s[field])
        for key, names in extract_entities(s).items():
            if key in EXTRACTED_RELATIONS:
                for name in names:
                    _link(g, sid, *EXTRACTED_RELATIONS[key], name)
    return g


def save_graph(g: nx.MultiDiGraph, path=GRAPH_PATH):
    path.write_text(json.dumps(nx.node_link_data(g, edges="edges"), ensure_ascii=False), encoding="utf-8")


def load_graph(path=GRAPH_PATH) -> nx.MultiDiGraph:
    return nx.node_link_graph(json.loads(path.read_text(encoding="utf-8")), edges="edges")
