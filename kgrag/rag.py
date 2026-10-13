import re

from kgrag.config import CHAT_MODEL
from kgrag.graph import client, embed, load_graph

SYSTEM_PROMPT = """You are an assistant for Tamil Nadu government schemes for farmers and MSMEs (manufacturers).
Answer only from the knowledge graph context below. If the answer is not in the context, say you don't know.
Answer only what is asked, citing scheme names and amounts. Be concise.

Context:
{context}"""


class GraphRAG:
    def __init__(self):
        self.g = load_graph()
        self.schemes = [n for n, d in self.g.nodes(data=True) if d["type"] == "Scheme"]

    def matched_entities(self, question: str) -> list[str]:
        """Graph entities whose name appears in the question."""
        q = question.lower()
        return [
            n for n, d in self.g.nodes(data=True)
            if d["type"] != "Scheme" and re.search(rf"\b{re.escape(d['name'].lower())}s?\b", q)
        ]

    def retrieve(self, question: str, k: int = 4, max_schemes: int = 8) -> tuple[list[str], list[str]]:
        q = embed([question])[0]
        score = {s: sum(a * b for a, b in zip(q, self.g.nodes[s]["embedding"])) for s in self.schemes}
        entities = self.matched_entities(question)
        linked = {s for e in entities for s in self.g.predecessors(e)}
        top = sorted(self.schemes, key=score.get, reverse=True)
        hits = top[:k] + [s for s in top[k:] if s in linked]
        return hits[:max_schemes], entities

    def scheme_context(self, sid: str) -> str:
        d = self.g.nodes[sid]
        lines = [f"Scheme: {d['name']}", f"URL: {d['url']}"]
        for _, target, rel in self.g.out_edges(sid, data="relation"):
            lines.append(f"{rel}: {self.g.nodes[target]['name']}")
        lines += [f"Funding: {d['funding']}", f"How to avail: {d['how_to_avail']}", f"Description: {d['description']}"]
        return "\n".join(lines)

    def entity_context(self, eid: str) -> str:
        d = self.g.nodes[eid]
        names = sorted({self.g.nodes[s]["name"] for s in self.g.predecessors(eid)})
        return f"{d['type']} '{d['name']}' is linked to {len(names)} schemes: {'; '.join(names)}"

    def answer(self, question: str, history: list[dict] | None = None) -> dict:
        history = history or []
        # include the previous user turn so follow-ups ("how do I apply?") retrieve the right scheme
        prev = next((m["content"] for m in reversed(history) if m["role"] == "user"), "")
        hits, entities = self.retrieve(f"{prev}\n{question}".strip())
        contexts = [self.entity_context(e) for e in entities] + [self.scheme_context(s) for s in hits]
        messages = [{"role": "system", "content": SYSTEM_PROMPT.format(context="\n\n".join(contexts))}]
        messages += history[-6:] + [{"role": "user", "content": question}]
        resp = client.chat.completions.create(model=CHAT_MODEL, temperature=0, messages=messages)
        return {
            "answer": resp.choices[0].message.content,
            "contexts": contexts,
            "sources": [self.g.nodes[s] for s in hits],
        }
