import os
from pathlib import Path

from dotenv import load_dotenv

load_dotenv()

ROOT = Path(__file__).resolve().parent.parent
GRAPH_PATH = ROOT / "data" / "graph.json"
GUARDRAILS_DIR = ROOT / "guardrails"

BASE_URL = "https://www.tn.gov.in/"
SCHEME_LIST_URL = os.getenv("SCHEME_LIST_URL", BASE_URL + "scheme_list.php?dep_id=Mg==")

CHAT_MODEL = os.getenv("CHAT_MODEL", "gpt-4o-mini")
EMBED_MODEL = os.getenv("EMBED_MODEL", "text-embedding-3-small")
EVAL_MODEL = os.getenv("EVAL_MODEL", "gpt-4.1")
