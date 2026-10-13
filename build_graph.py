"""Scrape the scheme pages and build the knowledge graph (run once, before starting the app)."""
import argparse
import json

from kgrag.config import ROOT
from kgrag.graph import build_graph, save_graph
from kgrag.scraper import scheme_details, scheme_links

SCHEMES_PATH = ROOT / "data" / "schemes.json"


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--refresh", action="store_true", help="re-scrape even if data/schemes.json exists")
    args = parser.parse_args()

    if args.refresh or not SCHEMES_PATH.exists():
        links = scheme_links()
        print(f"Scraping {len(links)} schemes...")
        schemes = [scheme_details(url) for url in links]
        SCHEMES_PATH.write_text(json.dumps(schemes, indent=2, ensure_ascii=False), encoding="utf-8")
    schemes = json.loads(SCHEMES_PATH.read_text(encoding="utf-8"))

    print("Extracting entities and building graph...")
    g = build_graph(schemes)
    save_graph(g)
    print(f"Saved graph: {g.number_of_nodes()} nodes, {g.number_of_edges()} edges")


if __name__ == "__main__":
    main()
