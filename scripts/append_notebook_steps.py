"""Append the production handoff to the existing notebook.

The first 19 cells are copied without modification. This is intentionally a
small, auditable transformation because the research notebook is user-owned.
"""

from __future__ import annotations

import json
from pathlib import Path

SOURCE = Path("/Users/root1/Downloads/Financial_Analysis.ipynb")
TARGET = Path(__file__).resolve().parents[1] / "notebooks" / "Financial_Analysis.ipynb"


def markdown(text: str) -> dict:
    return {"cell_type": "markdown", "metadata": {}, "source": text.splitlines(keepends=True)}


def code(text: str) -> dict:
    return {
        "cell_type": "code",
        "execution_count": None,
        "metadata": {},
        "outputs": [],
        "source": text.splitlines(keepends=True),
    }


def main() -> None:
    original = json.loads(SOURCE.read_text())
    if len(original.get("cells", [])) < 19:
        raise ValueError("The source notebook does not contain Steps 0-18.")
    cells = list(original["cells"][:19])

    cells.extend([
        markdown("""## Step 19 — Export the production artifact\n\nThis is the only notebook step that packages the outputs of Steps 1–18. It does not retrain or alter the research pipeline. It adds an optional metadata snapshot so the product can display sector, industry, market-cap, and extra fundamental context.\n"""),
        code("""from pathlib import Path\nimport json\nimport pickle\n\n# Prefer Drive so the artifact survives a Colab runtime reset.\nARTIFACT_DIR = Path(\"/content/drive/MyDrive/AI_Portfolio_Advisor\") if Path(\"/content/drive/MyDrive\").exists() else Path(\"/content/ai_portfolio_advisor\")\nARTIFACT_DIR.mkdir(parents=True, exist_ok=True)\n\nMETA_KEYS = [\n    \"sector\", \"industry\", \"marketCap\", \"forwardPE\", \"trailingEps\",\n    \"earningsGrowth\", \"grossMargins\", \"operatingMargins\", \"currentRatio\",\n    \"quickRatio\", \"beta\", \"recommendationKey\",\n]\n\ndef enriched_metadata(tickers):\n    rows = {}\n    for ticker in tickers:\n        try:\n            info = yf.Ticker(ticker).info\n            rows[ticker] = {\n                key: info.get(key, np.nan)\n                for key in META_KEYS\n            }\n        except Exception as exc:\n            print(f\"metadata unavailable for {ticker}: {exc}\")\n            rows[ticker] = {key: np.nan for key in META_KEYS}\n    meta = pd.DataFrame(rows).T\n    meta = meta.rename(columns={\"marketCap\": \"market_cap\"})\n    return meta\n\nmetadata = enriched_metadata(sorted(prices))\nfor col in metadata.columns:\n    if col not in (\"sector\", \"industry\", \"recommendationKey\"):\n        metadata[col] = pd.to_numeric(metadata[col], errors=\"coerce\")\n\n# Keep the original seven ratios used by the model and add extra display-only fields.\nfunda_export = funda.copy()\nfor col in [\"forwardPE\", \"trailingEps\", \"earningsGrowth\", \"grossMargins\",\n            \"operatingMargins\", \"currentRatio\", \"quickRatio\", \"beta\"]:\n    if col in metadata:\n        funda_export[\"f_\" + col] = metadata[col]\nfunda_export[\"f_market_cap\"] = metadata.get(\"market_cap\", np.nan)\n\nbundle = dict(\n    res=res, prices=prices, models=models, panel_test=test,\n    features=TOPK, cfg=CFG, funda=funda_export, metadata=metadata,\n    exporter_version=\"2026-08-27-production-v1\",\n)\nwith open(ARTIFACT_DIR / \"artifacts.pkl\", \"wb\") as handle:\n    pickle.dump(bundle, handle, protocol=pickle.HIGHEST_PROTOCOL)\nmetadata.to_csv(ARTIFACT_DIR / \"metadata.csv\")\nmanifest = {\n    \"exporter_version\": bundle[\"exporter_version\"],\n    \"assets\": sorted(prices),\n    \"selected_features\": TOPK,\n    \"test_start\": str(test.index.min()),\n    \"test_end\": str(test.index.max()),\n    \"fundamental_point_in_time\": False,\n    \"notes\": [\n        \"Fundamentals come from the current yfinance snapshot and are not historical point-in-time data.\",\n        \"The artifact is for research and educational use; it does not place trades.\",\n    ],\n}\n(ARTIFACT_DIR / \"manifest.json\").write_text(json.dumps(manifest, indent=2, default=str))\nprint(f\"Saved production artifact to {ARTIFACT_DIR / 'artifacts.pkl'}\")\nprint(f\"Saved metadata to {ARTIFACT_DIR / 'metadata.csv'}\")\nprint(f\"Assets: {len(prices)} | held-out rows: {len(res):,} | selected features: {len(TOPK)}\")\n"""),
        markdown("""## Step 20 — Validate the handoff\n\nThis checks that the artifact has the fields required by the standalone Python product. It intentionally does not claim the model is profitable; it checks packaging and coverage only.\n"""),
        code("""with open(ARTIFACT_DIR / \"artifacts.pkl\", \"rb\") as handle:\n    production_bundle = pickle.load(handle)\n\nrequired = {\"res\", \"prices\", \"models\", \"panel_test\", \"features\", \"funda\", \"metadata\"}\nmissing = required.difference(production_bundle)\nassert not missing, f\"Missing artifact keys: {sorted(missing)}\"\nassert len(production_bundle[\"res\"]) > 0\nassert len(production_bundle[\"features\"]) > 0\nprint(\"Artifact validation passed.\")\nprint(\"Copy artifacts.pkl, metadata.csv, and manifest.json into the app project's data/ directory.\")\n"""),
        markdown("""## Step 21 — Run the product\n\nThe production frontend and API live in the companion repository. In Colab, copy the exported `artifacts.pkl` into that repository's `data/` directory. For a persistent service, deploy the repository with its Dockerfile or Docker Compose configuration in Coolify.\n\nThe frontend is Python-only Streamlit. The optional API is Python-only FastAPI. No Django or JavaScript application is required.\n"""),
        code("""print(\"Local UI: streamlit run streamlit_app.py\")\nprint(\"Local API: uvicorn api_app:app --host 0.0.0.0 --port 8000\")\nprint(\"Docker UI: docker build -t ai-portfolio-advisor . && docker run -p 8501:8501 -v $PWD/data:/app/data:ro ai-portfolio-advisor\")\nprint(\"Docker Compose: docker compose up --build\")\n"""),
    ])

    output = dict(original)
    output["cells"] = cells
    TARGET.write_text(json.dumps(output, indent=1) + "\n")
    print(f"Wrote {TARGET} with {len(cells)} cells")


if __name__ == "__main__":
    main()
