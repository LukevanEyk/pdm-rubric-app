import os
import sqlite3
from typing import Optional, Dict, Tuple
import pandas as pd
import streamlit as st
from datetime import datetime

# =========================
# DB LOCATION (simple)
# =========================
DB_PATH = "rubric.db"  # If you see sync/refresh oddities on OneDrive, move this off OneDrive.

# =========================
# Compatibility helpers
# =========================
def df_show(df: pd.DataFrame, height: Optional[int] = None):
    """Show a dataframe, but tolerate older Streamlit versions that lack certain kwargs."""
    try:
        st.dataframe(df, use_container_width=True, height=height)
    except TypeError:
        try:
            st.dataframe(df, height=height)
        except TypeError:
            st.dataframe(df)

def selectbox_compat(label, options, index=0, key=None):
    """Label-visibility kwarg was added later; avoid it for maximum compatibility."""
    return st.selectbox(label, options=options, index=index, key=key)

def radio_compat(label, options, index=0, key=None, horizontal=True):
    """Use horizontal=True when available; otherwise fall back."""
    try:
        return st.radio(label, options, index=index, key=key, horizontal=horizontal)
    except TypeError:
        return st.radio(label, options, index=index, key=key)

# =========================
# DB helpers
# =========================
def get_conn():
    conn = sqlite3.connect(DB_PATH)
    conn.execute("PRAGMA foreign_keys = ON;")
    # WAL improves concurrent reads; harmless locally
    try:
        conn.execute("PRAGMA journal_mode = WAL;")
        conn.execute("PRAGMA synchronous = NORMAL;")
        conn.execute("PRAGMA wal_autocheckpoint = 1000;")
    except Exception:
        pass
    return conn

def init_db():
    conn = get_conn()
    cur = conn.cursor()

    # Rubric definitions
    cur.execute("""
        CREATE TABLE IF NOT EXISTS rubric (
            id INTEGER PRIMARY KEY,
            dimension TEXT NOT NULL,
            subdimension TEXT NOT NULL,
            score INTEGER NOT NULL,
            explanation TEXT NOT NULL
        );
    """)

    # Studies (metadata trimmed to: title, authors, year, reviewer)
    cur.execute("""
        CREATE TABLE IF NOT EXISTS study (
            id INTEGER PRIMARY KEY,
            title TEXT NOT NULL,
            authors TEXT,
            year INTEGER,
            reviewer TEXT,
            created_at TEXT NOT NULL
        );
    """)

    # Scores (one row per sub-dimension for a study)
    cur.execute("""
        CREATE TABLE IF NOT EXISTS study_score (
            id INTEGER PRIMARY KEY,
            study_id INTEGER NOT NULL,
            dimension TEXT NOT NULL,
            subdimension TEXT NOT NULL,
            score INTEGER NOT NULL,
            reason TEXT,
            FOREIGN KEY(study_id) REFERENCES study(id) ON DELETE CASCADE
        );
    """)

    # Weights (global, one per dimension)
    cur.execute("""
        CREATE TABLE IF NOT EXISTS weights (
            dimension TEXT PRIMARY KEY,
            base_weight REAL NOT NULL,
            reason TEXT,
            updated_at TEXT NOT NULL
        );
    """)

    conn.commit()

    # Seed rubric if empty
    cur.execute("SELECT COUNT(*) FROM rubric;")
    if cur.fetchone()[0] == 0:
        seed_rubric(cur)
        conn.commit()

    # Seed weights if empty
    cur.execute("SELECT COUNT(*) FROM weights;")
    if cur.fetchone()[0] == 0:
        seed_weights(cur)
        conn.commit()

    conn.close()

def seed_rubric(cur):
    # (dimension, subdimension, score, explanation)
    entries = [
        # INTERPRETABILITY
        ("Interpretability", "-", 0, "Black box — Deep/complex models without interpretation; includes hybrids where a data-driven model outputs the prediction."),
        ("Interpretability", "-", 1, "Post-hoc interpretability — SHAP/LIME/saliency explain predictions after the fact; decision process remains opaque."),
        ("Interpretability", "-", 2, "Interpretable throughout process — Built-in transparency (trees/rules/linear/logistic/simple signal rules); not physics-rooted."),
        ("Interpretability", "-", 3, "Physics-focused — Physics or hybrid models where outputs are based on physics; highest interpretability."),

        # MATURITY: Technique
        ("Maturity", "Technique", 1, "Anomaly detection — Detects anomalies in operating conditions."),
        ("Maturity", "Technique", 2, "Fault detection — Differentiates anomaly vs. fault."),
        ("Maturity", "Technique", 3, "Fault diagnosis — Identifies which fault type is present."),
        ("Maturity", "Technique", 4, "Fault prognosis / RUL — Predicts remaining useful life."),

        # MATURITY: Evidence
        ("Maturity", "Evidence", 0, "No convincing evidence — No validating data or claims/implementation are suspect (e.g., clear mistakes)."),
        ("Maturity", "Evidence", 1, "TRL3 Simulation / POC — Demonstrated via simulation; no physical experiments."),
        ("Maturity", "Evidence", 2, "TRL4 Lab testing — Demonstrated in a controlled lab setting."),
        ("Maturity", "Evidence", 3, "TRL5 Controlled field tests — Representative field environment, controlled conditions."),
        ("Maturity", "Evidence", 4, "TRL6 In-situ field tests — Actual asset in uncontrolled, production-like conditions."),

        # PRACTICALITY: Data
        ("Practicality", "Data", 0, "Completely impractical — Data unclear; implementation not practical."),
        ("Practicality", "Data", 1, "Impractical — Requires many examples of historical failure data for training."),
        ("Practicality", "Data", 2, "Somewhat practical — Requires limited historical failure examples (subset of modes)."),
        ("Practicality", "Data", 3, "Very practical — Requires no historical failure data for training/thresholds."),

        # PRACTICALITY: Asset
        ("Practicality", "Asset", 0, "Wrong asset for PdM — Non-critical asset; low production-loss risk; unlikely to justify PdM."),
        ("Practicality", "Asset", 1, "Sensible asset for PdM — Business-critical (e.g., mill, conveyor)."),

        # ADAPTABILITY: Environmental
        ("Adaptability", "Environmental", 1, "Narrow — Demonstrated under very limited operating conditions."),
        ("Adaptability", "Environmental", 2, "Broad — Demonstrated under varying operating conditions."),

        # ADAPTABILITY: Asset
        ("Adaptability", "Asset", 1, "Asset-specific — Tailored; not easily expanded beyond current application."),
        ("Adaptability", "Asset", 2, "Asset-agnostic — Can be applied across mining assets."),

        # COST: Economic
        ("Cost", "Economic", 1, "Unsubstantiated cost — Low production contribution or very high failure rate (PM would be better)."),
        ("Cost", "Economic", 2, "Acceptable cost — High production contribution with sporadic failures; use-based maintenance is poor."),

        # COST: Initial
        ("Cost", "Initial", 1, "High initial investment — Long expert time (months/years) or long shutdown (weeks) or mine-wide infrastructure."),
        ("Cost", "Initial", 2, "Moderate initial investment — Short expert time (weeks) or short shutdown (hours/days) or local infrastructure."),
        ("Cost", "Initial", 3, "Low initial investment — Minimal expert time/shutdown; leverages existing infra (e.g., CAN + small DAQ)."),

        # COST: Ongoing
        ("Cost", "Ongoing", 1, "High operational cost — Continuous monitoring/calibration and expert to interpret."),
        ("Cost", "Ongoing", 2, "Moderate operational cost — Regular replacement (e.g., wear sensor); no expert needed."),
        ("Cost", "Ongoing", 3, "Low operational cost — In-house skills; no ongoing calibration/replacement."),

        # TECHNICAL FIT: Organisational
        ("Technical fit", "Organisational", 1, "Expert team required — Needs expert personnel (e.g., data scientists)."),
        ("Technical fit", "Organisational", 2, "Standard engineering skills sufficient — Implement/maintain in-house."),

        # TECHNICAL FIT: Digital infrastructure
        ("Technical fit", "Digital infrastructure", 1, "New digital infrastructure — Requires installing additional sensors/infra (e.g., vibration monitoring systems)."),
        ("Technical fit", "Digital infrastructure", 2, "Existing digital infrastructure leveraged — Uses typical existing signals (currents, speeds, CAN, temperature)."),

        # TECHNICAL FIT: Edge-processing
        ("Technical fit", "Edge-processing", 1, "Solution requires cloud processing — Needs network/central compute or continual off-device tuning."),
        ("Technical fit", "Edge-processing", 2, "Solution can run on edge — Runs locally; stationary edge compute is fine for fixed assets."),
    ]
    cur.executemany(
        "INSERT INTO rubric (dimension, subdimension, score, explanation) VALUES (?, ?, ?, ?);",
        entries
    )

def seed_weights(cur):
    # Base weights (unnormalised) + blank reasons
    defaults = {
        "Interpretability": (1.0, ""),
        "Maturity": (1.0, ""),
        "Practicality": (0.6, ""),
        "Adaptability": (1.0, ""),
        "Cost": (0.3, ""),
        "Technical fit": (0.8, ""),
    }
    now = datetime.utcnow().isoformat()
    for dim, (w, reason) in defaults.items():
        cur.execute(
            "INSERT INTO weights (dimension, base_weight, reason, updated_at) VALUES (?, ?, ?, ?)",
            (dim, float(w), reason, now)
        )

def load_rubric_df() -> pd.DataFrame:
    conn = get_conn()
    df = pd.read_sql_query(
        "SELECT dimension, subdimension, score, explanation FROM rubric ORDER BY dimension, subdimension, score;",
        conn
    )
    conn.close()
    return df

def load_weights_df() -> pd.DataFrame:
    conn = get_conn()
    df = pd.read_sql_query(
        "SELECT dimension, base_weight, reason, updated_at FROM weights ORDER BY dimension;",
        conn
    )
    conn.close()
    return df

def save_weights(weights_map: Dict[str, Tuple[float, str]]):
    conn = get_conn()
    cur = conn.cursor()
    now = datetime.utcnow().isoformat()
    for dim, (w, reason) in weights_map.items():
        cur.execute("""
            INSERT INTO weights (dimension, base_weight, reason, updated_at)
            VALUES (?, ?, ?, ?)
            ON CONFLICT(dimension) DO UPDATE SET
                base_weight=excluded.base_weight,
                reason=excluded.reason,
                updated_at=excluded.updated_at;
        """, (dim, float(w), reason, now))
    conn.commit()
    conn.close()

def upsert_study(meta: dict, rows: pd.DataFrame, study_id: Optional[int] = None):
    conn = get_conn()
    cur = conn.cursor()

    if study_id is None:
        cur.execute("""
            INSERT INTO study (title, authors, year, reviewer, created_at)
            VALUES (?, ?, ?, ?, ?);
        """, (meta["title"], meta.get("authors"), meta.get("year"),
              meta.get("reviewer"), datetime.utcnow().isoformat()))
        study_id = cur.lastrowid
    else:
        cur.execute("""
            UPDATE study SET title=?, authors=?, year=?, reviewer=? WHERE id=?;
        """, (meta["title"], meta.get("authors"), meta.get("year"),
              meta.get("reviewer"), study_id))
        cur.execute("DELETE FROM study_score WHERE study_id=?;", (study_id,))

    for _, r in rows.iterrows():
        cur.execute("""
            INSERT INTO study_score (study_id, dimension, subdimension, score, reason)
            VALUES (?, ?, ?, ?, ?);
        """, (study_id, r["Dimension"], r["Sub-Dimension"], int(r["Score"]), r.get("Reason", "")))

    conn.commit()
    conn.close()
    return study_id

def all_studies_scores_df() -> pd.DataFrame:
    """Join studies with their scores."""
    conn = get_conn()
    df = pd.read_sql_query("""
        SELECT s.id, s.title, s.authors, s.year, s.reviewer, s.created_at,
               sc.dimension, sc.subdimension, sc.score, sc.reason
        FROM study s
        LEFT JOIN study_score sc ON s.id = sc.study_id
        ORDER BY s.created_at DESC, sc.dimension, sc.subdimension;
    """, conn)
    conn.close()
    return df

def list_studies_df() -> pd.DataFrame:
    """List study metadata only (for selection)."""
    conn = get_conn()
    df = pd.read_sql_query("""
        SELECT id, title, reviewer, year, created_at
        FROM study
        ORDER BY created_at DESC;
    """, conn)
    conn.close()
    return df

def get_study_scores(study_id: int):
    """Return (meta_dict, scores_df) for a study."""
    conn = get_conn()

    meta_df = pd.read_sql_query(
        "SELECT * FROM study WHERE id=?;",
        conn,
        params=(study_id,)
    )
    if meta_df.empty:
        conn.close()
        raise ValueError(f"Study ID {study_id} not found.")

    scores_df = pd.read_sql_query("""
        SELECT
            dimension AS 'Dimension',
            subdimension AS 'Sub-Dimension',
            score AS 'Score',
            reason AS 'Reason'
        FROM study_score
        WHERE study_id=?
        ORDER BY dimension, subdimension;
    """, conn, params=(study_id,))

    conn.close()
    return meta_df.iloc[0].to_dict(), scores_df

# =========================
# Rubric helpers
# =========================
def possible_scores_map(rubric_df: pd.DataFrame):
    """Return dict[(dimension, subdimension)] -> list of (score, explanation)."""
    mp = {}
    for _, r in rubric_df.iterrows():
        key = (r["dimension"], r["subdimension"])
        mp.setdefault(key, []).append((int(r["score"]), r["explanation"]))
    for k in mp:
        mp[k] = sorted(mp[k], key=lambda x: x[0])
    return mp

def max_score_for(key, mp):
    """Max allowed score for a (dimension, subdimension)."""
    return max(s for s, _ in mp.get(key, [(0, "")]))

def dimension_structure(rubric_df: pd.DataFrame):
    """Return ordered list of (dimension, [subdims]) using rubric contents."""
    dims = []
    for dim in rubric_df["dimension"].unique():
        sub = rubric_df[rubric_df["dimension"] == dim]["subdimension"].unique().tolist()
        dims.append((dim, sub))
    return dims

def blank_rows_from_rubric(rubric_df: pd.DataFrame):
    rows = []
    for dim, subdims in dimension_structure(rubric_df):
        for sd in subdims:
            rows.append({"Dimension": dim, "Sub-Dimension": sd, "Score": 0, "Reason": ""})
    return pd.DataFrame(rows)

def rows_from_db(rubric_df: pd.DataFrame, score_df: pd.DataFrame) -> pd.DataFrame:
    """Build score rows in rubric order, filled from DB (missing items default to 0/empty)."""
    rows = blank_rows_from_rubric(rubric_df)
    if score_df is None or score_df.empty:
        return rows
    for idx in rows.index:
        dim = rows.at[idx, "Dimension"]
        sub = rows.at[idx, "Sub-Dimension"]
        m = score_df[(score_df["Dimension"] == dim) & (score_df["Sub-Dimension"] == sub)]
        if not m.empty:
            s = m.iloc[0]["Score"]
            rows.at[idx, "Score"] = int(s) if pd.notnull(s) else 0
            rows.at[idx, "Reason"] = m.iloc[0].get("Reason", "") or ""
    return rows

# =========================
# Scoring logic
# =========================
def compute_dimension_scores(rows: pd.DataFrame, mp) -> Dict[str, Dict[str, float]]:
    """
    For each dimension:
      - Multiply sub-dimension scores to get raw dimension score
      - Compute max product from allowed maxima per sub-dimension
      - Normalise raw/max to [0,1]
    Returns: {dimension: {"raw": x, "max": m, "norm": n}}
    """
    out = {}
    if rows.empty:
        return out
    for dim, g in rows.groupby("Dimension"):
        raw = 1.0
        max_prod = 1.0
        has_any = False
        for _, r in g.iterrows():
            key = (r["Dimension"], r["Sub-Dimension"])
            s = int(r["Score"]) if pd.notnull(r["Score"]) else 0
            raw *= s
            max_prod *= max_score_for(key, mp)
            has_any = True
        if not has_any:
            raw = 0.0
            max_prod = 0.0
        norm = (raw / max_prod) if max_prod > 0 else 0.0
        out[dim] = {"raw": float(raw), "max": float(max_prod), "norm": float(norm)}
    return out

def weights_normalised(weights_df: pd.DataFrame, dim_order: list) -> Dict[str, float]:
    """Normalise base weights across provided dimensions so they sum to 1."""
    base = {row["dimension"]: float(row["base_weight"]) for _, row in weights_df.iterrows()}
    used = {d: base.get(d, 0.0) for d in dim_order}
    total = sum(used.values())
    if total <= 0:
        n = len(dim_order) if dim_order else 1
        return {d: 1.0 / n for d in dim_order}
    return {d: (w / total) for d, w in used.items()}

def final_impact_score(dim_scores: Dict[str, Dict[str, float]], wnorm: Dict[str, float]) -> float:
    """Σ (normalised_weight_d * normalised_dimension_score_d)"""
    total = 0.0
    for dim, w in wnorm.items():
        s = dim_scores.get(dim, {"norm": 0.0})["norm"]
        total += w * s
    return float(total)

# =========================
# Streamlit App
# =========================
st.set_page_config(page_title="PdM Rubric Scoring", layout="wide")
st.title("PdM Rubric Scoring & Repository")

init_db()
rubric_df = load_rubric_df()
weights_df = load_weights_df()
mp = possible_scores_map(rubric_df)
dim_order = [d for d, _ in dimension_structure(rubric_df)]
wnorm_map = weights_normalised(weights_df, dim_order)

tabs = st.tabs(["Score a Case Study", "Weights & Rationale", "Browse Database", "Manage Rubric"])

# ---- Tab 1: Scoring ----
with tabs[0]:
    # ----- Load / New controls -----
    st.subheader("Load or start new")
    lst = list_studies_df()
    if not lst.empty:
        # build labels like "12 — Paper Title (2024) — Reviewer"
        labels = []
        id_map = {}
        for _, row in lst.iterrows():
            yr = str(row['year']) if pd.notnull(row['year']) else 'n/a'
            label = f"{int(row['id'])} — {row['title']} ({yr}) — {row.get('reviewer','') or ''}"
            labels.append(label)
            id_map[label] = int(row["id"])
        cL, cBtnL, cNew = st.columns([6, 2, 2])
        with cL:
            choice = selectbox_compat("Pick a study to load", options=labels, index=0, key="load_choice")
        with cBtnL:
            if st.button("Load selected"):
                sid = id_map[st.session_state["load_choice"]]
                meta, score_df = get_study_scores(sid)
                # mark editing
                st.session_state["editing_study_id"] = sid
                # seed meta fields
                st.session_state["meta_title"] = meta.get("title", "")
                st.session_state["meta_authors"] = meta.get("authors", "")
                st.session_state["meta_year"] = int(meta["year"]) if meta.get("year") is not None else 2025
                st.session_state["meta_reviewer"] = meta.get("reviewer", "")
                # seed score rows
                st.session_state["score_rows"] = rows_from_db(rubric_df, score_df)
                st.success(f"Loaded study ID {sid} for editing.")
        with cNew:
            if st.button("New entry"):
                st.session_state["editing_study_id"] = None
                st.session_state["meta_title"] = ""
                st.session_state["meta_authors"] = ""
                st.session_state["meta_year"] = 2025
                st.session_state["meta_reviewer"] = ""
                st.session_state["score_rows"] = blank_rows_from_rubric(rubric_df)
                st.info("Cleared form for a new entry.")
    else:
        st.info("No saved studies yet. Start a new entry below.")

    st.markdown("---")
    st.subheader("1) Study Metadata")

    # Ensure meta defaults exist
    st.session_state.setdefault("editing_study_id", None)
    st.session_state.setdefault("meta_title", "")
    st.session_state.setdefault("meta_authors", "")
    st.session_state.setdefault("meta_year", 2025)
    st.session_state.setdefault("meta_reviewer", "")
    st.session_state.setdefault("score_rows", blank_rows_from_rubric(rubric_df))

    with st.form("meta_form"):
        col1, col2 = st.columns(2)
        with col1:
            title = st.text_input("Title *", key="meta_title")
            authors = st.text_input("Authors", key="meta_authors")
        with col2:
            year = st.number_input("Year", min_value=1900, max_value=2100, step=1, key="meta_year")
            reviewer = st.text_input("Reviewer (your name)", key="meta_reviewer")
        _meta_submitted = st.form_submit_button("Apply metadata (continue below)")

    st.markdown("---")
    st.subheader("2) Scores by Dimension (products, normalised per dimension)")

    if st.session_state["meta_title"]:
        edited_rows = st.session_state["score_rows"].copy()

        for idx in edited_rows.index:
            r = edited_rows.loc[idx]
            key = (r["Dimension"], r["Sub-Dimension"])

            with st.container():
                # Give comments more space; keep score control narrow.
                c1, c2, c3, c4 = st.columns([2, 6, 2, 8])

                # Labels
                c1.write(f"**{r['Dimension']}**")
                c1.write(r["Sub-Dimension"])

                # Show ALL rubric options inline (static list)
                opts = mp.get(key, [])
                if opts:
                    c2.markdown(
                        "**Options:**\n" +
                        "\n".join([f"- **{s}** — {e}" for s, e in opts])
                    )
                    options_only = [s for s, _ in opts]
                else:
                    c2.info("No options found in rubric for this item.")
                    options_only = [0]

                # Compact radio for score
                current = int(r["Score"]) if pd.notnull(r["Score"]) else options_only[0]
                try:
                    idx_opt = options_only.index(current)
                except ValueError:
                    idx_opt = 0

                sel = radio_compat(
                    "Score",
                    options=options_only,
                    index=idx_opt,
                    key=f"score_radio_{idx}",
                    horizontal=True,
                )
                edited_rows.at[idx, "Score"] = sel

                # Larger comment box (encourage detailed reasoning)
                reason = c4.text_area(
                    "Reason / Notes",
                    value=r.get("Reason", ""),
                    key=f"reason_{idx}",
                    height=140,
                    placeholder="Why did you pick this score? Cite evidence, assumptions, data availability, etc."
                )
                edited_rows.at[idx, "Reason"] = reason

                st.markdown("---")

        st.session_state["score_rows"] = edited_rows

        # === Live dimension products & final score ===
        weights_df = load_weights_df()
        wnorm_map = weights_normalised(weights_df, dim_order)

        dim_scores = compute_dimension_scores(edited_rows, mp)
        st.subheader("3) Dimension Scores (product & normalised)")
        cols = st.columns(3)
        with cols[0]:
            st.write("**Raw products**")
            for d in dim_order:
                v = dim_scores.get(d, {"raw": 0.0})["raw"]
                st.write(f"- {d}: {v:g}")
        with cols[1]:
            st.write("**Max products**")
            for d in dim_order:
                m = dim_scores.get(d, {"max": 0.0})["max"]
                st.write(f"- {d}: {m:g}")
        with cols[2]:
            st.write("**Normalised (0–1)**")
            for d in dim_order:
                n = dim_scores.get(d, {"norm": 0.0})["norm"]
                st.write(f"- {d}: {n:.3f}")

        # Final IMPACT adoptability score
        final_score = final_impact_score(dim_scores, wnorm_map)
        st.subheader("4) IMPACT adoptability score")
        st.write("**Normalised weights:**")
        st.write("\n".join([f"- {d}: w={wnorm_map.get(d, 0.0):.3f}" for d in dim_order]))
        st.write(f"**Final score:** `{final_score:.3f}`")

        # Save / Update
        is_editing = st.session_state.get("editing_study_id") is not None
        btn_label = "Update existing study" if is_editing else "Save as new study"

        if st.button(btn_label):
            meta = {
                "title": st.session_state["meta_title"],
                "authors": st.session_state["meta_authors"],
                "year": int(st.session_state["meta_year"]) if st.session_state["meta_year"] else None,
                "reviewer": st.session_state["meta_reviewer"]
            }
            sid = upsert_study(meta, edited_rows, st.session_state["editing_study_id"])
            st.session_state["editing_study_id"] = sid  # keep editing the same record
            # Optional: force a checkpoint so external viewers see changes immediately
            try:
                c = get_conn()
                c.execute("PRAGMA wal_checkpoint(FULL);")
                c.close()
            except Exception:
                pass
            st.success(f"{'Updated' if is_editing else 'Saved'} study ID {sid}")

# ---- Tab 2: Weights & Rationale ----
with tabs[1]:
    st.subheader("Global weights (base weights; they get normalised automatically)")
    st.caption("Change base weights and reasons below. The final score uses normalised weights that sum to 1.")
    # Reload weights fresh each run
    weights_df = load_weights_df()
    current = {row["dimension"]: (float(row["base_weight"]), row["reason"]) for _, row in weights_df.iterrows()}

    with st.form("weights_form"):
        edits = {}
        for d in dim_order:
            base_w, reason = current.get(d, (0.0, ""))
            st.markdown(f"### {d}")
            w = st.number_input(f"Base weight for {d}", min_value=0.0, step=0.1, value=float(base_w), key=f"w_{d}")
            r = st.text_area(f"Reason for {d}", value=reason, key=f"r_{d}", height=100,
                             placeholder="Explain why this dimension should be emphasised or down-weighted.")
            st.markdown("---")
            edits[d] = (w, r)
        submitted = st.form_submit_button("Save weights")
        if submitted:
            save_weights(edits)
            st.success("Weights saved. Scores will reflect the new weights immediately on next run/interaction.")

    # Show normalised weights preview
    weights_df = load_weights_df()
    wnorm_map = weights_normalised(weights_df, dim_order)
    st.write("**Current normalised weights:**")
    df_show(pd.DataFrame([{"Dimension": d, "Normalised Weight": wnorm_map.get(d, 0.0)} for d in dim_order]))

# ---- Tab 3: Browse ----
with tabs[2]:
    st.subheader("Search & Export")
    df_scores = all_studies_scores_df()
    weights_df = load_weights_df()
    wnorm_map = weights_normalised(weights_df, dim_order)

    if not df_scores.empty:
        colf1, colf2, colf3 = st.columns(3)
        with colf1:
            reviewer_f = st.text_input("Filter by reviewer")
        with colf2:
            year_f = st.text_input("Filter by year (e.g., 2023)")
        with colf3:
            text_f = st.text_input("Search in title/authors")

        view = df_scores.copy()
        if reviewer_f:
            view = view[view["reviewer"].fillna("").str.contains(reviewer_f, case=False, na=False)]
        if year_f:
            view = view[view["year"].astype(str) == year_f]
        if text_f:
            mask = (
                view["title"].fillna("").str.contains(text_f, case=False, na=False) |
                view["authors"].fillna("").str.contains(text_f, case=False, na=False)
            )
            view = view[mask]

        st.write("### Studies")
        overall_rows = []
        for sid, g in view.groupby("id"):
            rows = pd.DataFrame({
                "Dimension": g["dimension"],
                "Sub-Dimension": g["subdimension"],
                "Score": g["score"].fillna(0).astype(int),
                "Reason": g["reason"].fillna("")
            })
            dscores = compute_dimension_scores(rows, mp)
            final = final_impact_score(dscores, wnorm_map)
            meta = g.iloc[0]
            overall_rows.append({
                "Study ID": sid,
                "Title": meta["title"],
                "Reviewer": meta.get("reviewer", ""),
                "Year": meta.get("year", ""),
                "Final (0–1)": round(final, 3),
                **{f"{d} (norm)": round(dscores.get(d, {"norm": 0.0})["norm"], 3) for d in dim_order},
            })
        sdf = pd.DataFrame(overall_rows).sort_values(by="Study ID", ascending=False)
        df_show(sdf)

        # Drill into a single study
        st.write("### Drill into a study")
        ids = sorted(view["id"].dropna().unique().tolist(), reverse=True)
        if ids:
            sid = selectbox_compat("Choose Study ID", options=ids)
            sg = df_scores[df_scores["id"] == sid]
            rows = pd.DataFrame({
                "Dimension": sg["dimension"],
                "Sub-Dimension": sg["subdimension"],
                "Score": sg["score"].fillna(0).astype(int),
                "Reason": sg["reason"].fillna("")
            })
            dscores = compute_dimension_scores(rows, mp)
            final = final_impact_score(dscores, wnorm_map)
            st.write(f"**Final IMPACT adoptability score:** `{final:.3f}`")
            br = []
            for d in dim_order:
                info = dscores.get(d, {"raw": 0.0, "max": 0.0, "norm": 0.0})
                br.append({"Dimension": d, "Raw product": info["raw"], "Max product": info["max"], "Normalised": round(info["norm"], 3)})
            df_show(pd.DataFrame(br))
            # Export single study raw rows
            try:
                csv = sg.to_csv(index=False).encode("utf-8")
                st.download_button("Download raw study scores (CSV)", data=csv, file_name=f"study_{sid}_raw_scores.csv", mime="text/csv")
            except Exception:
                pass

        # Export summary
        try:
            csv2 = sdf.to_csv(index=False).encode("utf-8")
            st.download_button("Download summary (CSV)", data=csv2, file_name="studies_summary.csv", mime="text/csv")
        except Exception:
            pass
    else:
        st.info("No studies yet. Add one in the 'Score a Case Study' tab.")

# ---- Tab 4: Manage Rubric ----
with tabs[3]:
    st.write("View or export the current rubric.")
    df_show(rubric_df)
    # Export rubric
    try:
        csv3 = rubric_df.to_csv(index=False).encode("utf-8")
        st.download_button("Download rubric (CSV)", data=csv3, file_name="rubric_definitions.csv", mime="text/csv")
    except Exception:
        pass
