import os
from typing import Optional, Dict, Tuple
from datetime import datetime
from urllib.parse import urlparse

import pandas as pd
import streamlit as st
from sqlalchemy import create_engine, text

# =========================
# Streamlit page
# =========================
st.set_page_config(page_title="PdM Rubric Scoring", layout="wide")
st.title("PdM Rubric Scoring & Repository")

# =========================
# Helpers: UI compatibility
# =========================
def df_show(df: pd.DataFrame, height: Optional[int] = None):
    try:
        st.dataframe(df, use_container_width=True, height=height)
    except TypeError:
        try:
            st.dataframe(df, height=height)
        except TypeError:
            st.dataframe(df)

def selectbox_compat(label, options, index=0, key=None):
    return st.selectbox(label, options=options, index=index, key=key)

def radio_compat(label, options, index=0, key=None, horizontal=True):
    try:
        return st.radio(label, options, index=index, key=key, horizontal=horizontal)
    except TypeError:
        return st.radio(label, options, index=index, key=key)

# =========================
# Database: engine + schema
# =========================
def _database_url() -> str:
    url = st.secrets.get("DATABASE_URL", os.environ.get("DATABASE_URL", ""))
    return url

@st.cache_resource(show_spinner=False)
def get_engine_and_init():
    """
    Create the SQLAlchemy engine once per server and ensure schema is present.
    This prevents repeated connects/initialisation on every small rerun.
    """
    url = _database_url()
    if not url:
        raise RuntimeError("DATABASE_URL missing in secrets or environment.")

    # Small, conservative pool; pre-ping to drop dead conns; short connect timeout.
    engine = create_engine(
        url,
        pool_pre_ping=True,
        pool_size=2,
        max_overflow=0,
        pool_recycle=300,
        connect_args={"connect_timeout": 5},
    )

    with engine.begin() as conn:
        conn.execute(text("""
            CREATE TABLE IF NOT EXISTS rubric (
                id SERIAL PRIMARY KEY,
                dimension TEXT NOT NULL,
                subdimension TEXT NOT NULL,
                score INTEGER NOT NULL,
                explanation TEXT NOT NULL
            );
        """))
        conn.execute(text("""
            CREATE TABLE IF NOT EXISTS study (
                id SERIAL PRIMARY KEY,
                title TEXT NOT NULL,
                authors TEXT,
                year INTEGER,
                reviewer TEXT,
                created_at TIMESTAMPTZ NOT NULL
            );
        """))
        conn.execute(text("""
            CREATE TABLE IF NOT EXISTS study_score (
                id SERIAL PRIMARY KEY,
                study_id INTEGER NOT NULL REFERENCES study(id) ON DELETE CASCADE,
                dimension TEXT NOT NULL,
                subdimension TEXT NOT NULL,
                score INTEGER NOT NULL,
                reason TEXT
            );
        """))
        conn.execute(text("""
            CREATE TABLE IF NOT EXISTS weights (
                dimension TEXT PRIMARY KEY,
                base_weight DOUBLE PRECISION NOT NULL,
                reason TEXT,
                updated_at TIMESTAMPTZ NOT NULL
            );
        """))

        # Seed rubric if empty
        if conn.execute(text("SELECT COUNT(*) FROM rubric;")).scalar_one() == 0:
            seed_rubric(conn)

        # Seed weights if empty
        if conn.execute(text("SELECT COUNT(*) FROM weights;")).scalar_one() == 0:
            seed_weights(conn)

    return engine

def seed_rubric(conn):
    entries = [
        ("Interpretability", "-", 0, "Black box — Deep/complex models without interpretation; includes hybrids where a data-driven model outputs the prediction."),
        ("Interpretability", "-", 1, "Post-hoc interpretability — SHAP/LIME/saliency explain predictions after the fact; decision process remains opaque."),
        ("Interpretability", "-", 2, "Interpretable throughout process — Built-in transparency (trees/rules/linear/logistic/simple signal rules); not physics-rooted."),
        ("Interpretability", "-", 3, "Physics-focused — Physics or hybrid models where outputs are based on physics; highest interpretability."),

        ("Maturity", "Technique", 1, "Anomaly detection — Detects anomalies in operating conditions."),
        ("Maturity", "Technique", 2, "Fault detection — Differentiates anomaly vs. fault."),
        ("Maturity", "Technique", 3, "Fault diagnosis — Identifies which fault type is present."),
        ("Maturity", "Technique", 4, "Fault prognosis / RUL — Predicts remaining useful life."),

        ("Maturity", "Evidence", 0, "No convincing evidence — No validating data or claims/implementation are suspect (e.g., clear mistakes)."),
        ("Maturity", "Evidence", 1, "TRL3 Simulation / POC — Demonstrated via simulation; no physical experiments."),
        ("Maturity", "Evidence", 2, "TRL4 Lab testing — Demonstrated in a controlled lab setting."),
        ("Maturity", "Evidence", 3, "TRL5 Controlled field tests — Representative field environment, controlled conditions."),
        ("Maturity", "Evidence", 4, "TRL6 In-situ field tests — Actual asset in uncontrolled, production-like conditions."),

        ("Practicality", "Data", 0, "Completely impractical — Data unclear; implementation not practical."),
        ("Practicality", "Data", 1, "Impractical — Requires many examples of historical failure data for training."),
        ("Practicality", "Data", 2, "Somewhat practical — Requires limited historical failure examples (subset of modes)."),
        ("Practicality", "Data", 3, "Very practical — Requires no historical failure data for training/thresholds."),

        ("Practicality", "Asset", 0, "Wrong asset for PdM — Non-critical asset; low production-loss risk; unlikely to justify PdM."),
        ("Practicality", "Asset", 1, "Sensible asset for PdM — Business-critical (e.g., mill, conveyor)."),

        ("Adaptability", "Environmental", 1, "Narrow — Demonstrated under very limited operating conditions."),
        ("Adaptability", "Environmental", 2, "Broad — Demonstrated under varying operating conditions."),

        ("Adaptability", "Asset", 1, "Asset-specific — Tailored; not easily expanded beyond current application."),
        ("Adaptability", "Asset", 2, "Asset-agnostic — Can be applied across mining assets."),

        ("Cost", "Economic", 1, "Unsubstantiated cost — Low production contribution or very high failure rate (PM would be better)."),
        ("Cost", "Economic", 2, "Acceptable cost — High production contribution with sporadic failures; use-based maintenance is poor."),

        ("Cost", "Initial", 1, "High initial investment — Long expert time (months/years) or long shutdown (weeks) or mine-wide infrastructure."),
        ("Cost", "Initial", 2, "Moderate initial investment — Short expert time (weeks) or short shutdown (hours/days) or local infrastructure."),
        ("Cost", "Initial", 3, "Low initial investment — Minimal expert time/shutdown; leverages existing infra (e.g., CAN + small DAQ)."),

        ("Cost", "Ongoing", 1, "High operational cost — Continuous monitoring/calibration and expert to interpret."),
        ("Cost", "Ongoing", 2, "Moderate operational cost — Regular replacement (e.g., wear sensor); no expert needed."),
        ("Cost", "Ongoing", 3, "Low operational cost — In-house skills; no ongoing calibration/replacement."),

        ("Technical fit", "Organisational", 1, "Expert team required — Needs expert personnel (e.g., data scientists)."),
        ("Technical fit", "Organisational", 2, "Standard engineering skills sufficient — Implement/maintain in-house."),

        ("Technical fit", "Digital infrastructure", 1, "New digital infrastructure — Requires installing additional sensors/infra (e.g., vibration monitoring systems)."),
        ("Technical fit", "Digital infrastructure", 2, "Existing digital infrastructure leveraged — Uses typical existing signals (currents, speeds, CAN, temperature)."),

        ("Technical fit", "Edge-processing", 1, "Solution requires cloud processing — Needs network/central compute or continual off-device tuning."),
        ("Technical fit", "Edge-processing", 2, "Solution can run on edge — Runs locally; stationary edge compute is fine for fixed assets."),
    ]
    for dim, sub, sc, expl in entries:
        conn.execute(
            text("INSERT INTO rubric (dimension, subdimension, score, explanation) VALUES (:d, :s, :sc, :e)"),
            {"d": dim, "s": sub, "sc": int(sc), "e": expl}
        )

def seed_weights(conn):
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
        conn.execute(
            text("""
                INSERT INTO weights (dimension, base_weight, reason, updated_at)
                VALUES (:d, :w, :r, :t)
                ON CONFLICT(dimension) DO UPDATE SET
                    base_weight = EXCLUDED.base_weight,
                    reason      = EXCLUDED.reason,
                    updated_at  = EXCLUDED.updated_at;
            """),
            {"d": dim, "w": float(w), "r": reason, "t": now}
        )

# Try to create/connect once; don't crash UI on hiccups
try:
    engine = get_engine_and_init()
    DB_OK = True
except Exception as e:
    engine = None
    DB_OK = False
    st.warning("Database isn’t reachable right now. You can still fill the form; click **Retry connection** later to save.")
    if st.button("Retry connection"):
        st.rerun()

# =========================
# Data access funcs
# =========================
def load_rubric_df() -> pd.DataFrame:
    with engine.connect() as conn:
        return pd.read_sql_query(
            text("SELECT dimension, subdimension, score, explanation FROM rubric ORDER BY dimension, subdimension, score;"),
            conn
        )

def load_weights_df() -> pd.DataFrame:
    with engine.connect() as conn:
        return pd.read_sql_query(
            text("SELECT dimension, base_weight, reason, updated_at FROM weights ORDER BY dimension;"),
            conn
        )

def save_weights(weights_map: Dict[str, Tuple[float, str]]):
    now = datetime.utcnow().isoformat()
    with engine.begin() as conn:
        for dim, (w, reason) in weights_map.items():
            conn.execute(
                text("""
                    INSERT INTO weights (dimension, base_weight, reason, updated_at)
                    VALUES (:d, :w, :r, :t)
                    ON CONFLICT(dimension) DO UPDATE SET
                        base_weight = EXCLUDED.base_weight,
                        reason      = EXCLUDED.reason,
                        updated_at  = EXCLUDED.updated_at;
                """),
                {"d": dim, "w": float(w), "r": reason, "t": now}
            )

def upsert_study(meta: dict, rows: pd.DataFrame, study_id: Optional[int] = None):
    with engine.begin() as conn:
        if study_id is None:
            new_id = conn.execute(
                text("""
                    INSERT INTO study (title, authors, year, reviewer, created_at)
                    VALUES (:title, :authors, :year, :reviewer, :created_at)
                    RETURNING id;
                """),
                {
                    "title": meta["title"],
                    "authors": meta.get("authors"),
                    "year": meta.get("year"),
                    "reviewer": meta.get("reviewer"),
                    "created_at": datetime.utcnow().isoformat()
                }
            ).scalar_one()
            study_id = int(new_id)
        else:
            conn.execute(
                text("""UPDATE study SET title=:title, authors=:authors, year=:year, reviewer=:reviewer WHERE id=:id;"""),
                {
                    "title": meta["title"],
                    "authors": meta.get("authors"),
                    "year": meta.get("year"),
                    "reviewer": meta.get("reviewer"),
                    "id": study_id
                }
            )
            conn.execute(text("DELETE FROM study_score WHERE study_id=:id;"), {"id": study_id})

        for _, r in rows.iterrows():
            conn.execute(
                text("""
                    INSERT INTO study_score (study_id, dimension, subdimension, score, reason)
                    VALUES (:sid, :d, :s, :sc, :r);
                """),
                {"sid": study_id, "d": r["Dimension"], "s": r["Sub-Dimension"], "sc": int(r["Score"]), "r": r.get("Reason", "")}
            )
    return study_id

def all_studies_scores_df() -> pd.DataFrame:
    with engine.connect() as conn:
        return pd.read_sql_query(text("""
            SELECT s.id, s.title, s.authors, s.year, s.reviewer, s.created_at,
                   sc.dimension, sc.subdimension, sc.score, sc.reason
            FROM study s
            LEFT JOIN study_score sc ON s.id = sc.study_id
            ORDER BY s.created_at DESC, sc.dimension, sc.subdimension;
        """), conn)

def list_studies_df() -> pd.DataFrame:
    with engine.connect() as conn:
        return pd.read_sql_query(text("""
            SELECT id, title, reviewer, year, created_at
            FROM study
            ORDER BY created_at DESC;
        """), conn)

def get_study_scores(study_id: int):
    with engine.connect() as conn:
        meta_df = pd.read_sql_query(text("SELECT * FROM study WHERE id=:id;"), conn, params={"id": study_id})
        if meta_df.empty:
            raise ValueError(f"Study ID {study_id} not found.")
        scores_df = pd.read_sql_query(text("""
            SELECT
                dimension AS "Dimension",
                subdimension AS "Sub-Dimension",
                score AS "Score",
                reason AS "Reason"
            FROM study_score
            WHERE study_id=:id
            ORDER BY dimension, subdimension;
        """), conn, params={"id": study_id})
    return meta_df.iloc[0].to_dict(), scores_df

# =========================
# Rubric helpers & scoring
# =========================
def possible_scores_map(rubric_df: pd.DataFrame):
    mp = {}
    for _, r in rubric_df.iterrows():
        key = (r["dimension"], r["subdimension"])
        mp.setdefault(key, []).append((int(r["score"]), r["explanation"]))
    for k in mp:
        mp[k] = sorted(mp[k], key=lambda x: x[0])
    return mp

def max_score_for(key, mp):
    return max(s for s, _ in mp.get(key, [(0, "")]))

def dimension_structure(rubric_df: pd.DataFrame):
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

def compute_dimension_scores(rows: pd.DataFrame, mp) -> Dict[str, Dict[str, float]]:
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
    base = {row["dimension"]: float(row["base_weight"]) for _, row in weights_df.iterrows()}
    used = {d: base.get(d, 0.0) for d in dim_order}
    total = sum(used.values())
    if total <= 0:
        n = len(dim_order) if dim_order else 1
        return {d: 1.0 / n for d in dim_order}
    return {d: (w / total) for d, w in used.items()}

def final_impact_score(dim_scores: Dict[str, Dict[str, float]], wnorm: Dict[str, float]) -> float:
    total = 0.0
    for dim, w in wnorm.items():
        s = dim_scores.get(dim, {"norm": 0.0})["norm"]
        total += w * s
    return float(total)

# =========================
# Load rubric/weights (tolerant)
# =========================
if DB_OK:
    try:
        rubric_df = load_rubric_df()
    except Exception:
        st.info("Rubric temporarily unavailable; empty rubric shown.")
        rubric_df = pd.DataFrame(columns=["dimension","subdimension","score","explanation"])
    try:
        weights_df = load_weights_df()
    except Exception:
        st.info("Weights temporarily unavailable; defaults will apply.")
        weights_df = pd.DataFrame(columns=["dimension","base_weight","reason","updated_at"])
else:
    rubric_df = pd.DataFrame(columns=["dimension","subdimension","score","explanation"])
    weights_df = pd.DataFrame(columns=["dimension","base_weight","reason","updated_at"])

mp = possible_scores_map(rubric_df) if not rubric_df.empty else {}
dim_order = [d for d, _ in dimension_structure(rubric_df)] if not rubric_df.empty else []

# =========================
# Tabs (requested order)
# =========================
tabs = st.tabs(["Score a Case Study", "Browse Database", "Weights & Rationale", "Manage Rubric"])

# --------------------------------------------------------------------
# TAB 1: SCORE A CASE STUDY  (single Save button; no preview breakdown)
# --------------------------------------------------------------------
with tabs[0]:
    st.subheader("Load or start new")

    # List studies (tolerant)
    lst = pd.DataFrame()
    if DB_OK:
        try:
            lst = list_studies_df()
        except Exception:
            st.info("Couldn’t fetch the study list just now. You can still create a new entry; try reload later.")
            lst = pd.DataFrame()

    if not lst.empty:
        labels, id_map = [], {}
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
                if not rubric_df.empty:
                    sid = id_map[st.session_state["load_choice"]]
                    try:
                        meta, score_df = get_study_scores(sid)
                        st.session_state["editing_study_id"] = sid
                        st.session_state["meta_title"] = meta.get("title", "")
                        st.session_state["meta_authors"] = meta.get("authors", "")
                        st.session_state["meta_year"] = int(meta["year"]) if meta.get("year") is not None else 2025
                        st.session_state["meta_reviewer"] = meta.get("reviewer", "")
                        st.session_state["score_rows"] = rows_from_db(rubric_df, score_df)
                        st.success(f"Loaded study ID {sid} for editing.")
                    except Exception as e:
                        st.error(f"Failed to load study: {e}")
                else:
                    st.error("Rubric not available; cannot load scores right now.")
        with cNew:
            if st.button("New entry"):
                st.session_state["editing_study_id"] = None
                st.session_state["meta_title"] = ""
                st.session_state["meta_authors"] = ""
                st.session_state["meta_year"] = 2025
                st.session_state["meta_reviewer"] = ""
                st.session_state["score_rows"] = blank_rows_from_rubric(rubric_df) if not rubric_df.empty else pd.DataFrame()
                st.info("Cleared form for a new entry.")
    else:
        st.info("No saved studies yet. Start a new entry below.")

    st.markdown("---")
    st.subheader("1) Study metadata")

    # Defaults in session_state
    st.session_state.setdefault("editing_study_id", None)
    st.session_state.setdefault("meta_title", "")
    st.session_state.setdefault("meta_authors", "")
    st.session_state.setdefault("meta_year", 2025)
    st.session_state.setdefault("meta_reviewer", "")
    st.session_state.setdefault("score_rows", blank_rows_from_rubric(rubric_df) if not rubric_df.empty else pd.DataFrame())

    # Simple inputs (not saved until you click Save to database)
    col1, col2 = st.columns(2)
    with col1:
        st.text_input("Title *", key="meta_title")
        st.text_input("Authors", key="meta_authors")
    with col2:
        st.number_input("Year", min_value=1900, max_value=2100, step=1, key="meta_year")
        st.text_input("Reviewer (your name)", key="meta_reviewer")

    st.markdown("---")
    st.subheader("2) Scores by dimension")

    if rubric_df.empty:
        st.error("Rubric not available right now. Please try reloading later.")
    else:
        # Single form that also performs the save
        with st.form("scores_form"):
            edited_rows = st.session_state["score_rows"].copy()

            for idx in edited_rows.index:
                r = edited_rows.loc[idx]
                key = (r["Dimension"], r["Sub-Dimension"])

                with st.container():
                    c1, c2, c3, c4 = st.columns([2, 6, 2, 8])

                    # Labels
                    c1.write(f"**{r['Dimension']}**")
                    c1.write(r["Sub-Dimension"])

                    # Options list
                    opts = mp.get(key, [])
                    if opts:
                        c2.markdown("**Options:**\n" + "\n".join([f"- **{s}** — {e}" for s, e in opts]))
                        options_only = [s for s, _ in opts]
                    else:
                        c2.info("No options found in rubric for this item.")
                        options_only = [0]

                    # Score radio
                    current = int(r["Score"]) if pd.notnull(r["Score"]) else options_only[0]
                    try:
                        idx_opt = options_only.index(current)
                    except ValueError:
                        idx_opt = 0

                    sel = radio_compat("Score", options=options_only, index=idx_opt,
                                       key=f"score_radio_{idx}", horizontal=True)
                    edited_rows.at[idx, "Score"] = sel

                    # Reason box
                    reason = c4.text_area(
                        "Reason / Notes",
                        value=r.get("Reason", ""),
                        key=f"reason_{idx}",
                        height=140,
                        placeholder="Why did you pick this score? Cite evidence, assumptions, data availability, etc."
                    )
                    edited_rows.at[idx, "Reason"] = reason

                    st.markdown("---")

            # Single save button (writes metadata + scores)
            saved = st.form_submit_button("Save to database")
            if saved:
                # Persist staged rows locally so nothing is lost on hiccup
                st.session_state["score_rows"] = edited_rows

                # Validate minimal metadata
                if not st.session_state["meta_title"]:
                    st.error("Please enter a Title before saving.")
                    st.stop()

                if not DB_OK:
                    st.error("Database is currently unavailable. Your inputs are safe on this page—click Retry at the top, then Save again.")
                    st.stop()

                meta = {
                    "title": st.session_state["meta_title"],
                    "authors": st.session_state["meta_authors"],
                    "year": int(st.session_state["meta_year"]) if st.session_state["meta_year"] else None,
                    "reviewer": st.session_state["meta_reviewer"]
                }

                try:
                    sid = upsert_study(meta, edited_rows, st.session_state.get("editing_study_id"))
                    st.session_state["editing_study_id"] = sid
                    st.success(f"Saved study ID {sid}")
                except Exception as e:
                    st.error(f"Save failed due to a DB error: {e}. Your form entries are still here—press Save again once the DB is reachable.")
                    st.stop()

# ---------------------------------------
# TAB 2: BROWSE DATABASE (with summaries)
# ---------------------------------------
with tabs[1]:
    st.subheader("Search & export")
    if DB_OK:
        try:
            df_scores = all_studies_scores_df()
        except Exception:
            st.info("Scores view unavailable right now—DB connection issue.")
            df_scores = pd.DataFrame()
    else:
        df_scores = pd.DataFrame()

    # Load weights fresh for scoring summaries
    if DB_OK:
        try:
            weights_df = load_weights_df()
        except Exception:
            weights_df = pd.DataFrame(columns=["dimension","base_weight","reason","updated_at"])
    else:
        weights_df = pd.DataFrame(columns=["dimension","base_weight","reason","updated_at"])

    wnorm_map = weights_normalised(weights_df, dim_order) if not weights_df.empty else {d: 1/len(dim_order) for d in dim_order} if dim_order else {}

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
            dscores = compute_dimension_scores(rows, mp) if not rows.empty else {}
            final = final_impact_score(dscores, wnorm_map) if dscores else 0.0
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

        # Export summary
        try:
            csv2 = sdf.to_csv(index=False).encode("utf-8")
            st.download_button("Download summary (CSV)", data=csv2, file_name="studies_summary.csv", mime="text/csv")
        except Exception:
            pass
    else:
        st.info("No studies yet. Add one in the 'Score a Case Study' tab.")

# -------------------------------------
# TAB 3: WEIGHTS & RATIONALE (editable)
# -------------------------------------
with tabs[2]:
    st.subheader("Global weights (base weights; they get normalised automatically)")
    st.caption("Change base weights and reasons below. The final score uses normalised weights that sum to 1.")

    if not DB_OK:
        st.warning("DB is unavailable; cannot edit weights right now.")
    else:
        try:
            weights_df = load_weights_df()
        except Exception:
            weights_df = pd.DataFrame(columns=["dimension","base_weight","reason","updated_at"])

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
                try:
                    save_weights(edits)
                    st.success("Weights saved.")
                except Exception as e:
                    st.error(f"Failed to save weights: {e}")

        # Show normalised weights preview
        try:
            weights_df = load_weights_df()
            wnorm_map = weights_normalised(weights_df, dim_order)
            st.write("**Current normalised weights:**")
            df_show(pd.DataFrame([{"Dimension": d, "Normalised Weight": wnorm_map.get(d, 0.0)} for d in dim_order]))
        except Exception:
            st.info("Could not load weights for preview.")

# -------------------------
# TAB 4: MANAGE RUBRIC
# -------------------------
with tabs[3]:
    st.write("View or export the current rubric.")
    if DB_OK:
        try:
            df = load_rubric_df()
            df_show(df)
            csv3 = df.to_csv(index=False).encode("utf-8")
            st.download_button("Download rubric (CSV)", data=csv3, file_name="rubric_definitions.csv", mime="text/csv")
        except Exception:
            st.info("Rubric unavailable right now.")
    else:
        st.info("Rubric unavailable: DB not connected.")
