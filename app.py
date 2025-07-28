import streamlit as st
import pandas as pd
import plotly.graph_objects as go

from minimal_sim import (
    load_build_catalog_strict,
    load_visitors_strict,
    new_state,
    run_week,
    show_rooms,
    render_tower_image
)

# Sidebar: load data & set parameters
st.sidebar.header("Setup")
build_csv   = st.sidebar.text_input("Build CSV path", "build library.csv")
vis_csv     = st.sidebar.text_input("Visitor CSV path", "visitor_table_basic.csv")
start_yen   = st.sidebar.number_input("Starting Yen", min_value=0, value=300)
refuse      = st.sidebar.checkbox("Refuse overshoot", value=False)

if "build" not in st.session_state:
    st.session_state.build = load_build_catalog_strict(build_csv)
    st.session_state.vis   = load_visitors_strict(vis_csv)
    st.session_state.state = new_state(start_yen)

st.title("🏢 Tower Builder Sim")
s = st.session_state.state

# Build choice
choice = st.text_input("Build (ID or name)", "")
build = st.session_state.build
vis   = st.session_state.vis
state = st.session_state.state

# Step button
if st.button("Run one week"):
    try:
        out = run_week(
            state, build, vis,
            build_choice = choice or None,
            refuse_overshoot = refuse,
        )
        st.session_state.state = state
        st.subheader("This Week’s Summary")
        st.json(out["summary"])

        st.subheader("Assignments")
        st.dataframe(pd.DataFrame(out["assignments"]))

        st.subheader("Tower")
        st.image(render_tower_image(st.session_state.state), use_container_width=False)
    except ValueError as e:
        st.warning(str(e))
        # (optional) show what *is* affordable
        affordable = build[build["cost"] <= state["yen"]][["build_id","room_name","cost"]]
        st.info("Affordable right now:")
        st.dataframe(affordable)
# Show current totals
st.subheader("Totals")
s = st.session_state.state
log_df = pd.DataFrame(s.get("log", []))

# ---- BIG NUMBERS ----
avg_s = (sum(s["satisfaction_history"]) / len(s["satisfaction_history"])) if s["satisfaction_history"] else 0.0
c1, c2, c3, c4, c5, c6 = st.columns(6)
c1.metric("Week", int(s["week"]))
c2.metric("Yen", int(s["yen"]))
c3.metric("Population", round(s["population"], 1))
c4.metric("Influence",  round(s["influence"], 1))
c5.metric("Interest",   round(s["interest"], 1))
c6.metric("Avg Satisfaction", f"{avg_s:.0%}")

st.markdown("---")

# # ---- BAR CHART: totals vs this-week gains ----
# mode = st.radio("Bar chart:", ["Totals (since start)", "This week gains"], horizontal=True)
# if mode.startswith("Totals"):
#     bar_df = pd.DataFrame({
#         "Track": ["Population", "Influence", "Interest"],
#         "Value": [s["population"], s["influence"], s["interest"]],
#     }).set_index("Track")
# else:
#     # last row's tracks_gain dict -> DataFrame
#     if not log_df.empty and "tracks_gain" in log_df.columns and pd.notna(log_df["tracks_gain"].iloc[-1]):
#         gains = log_df["tracks_gain"].iloc[-1]
#     else:
#         gains = {"Population": 0, "Influence": 0, "Interest": 0}
#     bar_df = pd.DataFrame(list(gains.items()), columns=["Track","Value"]).set_index("Track")

# st.bar_chart(bar_df)

# st.markdown("---")

# # ---- GAUGE / METER for satisfaction ----
# fig = go.Figure(go.Indicator(
#     mode="gauge+number",
#     value=avg_s * 100.0,
#     number={"suffix": "%"},
#     title={"text": "Avg Satisfaction"},
#     gauge={
#         "axis": {"range": [0, 100]},
#         # Optional: simple zones (remove if you prefer default look)
#         "steps": [
#             {"range": [0, 50],  "color": "#f3d6d6"},
#             {"range": [50, 80], "color": "#f6efc3"},
#             {"range": [80, 100],"color": "#d7f0de"},
#         ],
#     },
# ))
# st.plotly_chart(fig, use_container_width=True)

# st.write({
#     "Week": s["week"],
#     "Yen": s["yen"],
#     "Population": s["population"],
#     "Influence": s["influence"],
#     "Interest": s["interest"],
#     "Avg Satisfaction": (sum(s["satisfaction_history"])/len(s["satisfaction_history"])) if s["satisfaction_history"] else 0
# })

st.sidebar.markdown("---")
if st.sidebar.button("Reset"):
    del st.session_state.build
    del st.session_state.vis
    del st.session_state.state
    st.experimental_rerun()
