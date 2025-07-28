import streamlit as st
import pandas as pd
from minimal_sim import (
    load_build_catalog_strict,
    load_visitors_strict,
    new_state,
    run_week,
    show_rooms,
)

# Sidebar: load data & set parameters
st.sidebar.header("Setup")
build_csv   = st.sidebar.text_input("Build CSV path", "build library.csv")
vis_csv     = st.sidebar.text_input("Visitor CSV path", "visitor_table_basic.csv")
start_yen   = st.sidebar.number_input("Starting Yen", min_value=0, value=300)
draws       = st.sidebar.number_input("Draws per week", min_value=1, value=1)
refuse      = st.sidebar.checkbox("Refuse overshoot", value=False)

if "build" not in st.session_state:
    st.session_state.build = load_build_catalog_strict(build_csv)
    st.session_state.vis   = load_visitors_strict(vis_csv)
    st.session_state.state = new_state(start_yen)

st.title("🏢 Tower Builder Sim")

# Show current totals
st.subheader("Totals")
s = st.session_state.state
st.write({
    "Week": s["week"],
    "Yen": s["yen"],
    "Population": s["population"],
    "Influence": s["influence"],
    "Interest": s["interest"],
    "Avg Satisfaction": (sum(s["satisfaction_history"])/len(s["satisfaction_history"])) if s["satisfaction_history"] else 0
})

# Build choice
choice = st.text_input("Build (ID or name)", "")
build = st.session_state.build
vis   = st.session_state.vis
state = st.session_state.state

# Step button
if st.button("Run one week"):
    out = run_week(
        state, build, vis,
        build_choice = choice or None,
        draws       = draws,
        refuse_overshoot = refuse,
    )
    st.session_state.state = state
    st.subheader("This Week’s Summary")
    st.json(out["summary"])

    st.subheader("Assignments")
    st.dataframe(pd.DataFrame(out["assignments"]))

    st.subheader("Tower Rooms")
    df_rooms = pd.DataFrame(s["rooms"])
    st.dataframe(df_rooms[["floor","room_name","level","slots_remaining","active"]])

st.sidebar.markdown("---")
if st.sidebar.button("Reset"):
    del st.session_state.build
    del st.session_state.vis
    del st.session_state.state
    st.experimental_rerun()