import streamlit as st
import pandas as pd
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
st.write({
    "Week": s["week"],
    "Yen": s["yen"],
    "Population": s["population"],
    "Influence": s["influence"],
    "Interest": s["interest"],
    "Avg Satisfaction": (sum(s["satisfaction_history"])/len(s["satisfaction_history"])) if s["satisfaction_history"] else 0
})

st.sidebar.markdown("---")
if st.sidebar.button("Reset"):
    del st.session_state.build
    del st.session_state.vis
    del st.session_state.state
    st.experimental_rerun()
