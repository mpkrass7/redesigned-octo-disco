import numpy as np
import pandas as pd
import streamlit as st

import helper_functions as hf
from helper_functions import column_type_mapping, base_fields, data_dictionary

st.set_page_config(
    page_title="Rolling Stone Top 500 Analyzer",
    layout="wide",
    page_icon=hf.open_favicon(),
)

hide_streamlit_style = """
    <style>
    footer {visibility: hidden;}
    </style>
    """


def layout_application():

    title = st.container().empty()
    row1 = st.container().empty()
    dropdown, _ = row1.columns([4, 2])
    scatters = st.container().empty()
    scatter_1, scatter_2 = scatters.columns([2, 2])
    cluster_box = st.container().empty()
    table = st.container().empty()
    return (
        title,
        row1,
        dropdown,
        scatters,
        scatter_1,
        scatter_2,
        cluster_box,
        table,
    )


@st.cache(allow_output_mutation=True)
def read_data():

    df = pd.read_csv("data/spotify_music_output.csv").fillna("")
    artists = sorted(df["artist_name"].unique())
    return df, artists


def _main():
    (
        title,
        row1,
        dropdown,
        scatters,
        scatter_1,
        scatter_2,
        cluster_box,
        table,
    ) = layout_application()

    params = st.experimental_get_query_params()
    if "debug" not in params:
        st.markdown(hide_streamlit_style, unsafe_allow_html=True)

    df, artists = read_data()
    with title:
        st.title("Rolling Stone Top 500 Investigator")

    artists_to_highlight = dropdown.multiselect(
        "Select artists to highlight",
        artists,
    )

    hf.plot_rank_vs_song_popularity(scatter_1, df, artists_to_highlight)
    hf.plot_rank_vs_artist_popularity(scatter_2, df, artists_to_highlight)

    if artists_to_highlight:
        table.write(df.loc[lambda x: x["artist_name"].isin(artists_to_highlight)])
    else:
        table.write(df)

    with st.sidebar.form(key="my_form"):
        st.write("Select Dimensionality Reduction and Clustering Parameters")

        plot_3d = st.radio("Plot Dimension", ["2D", "3D"])

        pressed = st.form_submit_button("Project Data")

        with st.expander("Advanced Options"):
            seed = st.number_input(
                "Random Seed",
                value=42,
            )
            use_svd = st.checkbox("Use SVD?")
            features = st.multiselect(
                "Select Relevant Features",
                list(column_type_mapping.keys()),
                default=base_fields,
            )
        with st.expander("What is this?"):
            st.markdown(data_dictionary, unsafe_allow_html=True)

    if pressed:
        pipe = hf.build_pipeline(features, use_svd=use_svd, seed=seed)
        print("Running Pipeline")
        dimensions = int(plot_3d[0])
        cluster_box.info(f"Projecting results onto {dimensions}D plane..")
        try:
            umap_embedding, clusters = hf.run_pipeline(
                df, pipe, {}, n_components=dimensions, seed=seed
            )
            hf.plot_umap_results(
                cluster_box,
                df,
                umap_embedding,
                clusters,
                artists=artists_to_highlight,
                is_3d=dimensions == 3,
            )
        except TypeError:
            cluster_box.error(
                "Error running pipeline. Change the seed in advanced options and try again."
            )


if __name__ == "__main__":
    _main()
