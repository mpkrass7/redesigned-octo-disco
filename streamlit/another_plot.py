import numpy as np
import plotly.express as px
import streamlit as st


def plot_rank_vs_artist_popularity(df, artists=[]):
    """Plot the best song rank vs artist overall popularity

    Parameters
    ----------
    df : _type_
        _description_
    artists : _type_
        _description_
    """

    df_artist = (
        df.groupby("artist_name")
        .agg({"rank": "min", "artist_popularity": "max"})
        .reset_index()
        .sort_values(by="artist_popularity", ascending=False)
    )

    df_artist["artist_indicator"] = np.where(df_artist.artist_name.isin(artists), 1, 0)

    fig = px.scatter(
        data_frame=df_artist,
        x="rank",
        y="artist_popularity",
        color="artist_indicator",
        hover_name="artist_name",
        title="Comparing Best Rank and Artist Popularity",
        hover_data=["artist_name"],
        trendline="lowess",
    )
    fig.update_xaxes(autorange="reversed")
    fig = fig.update_layout(
        plot_bgcolor="white",
        margin=dict(t=50, l=10, b=10, r=10),
        xaxis_title="Song Ranking",
        yaxis_title="Current Song Popularity",
    )

    config = {"displayModeBar": False}
    return st.plotly_chart(fig, config=config, use_container_width=True)
