import numpy as np
import pandas as pd
import plotly.express as px
import streamlit as st


@st.cache(allow_output_mutation=True)
def read_data():

    return pd.read_csv("data/spotify_music_output.csv")


def plot_rank_vs_song_popularity(df, artists=[]):
    """Plot the rank vs popularity of the songs in the dataframe

    Parameters
    ----------
    df : pd.DataFrame
        The rolling stone top 500 data
    artists : list
        a list of artists chosen by the select dropdown
    """

    df["artist_indicator"] = np.where(df.artist_name.isin(artists), 1, 0)

    fig = px.scatter(
        data_frame=df,
        x="rank",
        y="track_popularity",
        color="artist_indicator",
        hover_name="track_name",
        title="Comparing Rank and Track Popularity",
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


def plot_rank_vs_artist_popularity(df, artists=[]):
    """Plot the rank vs popularity of the songs in the dataframe

    Parameters
    ----------
    df : _type_
        _description_
    artists : _type_
        _description_
    """

    return


data = read_data()
st.title("Rolling Stone Top 500 Explorer")
artists = st.multiselect("Select artists to highlight", data["artist_name"].unique())

plot_rank_vs_song_popularity(data, artists)
plot_rank_vs_artist_popularity(data, artists)
