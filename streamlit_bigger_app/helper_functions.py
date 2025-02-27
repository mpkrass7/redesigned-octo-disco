import os

import numpy as np
import pandas as pd
from PIL import Image
import plotly.express as px
import plotly.graph_objects as go
from sklearn.cluster import DBSCAN
from sklearn.compose import ColumnTransformer
from sklearn.decomposition import TruncatedSVD
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.preprocessing import MinMaxScaler, OneHotEncoder, FunctionTransformer
from sklearn.pipeline import Pipeline
import umap

column_type_mapping = {
    "track_danceability": "numeric",
    "track_energy": "numeric",
    "track_loudness": "numeric",
    "track_speechiness": "numeric",
    "track_acousticness": "numeric",
    "track_instrumentalness": "numeric",
    "track_liveness": "numeric",
    "track_valence": "numeric",
    "track_tempo": "numeric",
    "track_duration_ms": "numeric",
    "track_popularity": "numeric",
    "track_is_explicit": "categorical",
    "artist_popularity": "numeric",
    "track_time_signature": "categorical",
    "track_key": "categorical",
    "album_release_year": "numeric",
    "artist_genre": "text",
}

base_fields = [
    i
    for i in column_type_mapping.keys()
    if column_type_mapping[i] == "numeric"
    and "track" in i
    and "duration" not in i
    and "popularity" not in i
]

data_dictionary = """
Pressing the run button allows you to project various features of the songs onto a lower dimensional space in order to visually compare music tracks. 

<em>Select Relevant Features</em> let's you choose the features you want to use for UMAP and Clustering. Some column meanings are as follows:

<strong>acousticness:</strong> A confidence measure from 0.0 to 1.0 of whether the track is acoustic. 1.0 represents high confidence the track is acoustic.

<strong>danceability:</strong> Danceability describes how suitable a track is for dancing based on a combination of musical elements including tempo, rhythm stability, beat strength, and overall regularity. A value of 0.0 is least danceable and 1.0 is most danceable.

<strong>energy:</strong> Energy is a measure from 0.0 to 1.0 and represents a perceptual measure of intensity and activity. Typically, energetic tracks feel fast, loud, and noisy. For example, death metal has high energy, while a Bach prelude scores low on the scale. Perceptual features contributing to this attribute include dynamic range, perceived loudness, timbre, onset rate, and general entropy.

<strong>instrumentalness:</strong> Predicts whether a track contains no vocals. "Ooh" and "aah" sounds are treated as instrumental in this context. Rap or spoken word tracks are clearly "vocal". The closer the instrumentalness value is to 1.0, the greater likelihood the track contains no vocal content. Values above 0.5 are intended to represent instrumental tracks, but confidence is higher as the value approaches 1.0.

<strong>key:</strong> The key the track is in. Integers map to pitches using standard Pitch Class notation. E.g. 0 = C, 1 = C♯/D♭, 2 = D, and so on. If no key was detected, the value is -1. >= -1, <= 11

<strong>liveness:</strong> Detects the presence of an audience in the recording. Higher liveness values represent an increased probability that the track was performed live. A value above 0.8 provides strong likelihood that the track is live.

<strong>loudness:</strong> The overall loudness of a track in decibels (dB). Loudness values are averaged across the entire track and are useful for comparing relative loudness of tracks. Loudness is the quality of a sound that is the primary psychological correlate of physical strength (amplitude). Values typically range between -60 and 0 db.

<strong>speechiness:</strong> Speechiness detects the presence of spoken words in a track. The more exclusively speech-like the recording (e.g. talk show, audio book, poetry), the closer to 1.0 the attribute value. Values above 0.66 describe tracks that are probably made entirely of spoken words. Values between 0.33 and 0.66 describe tracks that may contain both music and speech, either in sections or layered, including such cases as rap music. Values below 0.33 most likely represent music and other non-speech-like tracks.

<strong>tempo:</strong> The overall estimated tempo of a track in beats per minute (BPM). In musical terminology, tempo is the speed or pace of a given piece and derives directly from the average beat duration.

<strong>time_signature:</strong> An estimated time signature. The time signature (meter) is a notational convention to specify how many beats are in each bar (or measure). The time signature ranges from 3 to 7 indicating time signatures of "3/4", to "7/4". >= 3, <= 7

<strong>valence:</strong> A measure from 0.0 to 1.0 describing the musical positiveness conveyed by a track. Tracks with high valence sound more positive (e.g. happy, cheerful, euphoric), while tracks with low valence sound more negative (e.g. sad, depressed, angry).

"""


# Modified TruncatedSVD that doesn't fail if n_components > ncols
class MyTruncatedSVD(TruncatedSVD):
    def fit_transform(self, X, y=None):
        if X.shape[1] <= self.n_components:
            self.n_components = X.shape[1] - 1
        return TruncatedSVD.fit_transform(self, X=X, y=y)


def open_favicon():
    """Open the favicon"""
    return Image.open(os.path.join(os.path.dirname(__file__), "images/favicon.ico"))


def to_string(x):
    """Handle values as string.  They may be treated as booleans or numerics otherwise"""
    return x.astype(str)


def generate_hover_label(df, level="track"):
    if level == "track":
        return [
            f"Artist Name: {artist_name} </br> Track Name: {track_name} </br> Track Ranking: {rank} </br> Track Popularity: {popularity}"
            for artist_name, track_name, rank, popularity in zip(
                df.artist_name, df.track_name, df["rank"], df.track_popularity
            )
        ]
    else:
        return [
            f"Artist Name: {artist_name} </br> Best Ranking: {rank} </br> Artist Popularity: {popularity}"
            for artist_name, rank, popularity in zip(
                df.artist_name, df["rank"], df["artist_popularity"]
            )
        ]


def plot_rank_vs_song_popularity(container, df, artists=[]):
    """Plot the rank vs popularity of the songs in the dataframe

    Parameters
    ----------
    df : pd.DataFrame
        The rolling stone top 500 data
    artists : list
        a list of artists chosen by the select dropdown
    """

    df_selected_artist = df.loc[lambda x: x.artist_name.isin(artists)]
    df_not_selected_artist = df.loc[lambda x: ~x.artist_name.isin(artists)]
    hovertemplate = "</br>%{text}<extra></extra>"
    fig = go.Figure(
        go.Scatter(
            mode="markers",
            x=df_not_selected_artist["rank"],
            y=df_not_selected_artist.track_popularity,
            marker_size=8,
            text=generate_hover_label(df_not_selected_artist, level="track"),
            hovertemplate=hovertemplate,
            marker_color="#1f77b4",
            marker_symbol="circle",
            marker_line_width=1,
            opacity=0.7,
        )
    )
    fig = fig.add_trace(
        go.Scatter(
            mode="markers",
            x=df_selected_artist["rank"],
            y=df_selected_artist.track_popularity,
            marker_size=15,
            text=generate_hover_label(df_selected_artist, level="track"),
            marker_color="lightskyblue",
            marker_symbol="star",
            marker_line_width=1.5,
            opacity=1,
            showlegend=False,
        )
    )

    fig.update_traces(showlegend=False)
    fig.update_coloraxes(showscale=False)
    fig.update_xaxes(autorange="reversed")
    fig = fig.update_layout(
        title="Track Ranking against Track Popularity",
        legend=None,
        plot_bgcolor="white",
        margin=dict(t=50, l=10, b=10, r=10),
        xaxis_title="Track Ranking",
        yaxis_title="Track Current Popularity",
        hoverlabel=dict(
            bgcolor="white", font_size=16, font_family="Rockwell", namelength=-1
        ),
    )

    config = {"displayModeBar": False}
    return container.plotly_chart(fig, config=config, use_container_width=True)


def plot_rank_vs_artist_popularity(container, df, artists=[]):
    """Plot the best song rank vs artist overall popularity

    Parameters
    ----------
    df : pd.DataFrame
        The rolling stone top 500 data
    artists : list
        a list of artists chosen by the select dropdown
    """

    df_artist = (
        df.groupby("artist_name")
        .agg({"rank": "min", "artist_popularity": "max"})
        .reset_index()
        .sort_values(by="artist_popularity", ascending=False)
    )
    df_selected_artist = df_artist.loc[lambda x: x.artist_name.isin(artists)]
    df_not_selected_artist = df_artist.loc[lambda x: ~x.artist_name.isin(artists)]

    hovertemplate = "</br>%{text}<extra></extra>"
    fig = go.Figure(
        go.Scatter(
            mode="markers",
            x=df_not_selected_artist["rank"],
            y=df_not_selected_artist.artist_popularity,
            marker_size=8,
            text=generate_hover_label(df_not_selected_artist, level="artist"),
            hovertemplate=hovertemplate,
            marker_color="#1f77b4",
            marker_symbol="circle",
            marker_line_width=1,
            opacity=0.7,
        )
    )
    fig = fig.add_trace(
        go.Scatter(
            mode="markers",
            x=df_selected_artist["rank"],
            y=df_selected_artist.artist_popularity,
            marker_size=15,
            text=generate_hover_label(df_selected_artist, level="artist"),
            hovertemplate=hovertemplate,
            marker_color="lightskyblue",
            marker_symbol="star",
            marker_line_width=1.5,
            opacity=1,
            showlegend=False,
        )
    )

    fig.update_traces(showlegend=False)
    fig.update_coloraxes(showscale=False)
    fig.update_xaxes(autorange="reversed")
    fig.update_layout(
        title="Artist's Best Track Ranking against Artist Popularity",
        plot_bgcolor="white",
        margin=dict(t=50, l=10, b=10, r=10),
        xaxis_title="Artist Best Ranking",
        yaxis_title="Artist Current Popularity",
        hoverlabel=dict(
            bgcolor="white", font_size=16, font_family="Rockwell", namelength=-1
        ),
    )
    fig.layout.update(showlegend=True)

    config = {"displayModeBar": False}
    return container.plotly_chart(fig, config=config, use_container_width=True)


def build_pipeline(features, use_svd=False, seed=42):
    """Build a clustering and dimensionality reduction Pipeline

    Returns
    -------
    sklearn.PipeLine
        a pipeline with steps to reduce the dimensionality of a dataset and run clustering on it
    """
    numeric_cols = [i for i in features if column_type_mapping[i] == "numeric"]
    categorical_cols = [i for i in features if column_type_mapping[i] == "categorical"]
    text_cols = [i for i in features if column_type_mapping[i] == "text"]

    numeric_pipeline = Pipeline(
        steps=[
            ("scaler", MinMaxScaler()),
        ]
    )

    # Handle Categorical Variables
    categorical_pipeline = Pipeline(
        steps=[
            ("convert_to_string", FunctionTransformer(to_string)),
            ("onehot", OneHotEncoder(categories="auto", handle_unknown="ignore")),
        ]
    )

    text_pipeline = Pipeline(
        steps=[
            ("ngrams", CountVectorizer(ngram_range=(1, 2))),
        ]
    )

    transformers = []
    if len(numeric_cols) > 0:
        transformers.append(("num", numeric_pipeline, numeric_cols))
    if len(categorical_cols) > 0:
        transformers.append(("cat", categorical_pipeline, categorical_cols))
    if len(text_cols) > 0:
        for col in text_cols:
            transformers.append(("text", text_pipeline, col))

    preprocessing_pipeline = ColumnTransformer(
        transformers=transformers,
    )

    steps = [("preprocessing", preprocessing_pipeline)]
    if use_svd:
        steps.append(
            (
                "svd",
                MyTruncatedSVD(random_state=seed),
            )
        )

    pipeline = Pipeline(
        steps=steps + [("umap", umap.UMAP(random_state=seed)), ("dbscan", DBSCAN())],
        verbose=False,
    )
    return pipeline


def run_pipeline(df, pipeline, parameters_dict, n_components=2, seed=42):
    np.random.seed(seed)
    umap_embedding = (
        pipeline[:-1].set_params(umap__n_components=n_components).fit_transform(df)
    )
    clusters = pipeline.set_params(**parameters_dict).fit_predict(df)

    return umap_embedding, clusters


def plot_umap_results(container, df, umap_embedding, clusters, artists=[], is_3d=False):
    """Plot the results of the clustering and dimensionality reduction Pipeline

    Parameters
    ----------
    df : pd.DataFrame
        The rolling stone top 500 data
    umap_embedding : np.array
        The umap embedding of the data
    clusters : np.array
        The clusters of the data
    artists : list
        a list of artists chosen by the select dropdown
    is_3d : bool
        whether to plot in 3d or not
    """
    artist_indexes = df.loc[lambda x: x.artist_name.isin(artists)].index

    text = [
        f"Artist Name: {artist_name} </br> Song Name: {track_name}"
        for artist_name, track_name in zip(df.artist_name, df.track_name)
    ]
    # Add in the listed bands from above
    filter_df = df.iloc[artist_indexes]

    text_records = [
        f"Artist Name: {artist_name} </br> Song Name: {track_name}"
        for artist_name, track_name in zip(filter_df.artist_name, filter_df.track_name)
    ]
    hovertemplate = "</br>%{text}<extra></extra>"

    fig = go.Figure()

    if is_3d:
        fig = go.Figure(
            data=[
                go.Scatter3d(
                    x=umap_embedding[:, 0],
                    y=umap_embedding[:, 1],
                    z=umap_embedding[:, 2],
                    mode="markers",
                    text=text,
                    marker_line_color="black",
                    marker_line_width=1,
                    marker=dict(
                        size=10,
                        color=clusters,  # set color to an array/list of desired values
                        opacity=0.75,
                    ),
                )
            ]
        )
        fig = fig.add_trace(
            go.Scatter3d(
                x=umap_embedding[artist_indexes, 0],
                y=umap_embedding[artist_indexes, 1],
                z=umap_embedding[artist_indexes, 2],
                text=text_records,
                showlegend=True,
                mode="markers",
                marker_symbol="diamond",
                marker_line_color="midnightblue",
                marker_color="lightskyblue",
                marker_line_width=1,
                marker=dict(size=12),
            )
        )
    else:
        fig = fig.add_trace(
            go.Scatter(
                x=umap_embedding[:, 0],
                y=umap_embedding[:, 1],
                hovertemplate=hovertemplate,
                text=text,
                showlegend=False,
                mode="markers",
                marker_color=clusters,
                marker_line_color="black",
                marker_line_width=1,
            )
        )

        fig = fig.add_trace(
            go.Scatter(
                x=umap_embedding[artist_indexes, 0],
                y=umap_embedding[artist_indexes, 1],
                text=text_records,
                showlegend=True,
                mode="markers",
                marker_symbol="star",
                marker_line_color="midnightblue",
                marker_color="lightskyblue",
                marker_line_width=1,
                marker=dict(size=15),
            )
        )
    dimensions = "3" if is_3d else "2"
    title = (
        f"UMAP Projection of Rolling Stone Top 500 Tracks in {dimensions} Dimensions"
    )
    fig = fig.update_layout(
        title_font_size=20,
        title=title,
        plot_bgcolor="#ffffff",
        hoverlabel=dict(
            bgcolor="white", font_size=16, font_family="Rockwell", namelength=-1
        ),
        margin=dict(l=20, r=20, t=30, b=20),
    ).update_yaxes(showgrid=False)
    return container.plotly_chart(fig, use_container_width=True)
