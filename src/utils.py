import numpy as np
import pandas as pd
import polars as pl
from sentence_transformers import SentenceTransformer
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import (
    FunctionTransformer,
    OneHotEncoder,
    StandardScaler,
    TargetEncoder,
)

NUMERICAL_FEATURES = [
    "danceability",
    "energy",
    "loudness",
    "mode",
    "speechiness",
    "acousticness",
    "instrumentalness",
    "liveness",
    "valence",
    "tempo",
    "duration_ms",
    "time_signature",
]
CATEGORICAL_FEATURES = ["genre", "key"]
YEAR_COL = ["year"]


def generate_and_save_embeddings(
    input_csv,
    output_parquet,
    model_name="all-MiniLM-L6-v2",
    device="cpu",
    batch_size=128,
):
    # Track: {track_name} | Artist: {artist_name} | Genre: {genre}
    # Track: Hello | Artist: Adele | Genre: Pop
    df = (
        pl.scan_csv(input_csv)
        .select(
            pl.col("track_id"),
            pl.concat_str(
                [
                    pl.lit("Track: "),
                    pl.col("track_name"),
                    pl.lit(" | Artist: "),
                    pl.col("artist_name"),
                    pl.lit(" | Genre: "),
                    pl.col("genre"),
                ]
            ).alias("text"),
        )
        .collect()
    )
    texts = df["text"].to_list()

    model = SentenceTransformer(model_name, device=device)
    embeddings = model.encode(texts, batch_size=batch_size, show_progress_bar=True)

    pl.DataFrame({"track_id": df["track_id"], "embedding": embeddings}).write_parquet(
        output_parquet
    )


def _get_age_pipeline():
    def calculate_age(year_data):
        return 2023 - year_data

    return Pipeline(
        [
            ("imputer", SimpleImputer(strategy="median")),
            ("age_calc", FunctionTransformer(calculate_age, validate=True)),
            ("scaler", StandardScaler()),
        ]
    )


def _get_categorical_pipeline(encoding_type, random_state, sparse_output):
    steps = [("imputer", SimpleImputer(strategy="most_frequent"))]

    if encoding_type == "one_hot":
        steps.append(
            (
                "onehot",
                OneHotEncoder(
                    handle_unknown="ignore",
                    sparse_output=sparse_output,
                    dtype=np.float32,
                ),
            )
        )
    elif encoding_type == "target_encoding":
        steps.append(("target_encode", TargetEncoder(random_state=random_state)))

    return Pipeline(steps)


def _get_embedding_pipeline():
    def extract_embedding(data):
        return np.vstack(data["embedding"].to_numpy())

    return Pipeline(
        [
            ("extract", FunctionTransformer(extract_embedding, validate=False)),
            ("scaler", StandardScaler()),
        ]
    )


def load_and_preprocess_data(
    file,
    test_size=0.2,
    problem_type="regression",
    use_categorical=True,
    use_categorical_embeddings=False,
    embeddings_path="",
    categorical_encoding="one_hot",
    popularity_threshold=50,
    random_state=42,
    sample_size=0.1,
    one_hot_sparse=False,
    stratify_by=None,
):
    """Load and preprocess the dataset for model training and evaluation.
    Args:
        file (str or pl.DataFrame): Path to the CSV file or a Polars DataFrame.
        test_size (float): Proportion of the dataset to include in the test split.
        problem_type (str): Type of problem - "regression" or "classification".
        use_categorical (bool): Whether to include categorical features.
        use_categorical_embeddings (bool): Whether to use embeddings for categorical features.
        embeddings_path (str): Path to the Parquet file containing embeddings.
        categorical_encoding (str): Encoding method for categorical features - "one_hot" or "target_encoding".
        popularity_threshold (int or list): Threshold(s) for classification tasks.
        random_state (int): Random seed for reproducibility.
        sample_size (float): Fraction of the dataset to sample for processing.
        one_hot_sparse (bool): Whether to use sparse output for one-hot encoding.
        stratify_by (str or None): Column name to use for stratification during train-test split.
    Returns:
        X_train (np.ndarray): Preprocessed training features.
        X_test (np.ndarray): Preprocessed testing features.
        y_train (np.ndarray): Training labels.
        y_test (np.ndarray): Testing labels.
        feature_names (list): List of feature names after preprocessing.
    """
    if isinstance(file, pl.DataFrame):
        df = file
    else:
        df = pl.read_csv(file)

    df = df.with_columns(pl.selectors.float().cast(pl.Float32))
    df = df.with_columns((pl.col("popularity") / 100).alias("popularity"))

    if sample_size < 1.0:
        df = df.sample(fraction=sample_size, seed=random_state)

    if use_categorical and use_categorical_embeddings:
        embeddings = (
            pl.scan_parquet(embeddings_path)
            .filter(pl.col("track_id").is_in(df["track_id"]))
            .collect()
        )
        df = df.join(embeddings, on="track_id", how="inner")

    transformers = [
        ("track_age", _get_age_pipeline(), YEAR_COL),
        (
            "num",
            Pipeline(
                [
                    ("imputer", SimpleImputer(strategy="median")),
                    ("scaler", StandardScaler()),
                ]
            ),
            NUMERICAL_FEATURES,
        ),
    ]

    if use_categorical:
        if use_categorical_embeddings:
            transformers.append(
                ("embeddings", _get_embedding_pipeline(), ["embedding"])
            )
        else:
            cat_pipe = _get_categorical_pipeline(
                categorical_encoding, random_state, one_hot_sparse
            )
            transformers.append(("cat", cat_pipe, CATEGORICAL_FEATURES))

    preprocessor = ColumnTransformer(transformers=transformers, remainder="drop")

    drop_cols = ["track_id", "artist_name", "track_name", "popularity"]
    if use_categorical and use_categorical_embeddings:
        drop_cols.append("genre")

    X = df.drop([c for c in drop_cols if c in df.columns]).to_pandas()
    y_series = df["popularity"].to_pandas()

    if use_categorical and not use_categorical_embeddings:
        for col in [c for c in CATEGORICAL_FEATURES if c in X.columns]:
            X[col] = X[col].astype("category")

    if problem_type == "classification":
        if isinstance(popularity_threshold, list):
            bins = [-0.01] + sorted([th / 100 for th in popularity_threshold]) + [1.01]
            y = pd.cut(y_series, bins=bins, labels=False).astype(int)
        else:
            y = (y_series >= (popularity_threshold / 100)).astype(int)
    else:
        y = y_series

    stratify = X[stratify_by] if stratify_by and stratify_by in X.columns else None

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=test_size, random_state=random_state, stratify=stratify
    )

    X_train_processed = preprocessor.fit_transform(X_train, y_train)
    X_test_processed = preprocessor.transform(X_test)

    feature_names = ["track_age"] + NUMERICAL_FEATURES

    if use_categorical:
        if use_categorical_embeddings:
            # Infer embedding dimension from output shape
            n_embed = X_train_processed.shape[1] - len(feature_names)
            feature_names.extend([f"embedding_{i}" for i in range(n_embed)])
        elif "cat" in preprocessor.named_transformers_:
            cat_tr = preprocessor.named_transformers_["cat"]
            if categorical_encoding == "one_hot" and hasattr(
                cat_tr.named_steps["onehot"], "get_feature_names_out"
            ):
                feature_names.extend(
                    cat_tr.named_steps["onehot"].get_feature_names_out(
                        CATEGORICAL_FEATURES
                    )
                )

    return X_train_processed, X_test_processed, y_train, y_test, feature_names
