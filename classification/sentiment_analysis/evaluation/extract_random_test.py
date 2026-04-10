from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parents[2]
CORPUS_PATH = ROOT / "data" / "mcu_imdb_reviews.csv"
EXISTING_EVAL_PATH = ROOT / "classification" / "evaluation" / "evaluation_dataset.xlsx"
OUTPUT_PATH = ROOT / "classification" / "evaluation" / "random_test_set_300.xlsx"

SAMPLE_SIZE = 300
RANDOM_SEED = 99
RATING_MIN = 1
RATING_MAX = 10


def main() -> None:
    corpus_df = pd.read_csv(CORPUS_PATH)
    existing_eval_df = pd.read_excel(EXISTING_EVAL_PATH)

    existing_review_ids = set(existing_eval_df["review_id"].dropna().astype(str))
    corpus_df["review_id_str"] = corpus_df["review_id"].astype(str)
    corpus_df["author_rating"] = pd.to_numeric(corpus_df["author_rating"], errors="coerce")
    corpus_df["rating_int"] = corpus_df["author_rating"].round().astype("Int64")

    candidate_df = corpus_df.loc[~corpus_df["review_id_str"].isin(existing_review_ids)].copy()
    candidate_df = candidate_df.loc[candidate_df["author_rating"].notna()].copy()

    if len(candidate_df) < SAMPLE_SIZE:
        raise ValueError(
            f"Only {len(candidate_df)} non-overlapping reviews available, "
            f"but SAMPLE_SIZE is {SAMPLE_SIZE}."
        )

    candidate_df = candidate_df.loc[
        candidate_df["rating_int"].between(RATING_MIN, RATING_MAX)
    ].copy()

    ratings = list(range(RATING_MIN, RATING_MAX + 1))
    n_groups = len(ratings)
    n_each = SAMPLE_SIZE // n_groups

    if SAMPLE_SIZE % n_groups != 0:
        raise ValueError(
            f"SAMPLE_SIZE={SAMPLE_SIZE} must be divisible by {n_groups} "
            "for equal-per-rating sampling."
        )

    sampled_parts = []
    for i, rating in enumerate(ratings):
        pool = candidate_df.loc[candidate_df["rating_int"] == rating]
        if len(pool) < n_each:
            raise ValueError(
                f"Not enough reviews for rating {rating}: need {n_each}, got {len(pool)}."
            )
        sampled_parts.append(pool.sample(n=n_each, random_state=RANDOM_SEED + i))

    sample_df = pd.concat(sampled_parts, ignore_index=True)
    sample_df = sample_df.sample(frac=1, random_state=RANDOM_SEED + 99).reset_index(drop=True)

    output_df = sample_df[
        [
            "review_id",
            "movie_id",
            "movie_title",
            "author_rating",
            "summary",
            "content",
        ]
    ].reset_index(drop=True)

    output_df["ground_truth"] = ""

    output_df.to_excel(OUTPUT_PATH, index=False)

    print(f"Wrote {len(output_df)} rows to: {OUTPUT_PATH}")
    print(sample_df["rating_int"].value_counts().sort_index().to_string())


if __name__ == "__main__":
    main()
