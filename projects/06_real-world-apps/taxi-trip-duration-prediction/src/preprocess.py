import polars as pl

def clean_data(df: pl.DataFrame) -> pl.DataFrame:
    """Remove obvious outliers & nulls."""
    return (
        df.filter(
            (pl.col("trip_duration").is_between(60, 7200))
            & (pl.col("passenger_count") > 0)
        )
        .drop_nulls()
    )
