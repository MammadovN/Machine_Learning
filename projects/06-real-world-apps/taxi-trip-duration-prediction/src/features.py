import polars as pl

def build_features(df: pl.DataFrame) -> pl.DataFrame:
    df = df.with_columns(
        pl.col("pickup_datetime").str.strptime(pl.Datetime).alias("pickup_dt")
    )
    return (
        df.with_columns([
            pl.col("pickup_dt").dt.hour().alias("pickup_hour"),
            pl.col("pickup_dt").dt.weekday().alias("pickup_wday"),
            pl.col("pickup_dt").dt.month().alias("pickup_month"),
            (
                (pl.col("pickup_longitude") - pl.col("dropoff_longitude")).abs()
              + (pl.col("pickup_latitude")  - pl.col("dropoff_latitude")).abs()
            ).alias("manhattan_dist")
        ])
        .select([
            "vendor_id", "passenger_count",
            "pickup_hour", "pickup_wday", "pickup_month",
            "manhattan_dist",
            "trip_duration"
        ])
    )
