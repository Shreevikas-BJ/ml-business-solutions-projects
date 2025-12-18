import pandas as pd
from pyspark.sql import SparkSession, functions as F, Window
from src.common.snowflake import exec_sql, write_df

RAW_PATH = "data/raw/superstore/sales.csv"

def build_daily(df, date_col="Order Date", sales_col="Sales"):
    df2 = (
        df
        # ✅ Superstore dates look like 08-11-2017 -> dd-MM-yyyy
        .withColumn("DS", F.to_date(F.trim(F.col(date_col)), "dd-MM-yyyy"))
        .withColumn("Y", F.col(sales_col).cast("double"))
        .where(F.col("DS").isNotNull())
        .where(F.col("Y").isNotNull())
        .withColumn("SERIES_ID", F.lit("GLOBAL"))
    )

    daily = (
        df2.groupBy("SERIES_ID", "DS")
        .agg(F.sum("Y").alias("Y"))
        .orderBy("DS")
    )
    return daily

def add_features(daily):
    # Window by series ordered by date
    w = Window.partitionBy("SERIES_ID").orderBy(F.col("DS"))

    # Rolling windows should NOT peek into the future:
    w7 = w.rowsBetween(-7, -1)
    w28 = w.rowsBetween(-28, -1)

    feat = (
        daily
        .withColumn("LAG_1", F.lag("Y", 1).over(w))
        .withColumn("LAG_7", F.lag("Y", 7).over(w))
        .withColumn("ROLL7_MEAN", F.avg("Y").over(w7))
        .withColumn("ROLL28_MEAN", F.avg("Y").over(w28))
        .withColumn("DOW", F.dayofweek("DS"))  # 1=Sun ... 7=Sat
        .withColumn("IS_WEEKEND", F.col("DOW").isin([1, 7]))
        .withColumn("UPDATED_AT", F.current_timestamp())
    )

    # Drop early rows that don't have enough history (basic & clean)
    feat = feat.where(F.col("LAG_7").isNotNull() & F.col("ROLL7_MEAN").isNotNull())
    return feat

def main():
    spark = (
        SparkSession.builder
        .appName("02_feature_engineering")
        .master("local[*]")
        .getOrCreate()
    )
    spark.sparkContext.setLogLevel("WARN")

    print(f"Reading: {RAW_PATH}")
    df = spark.read.option("header", True).option("inferSchema", True).csv(RAW_PATH)

    # ✅ Trim column names (handles 'Sales ' etc.)
    df = df.toDF(*[c.strip() for c in df.columns])

    # If your CSV uses different column names, update these:
    date_col = "Order Date"
    sales_col = "Sales"

    daily = build_daily(df, date_col=date_col, sales_col=sales_col)
    print("Daily rows:", daily.count())
    daily.show(5, truncate=False)

    feat = add_features(daily)
    print("Feature rows:", feat.count())
    feat.show(5, truncate=False)

    # ✅ Convert DS/UPDATED_AT to strings before toPandas (avoids datetime64 precision error)
    pdf = feat.select(
        "SERIES_ID",
        F.date_format("DS", "yyyy-MM-dd").alias("DS"),
        "Y", "LAG_1", "LAG_7", "ROLL7_MEAN", "ROLL28_MEAN",
        "DOW", "IS_WEEKEND",
        F.date_format("UPDATED_AT", "yyyy-MM-dd HH:mm:ss").alias("UPDATED_AT")
    ).toPandas()

    # ---------- Incremental upsert via MERGE ----------
    exec_sql("CREATE TABLE IF NOT EXISTS FEATURES_STG LIKE FEATURES;")
    exec_sql("TRUNCATE TABLE FEATURES_STG;")

    success, nchunks, nrows, _ = write_df(pdf, "FEATURES_STG")
    print(f"Snowflake write_pandas -> success={success}, chunks={nchunks}, rows={nrows}")

    merge_sql = """
    MERGE INTO FEATURES t
    USING FEATURES_STG s
    ON t.SERIES_ID = s.SERIES_ID AND t.DS = TO_DATE(s.DS)
    WHEN MATCHED THEN UPDATE SET
      t.Y = s.Y,
      t.LAG_1 = s.LAG_1,
      t.LAG_7 = s.LAG_7,
      t.ROLL7_MEAN = s.ROLL7_MEAN,
      t.ROLL28_MEAN = s.ROLL28_MEAN,
      t.DOW = s.DOW,
      t.IS_WEEKEND = s.IS_WEEKEND,
      t.UPDATED_AT = TO_TIMESTAMP_NTZ(s.UPDATED_AT)
    WHEN NOT MATCHED THEN INSERT
      (SERIES_ID, DS, Y, LAG_1, LAG_7, ROLL7_MEAN, ROLL28_MEAN, DOW, IS_WEEKEND, UPDATED_AT)
    VALUES
      (s.SERIES_ID, TO_DATE(s.DS), s.Y, s.LAG_1, s.LAG_7, s.ROLL7_MEAN, s.ROLL28_MEAN, s.DOW, s.IS_WEEKEND, TO_TIMESTAMP_NTZ(s.UPDATED_AT));
    """
    exec_sql(merge_sql)
    print("✅ MERGE complete: FEATURES updated.")

    print("✅ Feature Engineering Step 02 complete.")
    spark.stop()

if __name__ == "__main__":
    main()
