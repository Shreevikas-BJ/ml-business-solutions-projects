import os
os.environ["JAVA_HOME"] = r"C:\Program Files\Java\jdk-17"
os.environ["PATH"] = os.environ["JAVA_HOME"] + r"\bin;" + os.environ["PATH"]

from pyspark.sql import SparkSession, functions as F

RAW_PATH = "data/raw/superstore/sales.csv"
OUT_FILE = "data/sample_outputs/daily_sales_sample.csv"  # single file (Windows-safe)

def main():
    spark = (
        SparkSession.builder
        .appName("01_ingest_clean")
        .master("local[*]")
        .getOrCreate()
    )
    spark.sparkContext.setLogLevel("WARN")

    print(f"Reading: {RAW_PATH}")
    df = spark.read.option("header", True).option("inferSchema", True).csv(RAW_PATH)

    # ✅ Fix 1: trim column names (handles 'Sales ' etc.)
    df = df.toDF(*[c.strip() for c in df.columns])

    # quick peek at raw date strings
    print("Sample Order Date values:")
    df.select(F.col("Order Date")).show(10, truncate=False)

    print("Columns:", df.columns)
    print("Row count:", df.count())

    date_col = "Order Date"
    sales_col = "Sales"

    # ✅ Fix 2: parse date explicitly (dd/MM/yyyy) + trim
    df2 = (
        df
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

    print("Daily rows:", daily.count())
    daily.show(10, truncate=False)

    # ✅ Windows-safe local export for sanity (avoids winutils/HADOOP_HOME issues)
    os.makedirs(os.path.dirname(OUT_FILE), exist_ok=True)
    daily.limit(200).toPandas().to_csv(OUT_FILE, index=False)
    print(f"Wrote sample file: {OUT_FILE}")
    print("✅ ETL Step 01 complete.")

    spark.stop()

if __name__ == "__main__":
    main()
