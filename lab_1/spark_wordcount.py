import time
from pyspark.sql import SparkSession
from pyspark.sql.functions import explode, split, col, lower

def main():
    spark = SparkSession.builder \
        .appName("WordCount-RDD-vs-DataFrame") \
        .getOrCreate()

    sc = spark.sparkContext
    sc.setLogLevel("WARN")

    input_path = "/workspace/input.txt"

    print("\n" + "="*50)
    print(">>> WORDCOUNT ЧЕРЕЗ RDD API <<<")
    print("="*50)
    start_rdd = time.time()

    text_rdd = sc.textFile(input_path)

    word_counts_rdd = text_rdd.flatMap(lambda line: line.strip().split()) \
                              .map(lambda word: (word.lower(), 1)) \
                              .reduceByKey(lambda a, b: a + b)

    top_words_rdd = word_counts_rdd.takeOrdered(10, key=lambda x: -x[1])
    time_rdd = time.time() - start_rdd

    for word, count in top_words_rdd:
        print(f"{word}: {count}")
    print(f"--- Время выполнения RDD: {time_rdd:.2f} сек ---")


    print("\n" + "="*50)
    print(">>> WORDCOUNT ЧЕРЕЗ DATAFRAME API <<<")
    print("="*50)
    start_df = time.time()

    df = spark.read.text(input_path)

    words_df = df.select(explode(split(col("value"), "\\s+")).alias("word")) \
                 .filter(col("word") != "") \
                 .select(lower(col("word")).alias("word"))

    word_counts_df = words_df.groupBy("word").count().orderBy(col("count").desc())

    word_counts_df.show(10)
    time_df = time.time() - start_df

    print(f"--- Время выполнения DataFrame API: {time_df:.2f} сек ---")

    spark.stop()

if __name__ == "__main__":
    main()