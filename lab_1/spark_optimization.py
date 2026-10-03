import time
from pyspark.sql import SparkSession
from pyspark.storagelevel import StorageLevel

def main():
    spark = SparkSession.builder \
        .appName("Optimization-Partitioning-Caching") \
        .getOrCreate()

    sc = spark.sparkContext
    sc.setLogLevel("WARN")

    input_path = "/workspace/input.txt"

    
    print("\n" + "="*60)
    print(">>> ТЕСТИРОВАНИЕ РАЗНОГО ЧИСЛА ПАРТИЦИЙ <<<")
    print("="*60)

    raw_rdd = sc.textFile(input_path)
    partition_counts = [2, 10, 50, 200]

    for p in partition_counts:
        test_rdd = raw_rdd.repartition(p)
        start_time = time.time()

        counts = test_rdd.flatMap(lambda line: line.split()) \
                         .map(lambda word: (word, 1)) \
                         .reduceByKey(lambda a, b: a + b) \
                         .count() 
        elapsed = time.time() - start_time
        print(f"Партиций: {p:3d} | Число уникальных слов: {counts} | Время: {elapsed:.2f} сек")

    
    print("\n" + "="*60)
    print(">>> СРАВНЕНИЕ REPARTITION VS COALESCE <<<")
    print("="*60)

    base_rdd = raw_rdd.repartition(40)
    print(f"Исходное число партиций: {base_rdd.getNumPartitions()}")

    start_rep = time.time()
    rep_rdd = base_rdd.repartition(2)
    rep_rdd.map(lambda line: len(line)).reduce(lambda a, b: a + b)
    time_rep = time.time() - start_rep
    print(f"Уменьшение до 2 партиций через [repartition]: {time_rep:.2f} сек (с Shuffle)")

    start_coal = time.time()
    coal_rdd = base_rdd.coalesce(2)
    coal_rdd.map(lambda line: len(line)).reduce(lambda a, b: a + b)
    time_coal = time.time() - start_coal
    print(f"Уменьшение до 2 партиций через [coalesce]:   {time_coal:.2f} сек (без Shuffle)")

    
    print("\n" + "="*60)
    print(">>> СРАВНЕНИЕ ВЫПОЛНЕНИЯ БЕЗ КЭША И С КЭШЕМ <<<")
    print("="*60)

    heavy_rdd = raw_rdd.flatMap(lambda line: line.split()) \
                       .filter(lambda w: len(w) > 3) \
                       .map(lambda w: (w.lower(), 1)) \
                       .reduceByKey(lambda a, b: a + b)

    print("\n-- 1. Выполнение БЕЗ кэширования --")
    t1 = time.time()
    count1 = heavy_rdd.count()
    print(f"Действие 1 (count): {count1} | Заняло: {time.time() - t1:.2f} сек")

    t2 = time.time()
    top1 = heavy_rdd.takeOrdered(5, key=lambda x: -x[1])
    print(f"Действие 2 (takeOrdered, повторный расчет): Заняло: {time.time() - t2:.2f} сек")

    print("\n-- 2. Выполнение С кэшированием (.persist) --")
    cached_rdd = heavy_rdd.persist(StorageLevel.MEMORY_AND_DISK)

    t3 = time.time()
    count2 = cached_rdd.count() 
    print(f"Действие 1 (первый прогрев и запись в кэш): {count2} | Заняло: {time.time() - t3:.2f} сек")

    t4 = time.time()
    top2 = cached_rdd.takeOrdered(5, key=lambda x: -x[1]) 
    print(f"Действие 2 (чтение из кэша памяти): Заняло: {time.time() - t4:.2f} сек")

    print("Приложение поставлено на паузу на 300 секунд")
    
    time.sleep(300) 
    spark.stop()

if __name__ == "__main__":
    main()