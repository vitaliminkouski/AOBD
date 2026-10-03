import random
import time
from pyspark.sql import SparkSession

def main():
    spark = SparkSession.builder \
        .appName("PageRank-1000-Nodes") \
        .getOrCreate()

    sc = spark.sparkContext
    sc.setLogLevel("WARN")

    print("\n" + "="*50)
    print(">>> ГЕНЕРАЦИЯ ГРАФА ИЗ 1000 ВЕРШИН <<<")
    print("="*50)

    random.seed(42)
    num_nodes = 1000
    edges = []
    for src in range(1, num_nodes + 1):
        num_neighbors = random.randint(1, 10)
        neighbors = random.sample(range(1, num_nodes + 1), num_neighbors)
        for dst in neighbors:
            if src != dst:
                edges.append((src, dst))

    
    links = sc.parallelize(edges).groupByKey().partitionBy(4).cache()

    ranks = links.mapValues(lambda _: 1.0)

    num_iterations = 10
    damping_factor = 0.85

    print(f"Запуск {num_iterations} итераций PageRank...")
    start_time = time.time()

    for iteration in range(num_iterations):
        
        contribs = links.join(ranks).flatMap(
            lambda pair: [(dest, pair[1][1] / len(pair[1][0])) for dest in pair[1][0]]
        )

        ranks = contribs.reduceByKey(lambda a, b: a + b).mapValues(
            lambda rank_sum: (1 - damping_factor) + damping_factor * rank_sum
        )

    top_ranks = ranks.takeOrdered(10, key=lambda x: -x[1])
    total_time = time.time() - start_time

    print("\n" + "="*50)
    print(">>> ТОП-10 ВЕРШИН ПО РЕЗУЛЬТАТАМ PAGERANK <<<")
    print("="*50)
    for node, rank in top_ranks:
        print(f"Страница #{node:4d} | Ранг: {rank:.4f}")

    print(f"\nВремя работы алгоритма ({num_iterations} итераций): {total_time:.2f} сек")

    spark.stop()

if __name__ == "__main__":
    main()