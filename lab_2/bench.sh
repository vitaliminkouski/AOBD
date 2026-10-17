#!/usr/bin/env bash
set -euo pipefail
ENGINE=$1; RATE=$2; DUR=$3; EXTRA=${4:-}   # EXTRA=tuned для Flink
PY=${PY:-python3}
cd "$(dirname "$0")"

docker exec spark pkill -f spark_streaming.py 2>/dev/null || true
for j in $(curl -s localhost:8081/jobs/overview | python3 -c "import sys,json;[print(x['jid']) for x in json.load(sys.stdin)['jobs'] if x['state']=='RUNNING']"); do
  docker exec flink-jobmanager flink cancel "$j" >/dev/null
done
sleep 3

if [ "$ENGINE" = spark ]; then
  docker exec spark rm -rf /tmp/ckpt-latency
  docker exec -d spark bash -c "/opt/spark/bin/spark-submit --master 'local[2]' --conf spark.jars.ivy=/tmp/ivy --packages org.apache.spark:spark-sql-kafka-0-10_2.13:4.2.0 /workspace/spark_streaming.py latency > /workspace/results/spark_latency.log 2>&1"
  sleep 45
  TOPIC=latency-spark; LABEL="Spark"
else
  docker exec flink-jobmanager flink run -d -py /workspace/flink_streaming.py latency $EXTRA >/dev/null
  sleep 25
  TOPIC=latency-flink; LABEL="Flink${EXTRA:+ ($EXTRA)}"
fi

SINCE=$(python3 -c "import time;print(int(time.time()*1000))")
$PY producer.py --rate "$RATE" --duration "$DUR" --late-prob 0 | tail -1
sleep 12
$PY measure.py --topic "$TOPIC" --label "$LABEL" --since-ms "$SINCE" --out "results/${ENGINE}${EXTRA:+_$EXTRA}_rate${RATE}.json"
