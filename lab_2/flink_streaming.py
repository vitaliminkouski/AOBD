"""Flink DataStream API (PyFlink): event-time, watermark, tumbling/sliding/session окна,
keyed state и замер задержки.

Режимы: tumbling | sliding | session | state | latency [tuned]
"""
import json
import sys
from datetime import datetime, timezone

from pyflink.common import Configuration, Duration, Time, Types, WatermarkStrategy
from pyflink.common.serialization import SimpleStringSchema
from pyflink.common.watermark_strategy import TimestampAssigner
from pyflink.datastream import StreamExecutionEnvironment
from pyflink.datastream.connectors.kafka import (
    KafkaOffsetsInitializer, KafkaRecordSerializationSchema, KafkaSink, KafkaSource,
)
from pyflink.datastream.functions import KeyedProcessFunction, ProcessWindowFunction
from pyflink.datastream.state import MapStateDescriptor
from pyflink.datastream.window import (
    EventTimeSessionWindows, SlidingEventTimeWindows, TumblingEventTimeWindows,
)

BOOTSTRAP = "kafka:9092"
TOPIC = "events"
OUT_TOPIC = "latency-flink"
WATERMARK_SEC = 10

EVENT_TYPE = Types.TUPLE([Types.STRING(), Types.STRING(), Types.DOUBLE(), Types.LONG(), Types.LONG()])


def to_ms(iso: str) -> int:
    return int(datetime.fromisoformat(iso.replace("Z", "+00:00")).timestamp() * 1000)


def parse(raw: str):
    """JSON -> (event_id, key, value, event_time_ms, send_time_ms)"""
    d = json.loads(raw)
    return d["event_id"], d["key"], float(d["value"]), to_ms(d["event_time"]), to_ms(d["send_time"])


class EventTimeAssigner(TimestampAssigner):
    def extract_timestamp(self, value, record_timestamp):
        return value[3]


def fmt(ms: int) -> str:
    return datetime.fromtimestamp(ms / 1000, tz=timezone.utc).strftime("%H:%M:%S")


class CountWindowFunction(ProcessWindowFunction):
    """Считает события по ключу в окне и печатает границы окна"""

    def process(self, key, context, elements):
        w = context.window()
        cnt = sum(1 for _ in elements)
        yield f"[{self.kind}] window=[{fmt(w.start)}, {fmt(w.end)}) key={key} count={cnt}"

    kind = "window"


class TumblingFn(CountWindowFunction):
    kind = "tumbling"


class SlidingFn(CountWindowFunction):
    kind = "sliding"


class SessionFn(CountWindowFunction):
    kind = "session"


class MinuteCounter(KeyedProcessFunction):

    def open(self, runtime_context):
        self.counts = runtime_context.get_map_state(
            MapStateDescriptor("minute_counts", Types.LONG(), Types.LONG()))

    def process_element(self, value, ctx):
        bucket_end = (value[3] // 60000 + 1) * 60000
        if bucket_end <= ctx.timer_service().current_watermark():
            return  # событие опоздало (минута уже закрыта водяной меткой)
        self.counts.put(bucket_end, (self.counts.get(bucket_end) or 0) + 1)
        ctx.timer_service().register_event_time_timer(bucket_end)

    def on_timer(self, timestamp, ctx):
        yield (f"[state] minute=[{fmt(timestamp - 60000)}, {fmt(timestamp)}) "
               f"key={ctx.get_current_key()} count={self.counts.get(timestamp)}")
        self.counts.remove(timestamp)


def add_latency(raw: str) -> str:
    """Добавляет processed_at к событию (для расчёта latency)."""
    d = json.loads(raw)
    d["processed_at"] = datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%S.%f")[:-3] + "Z"
    return json.dumps({"event_id": d["event_id"], "send_time": d["send_time"],
                       "processed_at": d["processed_at"]})


def main(mode: str, tuned: bool = False):
    env = StreamExecutionEnvironment.get_execution_environment()
    env.set_parallelism(2)
    if tuned:
        # по умолчанию Python-операторы копят записи в бандл до 1000 мс -> задержка ~1 с
        conf = Configuration()
        conf.set_string("python.fn-execution.bundle.time", "10")
        conf.set_string("python.fn-execution.bundle.size", "100")
        env.configure(conf)

    source = KafkaSource.builder() \
        .set_bootstrap_servers(BOOTSTRAP) \
        .set_topics(TOPIC) \
        .set_group_id(f"flink-lab2-{mode}") \
        .set_starting_offsets(KafkaOffsetsInitializer.latest()) \
        .set_value_only_deserializer(SimpleStringSchema()) \
        .build()

    if mode == "latency":
        raw = env.from_source(source, WatermarkStrategy.no_watermarks(), "kafka-events")
        sink = KafkaSink.builder() \
            .set_bootstrap_servers(BOOTSTRAP) \
            .set_record_serializer(KafkaRecordSerializationSchema.builder()
                                   .set_topic(OUT_TOPIC)
                                   .set_value_serialization_schema(SimpleStringSchema())
                                   .build()) \
            .build()
        raw.map(add_latency, output_type=Types.STRING()).sink_to(sink)
        env.execute("lab2-flink-latency")
        return

    # event-time + bounded out-of-orderness watermark (10 c)
    strategy = WatermarkStrategy \
        .for_bounded_out_of_orderness(Duration.of_seconds(WATERMARK_SEC)) \
        .with_timestamp_assigner(EventTimeAssigner()) \
        .with_idleness(Duration.of_seconds(5))

    stream = env.from_source(source, WatermarkStrategy.no_watermarks(), "kafka-events") \
        .map(parse, output_type=EVENT_TYPE) \
        .assign_timestamps_and_watermarks(strategy)

    keyed = stream.key_by(lambda e: e[1], key_type=Types.STRING())

    if mode == "tumbling":
        out = keyed.window(TumblingEventTimeWindows.of(Time.minutes(1))) \
            .process(TumblingFn(), Types.STRING())
    elif mode == "sliding":
        out = keyed.window(SlidingEventTimeWindows.of(Time.minutes(1), Time.seconds(30))) \
            .process(SlidingFn(), Types.STRING())
    elif mode == "session":
        out = keyed.window(EventTimeSessionWindows.with_gap(Time.seconds(10))) \
            .process(SessionFn(), Types.STRING())
    elif mode == "state":
        out = keyed.process(MinuteCounter(), Types.STRING())
    else:
        raise SystemExit(f"unknown mode {mode}")

    out.print()
    env.execute(f"lab2-flink-{mode}")


if __name__ == "__main__":
    main(sys.argv[1] if len(sys.argv) > 1 else "tumbling", "tuned" in sys.argv[2:])
