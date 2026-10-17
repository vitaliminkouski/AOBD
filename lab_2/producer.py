"""Продюсер: имитирует поток событий (JSON с timestamp) и отправляет в Kafka"""
import argparse
import json
import random
import time
import uuid
from datetime import datetime, timezone, timedelta

from kafka import KafkaProducer

KEYS = ["user_1", "user_2", "user_3", "user_4", "user_5"]


def iso(ts: float) -> str:
    return datetime.fromtimestamp(ts, tz=timezone.utc).strftime("%Y-%m-%dT%H:%M:%S.%f")[:-3] + "Z"


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--bootstrap", default="localhost:29092")
    p.add_argument("--topic", default="events")
    p.add_argument("--rate", type=int, default=50, help="событий в секунду")
    p.add_argument("--duration", type=int, default=0, help="секунд (0 = бесконечно)")
    p.add_argument("--late-prob", type=float, default=0.1,
                   help="доля событий, пришедших с опозданием (event_time в прошлом)")
    p.add_argument("--max-late", type=float, default=20.0,
                   help="максимальное опоздание, сек")
    p.add_argument("--burst-pause", type=float, default=0.0,
                   help="каждые 40 сек делать паузу N сек (для session-окон)")
    args = p.parse_args()

    producer = KafkaProducer(
        bootstrap_servers=args.bootstrap,
        value_serializer=lambda v: json.dumps(v).encode("utf-8"),
        key_serializer=lambda k: k.encode("utf-8"),
        linger_ms=5,
    )

    start = time.time()
    sent = 0
    next_pause = start + 40
    interval = 1.0 / args.rate
    next_tick = start
    try:
        while args.duration == 0 or time.time() - start < args.duration:
            now = time.time()
            if args.burst_pause and now >= next_pause:
                print(f"[producer] пауза {args.burst_pause}s (разрыв для session-окна)")
                time.sleep(args.burst_pause)
                next_pause = time.time() + 40
                next_tick = time.time()
                continue
            event_time = now
            if random.random() < args.late_prob:
                event_time = now - random.uniform(1.0, args.max_late)
            key = random.choice(KEYS)
            msg = {
                "event_id": str(uuid.uuid4()),
                "key": key,
                "value": round(random.uniform(0, 100), 2),
                "event_time": iso(event_time),
                "send_time": iso(now),
            }
            producer.send(args.topic, key=key, value=msg)
            sent += 1
            if sent % (args.rate * 5) == 0:
                print(f"[producer] отправлено {sent} событий, {sent / (time.time() - start):.1f} ev/s")
            next_tick += interval
            delay = next_tick - time.time()
            if delay > 0:
                time.sleep(delay)
    except KeyboardInterrupt:
        pass
    finally:
        producer.flush()
        elapsed = time.time() - start
        print(f"[producer] итого {sent} событий за {elapsed:.1f}s ({sent / elapsed:.1f} ev/s)")


if __name__ == "__main__":
    main()
