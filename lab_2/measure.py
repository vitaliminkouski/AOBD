"""Замер latency и throughput по результатам latency-режима (топики latency-spark / latency-flink)"""
import argparse
import json
import statistics
from datetime import datetime

from kafka import KafkaConsumer


def ms(iso: str) -> float:
    return datetime.fromisoformat(iso.replace("Z", "+00:00")).timestamp() * 1000


def pct(data, q):
    data = sorted(data)
    return data[min(len(data) - 1, int(len(data) * q))]


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--bootstrap", default="localhost:29092")
    p.add_argument("--topic", required=True)
    p.add_argument("--label", required=True)
    p.add_argument("--since-ms", type=float, default=0, help="учитывать записи с send_time >= значения")
    p.add_argument("--out", default="")
    args = p.parse_args()

    c = KafkaConsumer(args.topic, bootstrap_servers=args.bootstrap, auto_offset_reset="earliest",
                      consumer_timeout_ms=8000, enable_auto_commit=False)
    proc, e2e, done = [], [], []
    for m in c:
        d = json.loads(m.value)
        send, processed = ms(d["send_time"]), ms(d["processed_at"])
        if send < args.since_ms:
            continue
        proc.append(processed - send)
        e2e.append(m.timestamp - send)
        done.append(processed)

    if not proc:
        raise SystemExit("нет данных")
    span = (max(done) - min(done)) / 1000
    res = {
        "engine": args.label, "events": len(proc), "span_s": round(span, 1),
        "throughput_eps": round(len(proc) / span, 1),
        "proc_ms": {"p50": round(pct(proc, .5)), "p95": round(pct(proc, .95)),
                    "max": round(max(proc)), "mean": round(statistics.mean(proc))},
        "e2e_ms": {"p50": round(pct(e2e, .5)), "p95": round(pct(e2e, .95)),
                   "max": round(max(e2e)), "mean": round(statistics.mean(e2e))},
    }
    print(json.dumps(res, ensure_ascii=False, indent=2))
    if args.out:
        json.dump(res, open(args.out, "w"), ensure_ascii=False, indent=2)


if __name__ == "__main__":
    main()
