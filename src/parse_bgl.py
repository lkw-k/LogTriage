"""BGL.log -> parquet. 헤더 9개 + 메시지로 분리한다."""

import argparse
from collections import Counter
from pathlib import Path

import pyarrow as pa
import pyarrow.parquet as pq

COLUMNS = [
    "label",
    "unix_ts",
    "date",
    "node",
    "time",
    "node_repeat",
    "type",
    "component",
    "level",
    "message",
]

CHUNK = 500_000

SCHEMA = pa.schema(
    [(c, pa.int64() if c == "unix_ts" else pa.string()) for c in COLUMNS]
)


def parse_line(line):
    """성공하면 10개 필드 리스트, 실패하면 None.

    헤더 9개만 있고 메시지가 빈 줄은 깨진 줄이 아니라 메시지 없는 유효 레코드다.
    message="" 로 채운다.
    """
    parts = line.rstrip().split(maxsplit=9)
    if len(parts) == 9:
        parts.append("")
    if len(parts) != 10:
        return None
    if not parts[1].lstrip("-").isdigit():
        return None
    parts[1] = int(parts[1])
    return parts


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--input", required=True)
    ap.add_argument("--output", required=True)
    args = ap.parse_args()

    out = Path(args.output)
    out.parent.mkdir(parents=True, exist_ok=True)

    total = failed = empty_msg = 0
    label_counts = Counter()
    ts_min = ts_max = None
    buf = []
    writer = pq.ParquetWriter(out, SCHEMA)

    def flush():
        if not buf:
            return
        cols = list(zip(*buf))
        writer.write_table(
            pa.table({c: list(v) for c, v in zip(COLUMNS, cols)}, schema=SCHEMA)
        )
        buf.clear()

    with open(args.input, encoding="utf-8", errors="replace") as f:
        for line in f:
            if not line.strip():
                continue
            total += 1
            row = parse_line(line)
            if row is None:
                failed += 1
                continue
            label_counts[row[0]] += 1
            if not row[9]:
                empty_msg += 1
            ts = row[1]
            if ts_min is None or ts < ts_min:
                ts_min = ts
            if ts_max is None or ts > ts_max:
                ts_max = ts
            buf.append(row)
            if len(buf) >= CHUNK:
                flush()
    flush()
    writer.close()

    parsed = total - failed
    alerts = parsed - label_counts.get("-", 0)
    print(f"총 줄 수      : {total:,}")
    print(f"파싱 성공     : {parsed:,}")
    print(f"파싱 실패     : {failed:,} ({failed / total * 100:.4f}%)")
    print(f"이상(알럿)    : {alerts:,} ({alerts / parsed * 100:.2f}%)")
    print(f"원본 카테고리 : {len(label_counts) - 1}종")
    print(f"빈 메시지     : {empty_msg:,} ({empty_msg / parsed * 100:.4f}%)")
    print(f"unix_ts 범위  : {ts_min} ~ {ts_max}")
    print(f"출력          : {out}")


if __name__ == "__main__":
    main()
