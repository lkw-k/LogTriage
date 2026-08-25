"""메시지 정규화. 순서가 중요하다 (SPEC 3-3). 숫자 치환은 반드시 마지막."""

import argparse
import re
from pathlib import Path

import pandas as pd

# (이름, 정규식, 치환) — 이 순서를 바꾸면 노드 ID가 부서진다.
#
# path 앞의 (?<![\w]) 는 단어 중간의 슬래시를 막는다. 이게 없으면
#   force load/store alignment   -> force load[PATH] alignment
#   Controlling BG/L rows        -> Controlling BG[PATH] rows
#   Torus/Tree/GI read error     -> Torus[PATH] read error
# 처럼 분류 단서가 통째로 사라진다. 진짜 경로는 공백이나 = 뒤에서 시작하므로
# 영향받지 않는다.
RULES = [
    ("node", re.compile(r"R\d+-M\d+-N\d+-C:J\d+-U\d+"), "[NODE]"),
    ("ip", re.compile(r"\d+\.\d+\.\d+\.\d+"), "[IP]"),
    ("hex", re.compile(r"0x[0-9a-fA-F]+"), "[HEX]"),
    ("path", re.compile(r"(?<![\w])(/[\w.\-]+)+"), "[PATH]"),
    ("num", re.compile(r"\d+"), "[NUM]"),
]

N_PATH_SAMPLES = 200


def normalize(msg):
    for _, pat, repl in RULES:
        msg = pat.sub(repl, msg)
    return msg


def path_step(msg):
    """path 규칙 직전까지 적용한 결과와 적용 후 결과. 과매칭 확인용."""
    for _, pat, repl in RULES[:3]:
        msg = pat.sub(repl, msg)
    _, pat, repl = RULES[3]
    return msg, pat.sub(repl, msg)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--input", required=True)
    ap.add_argument("--output", required=True)
    args = ap.parse_args()

    df = pd.read_parquet(args.input)
    df["message"] = df["message"].fillna("")
    before = df["message"].nunique()

    uniq = pd.Series(df["message"].unique())
    mapping = dict(zip(uniq, uniq.map(normalize)))
    df["msg_norm"] = df["message"].map(mapping)

    after = df["msg_norm"].nunique()
    emptied = ((df["message"].str.len() > 0) & (df["msg_norm"].str.len() == 0)).sum()

    # 측정 2: [PATH] 치환이 일어난 고유 메시지의 전/후 덤프
    out = Path(args.output)
    out.parent.mkdir(parents=True, exist_ok=True)
    dump = out.with_name("normalize_path_samples.txt")
    n_path = 0
    with open(dump, "w", encoding="utf-8") as f:
        for msg in uniq:
            pre, post = path_step(msg)
            if pre == post:
                continue
            n_path += 1
            if n_path <= N_PATH_SAMPLES:
                f.write(f"- {pre}\n+ {post}\n\n")

    df.to_parquet(out, index=False)

    print(f"고유 메시지 : {before:,} -> {after:,} ({(1 - after / before) * 100:.1f}% 감소)")
    print(f"[PATH] 치환 : 고유 메시지 {n_path:,}건")
    print(f"정규화로 비워진 줄 : {emptied:,}  (0 이어야 정상)")
    print(f"전/후 샘플  : {dump} (상위 {min(n_path, N_PATH_SAMPLES)}건)")
    print(f"출력        : {out}")
    if emptied:
        raise SystemExit("정규화가 메시지를 통째로 지웠다. 정규식을 확인할 것.")


if __name__ == "__main__":
    main()
