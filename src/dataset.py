"""토크나이징. 입력 구성은 configs/base.yaml 의 dataset.input_mode 가 정한다 (SPEC 3-6)."""

import numpy as np
import torch

CLASSES = ["normal", "kernel_mem", "kernel_ops", "app"]


def build_text(df, input_mode):
    if input_mode == "msg_only":
        return df["msg_norm"].to_numpy()
    if input_mode == "with_meta":
        # level(FATAL/INFO)은 정답에 가까운 힌트다. E5 비교용이며 대표 성능이 아니다.
        return (df["component"] + " " + df["level"] + " " + df["msg_norm"]).to_numpy()
    raise SystemExit(f"모르는 input_mode: {input_mode} (msg_only | with_meta)")


def encode_unique(texts, tokenizer, max_length):
    """고유 텍스트만 토크나이징하고, 각 행이 가리키는 인덱스를 돌려준다.

    train 은 고유 템플릿 16,947종에 194.7배 중복이다. 행마다 토크나이징하면
    3,299,255 x 64 토큰을 들고 있어야 하고, 예측도 같은 계산을 200번 반복한다.
    """
    uniq, codes = np.unique(texts, return_inverse=True)
    enc = tokenizer(
        list(uniq), truncation=True, max_length=max_length, padding="max_length",
        return_tensors="pt",
    )
    return enc, codes


class LogDataset(torch.utils.data.Dataset):
    def __init__(self, df, tokenizer, max_length, input_mode):
        enc, self.codes = encode_unique(build_text(df, input_mode), tokenizer, max_length)
        self.input_ids = enc["input_ids"]
        self.attention_mask = enc["attention_mask"]
        self.labels = torch.tensor([CLASSES.index(c) for c in df["label4"]], dtype=torch.long)

    def __len__(self):
        return len(self.codes)

    def __getitem__(self, i):
        c = self.codes[i]
        return {
            "input_ids": self.input_ids[c],
            "attention_mask": self.attention_mask[c],
            "labels": self.labels[i],
        }
