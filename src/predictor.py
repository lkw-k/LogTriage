"""학습된 모델을 exp_id 로 불러온다. calibrate 와 infer 가 같은 경로로 쓴다.

E1 계열은 `runs/<exp_id>/model.joblib` (sklearn), E2 계열은 `runs/<exp_id>/checkpoint`
(transformers) 다. 어느 쪽 파일이 있는지 보고 고른다.

확률은 항상 `CLASSES` 순서로 돌려준다. sklearn 의 `clf.classes_` 는 알파벳순이라
그대로 쓰면 kernel_mem 과 app 이 뒤바뀐다.
"""

from pathlib import Path

import numpy as np

from src.dataset import CLASSES


class SklearnPredictor:
    def __init__(self, path):
        import joblib

        m = joblib.load(path)
        self.vec, self.clf = m["vectorizer"], m["clf"]
        self.order = [list(self.clf.classes_).index(c) for c in CLASSES]

    def predict_proba(self, texts):
        return self.clf.predict_proba(self.vec.transform(texts))[:, self.order]


class BertPredictor:
    def __init__(self, path, max_length, batch_size):
        import torch
        from transformers import AutoModelForSequenceClassification, AutoTokenizer

        self.torch = torch
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.tok = AutoTokenizer.from_pretrained(path)
        self.model = AutoModelForSequenceClassification.from_pretrained(path).to(self.device)
        self.model.eval()
        self.max_length, self.batch_size = max_length, batch_size

    def predict_proba(self, texts):
        torch = self.torch
        enc = self.tok(
            list(texts), truncation=True, max_length=self.max_length,
            padding="max_length", return_tensors="pt",
        )
        out = []
        with torch.no_grad():
            for i in range(0, len(enc["input_ids"]), self.batch_size):
                with torch.autocast("cuda", dtype=torch.bfloat16,
                                    enabled=self.device.type == "cuda"):
                    logits = self.model(
                        input_ids=enc["input_ids"][i : i + self.batch_size].to(self.device),
                        attention_mask=enc["attention_mask"][i : i + self.batch_size].to(
                            self.device),
                    ).logits
                out.append(torch.softmax(logits.float(), dim=-1).cpu())
        return torch.cat(out).numpy()


def load(exp_id, cfg):
    """runs/<exp_id> 에 있는 것을 보고 sklearn / BERT 중 하나를 고른다."""
    rundir = Path(cfg["paths"]["runs"]) / exp_id
    joblib_path, ckpt = rundir / "model.joblib", rundir / "checkpoint"
    if joblib_path.exists():
        return SklearnPredictor(joblib_path), "sklearn"
    if ckpt.exists():
        return BertPredictor(ckpt, cfg["dataset"]["max_length"],
                             cfg["train"]["batch_size"] * 4), "bert"
    raise SystemExit(
        f"{rundir} 에 model.joblib 도 checkpoint 도 없다. "
        f"baseline 또는 train 을 --exp-id {exp_id} 로 먼저 돌려야 한다."
    )


def predict_unique(pred, texts):
    """고유 텍스트만 모델에 통과시키고 행으로 되돌린다. 예측은 텍스트에만 의존한다."""
    uniq, codes = np.unique(np.asarray(texts), return_inverse=True)
    return pred.predict_proba(uniq)[codes]
