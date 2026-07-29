import numpy as np

class MetricsEvaluator:
    """
    精准计算生物医学事件提取任务中的 P, R, F1 指标 (Eq. 1, 2, 3)[cite: 1]
    """
    def __init__(self, id2label, ignore_idx=-100):
        self.id2label = id2label
        self.ignore_idx = ignore_idx

    def compute_metrics(self, predictions, targets):
        """
        :param predictions: 概率预测转成标号矩阵 [batch, seq_len]
        :param targets: 真实 BIO 序列标签矩阵 [batch, seq_len]
        """
        tp, fp, fn = 0, 0, 0

        for pred_seq, target_seq in zip(predictions, targets):
            for p_id, t_id in zip(pred_seq, target_seq):
                if t_id == self.ignore_idx:
                    continue  # 跳过 Subword padding 或 [CLS]/[SEP]

                pred_label = self.id2label.get(p_id, "O")
                target_label = self.id2label.get(t_id, "O")

                is_pred_event = (pred_label != "O")
                is_target_event = (target_label != "O")

                if is_pred_event and is_target_event:
                    if pred_label == target_label:
                        tp += 1
                    else:
                        fp += 1
                        fn += 1
                elif is_pred_event and not is_target_event:
                    fp += 1
                elif not is_pred_event and is_target_event:
                    fn += 1

        precision = (tp / (tp + fp)) * 100.0 if (tp + fp) > 0 else 0.0
        recall = (tp / (tp + fn)) * 100.0 if (tp + fn) > 0 else 0.0
        f1 = (2 * precision * recall) / (precision + recall) if (precision + recall) > 0 else 0.0

        return {
            "precision": round(precision, 2),
            "recall": round(recall, 2),
            "f1": round(f1, 2),
            "tp": tp, "fp": fp, "fn": fn
        }