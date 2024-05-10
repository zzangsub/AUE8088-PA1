from torchmetrics import Metric
import torch

# [TODO] Implement this!
class MyF1Score(Metric):
    
    def __init__(self):
        super(MyF1Score, self).__init__()
        self.add_state("tp", default=torch.tensor(0), dist_reduce_fx="sum")
        self.add_state("fp", default=torch.tensor(0), dist_reduce_fx="sum")
        self.add_state("fn", default=torch.tensor(0), dist_reduce_fx="sum")

    def update(self, preds: torch.Tensor, target: torch.Tensor):
        # 예측값에서 가장 높은 확률의 인덱스를 찾습니다.
        preds_max_index = torch.argmax(preds, dim=1)

        # True Positive (정답을 정답이라고 예측)
        tp = torch.sum((preds_max_index == target) & (target == 1))
        # False Positive (오답을 정답이라고 예측)
        fp = torch.sum((preds_max_index != target) & (preds_max_index == 1))
        # False Negative (정답을 오답이라고 예측)
        fn = torch.sum((preds_max_index != target) & (target == 1))

        self.tp += tp
        self.fp += fp
        self.fn += fn

    def compute(self):
        # 정밀도와 재현율을 계산
        precision = self.tp / (self.tp + self.fp + 1e-6)  # 분모가 0이 되는 것을 방지
        recall = self.tp / (self.tp + self.fn + 1e-6)
        # F1 Score 계산
        f1_score = 2 * (precision * recall) / (precision + recall + 1e-6)  # 분모가 0이 되는 것을 방지
        return f1_score

class MyAccuracy(Metric):
    def __init__(self):
        super().__init__()
        self.add_state('total', default=torch.tensor(0), dist_reduce_fx='sum')
        self.add_state('correct', default=torch.tensor(0), dist_reduce_fx='sum')

    def update(self, preds, target):
        # [TODO] The preds (B x C tensor), so take argmax to get index with highest confidence
        preds_max_index = torch.argmax(preds, dim=1)

     
        # [TODO] check if preds and target have equal shape
        assert preds_max_index.shape == target.shape, "preds_max_index and target must have the same shape"
        
        # [TODO] Cound the number of correct prediction
        correct = torch.sum(preds_max_index == target)
 
        # Accumulate to self.correct
        self.correct += correct

        # Count the number of elements in target
        self.total += target.numel()

    def compute(self):
        return self.correct.float() / self.total.float()
