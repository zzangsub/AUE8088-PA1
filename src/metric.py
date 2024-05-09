from torchmetrics import Metric
import torch

# [TODO] Implement this!
class MyF1Score(Metric):
    pass

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
