import torch.nn as nn
import torch


target = torch.ones([2, 4], dtype=torch.float32)

output = torch.full([2, 4], 0.999)

pos_weight = torch.ones([4])

criterion = torch.nn.BCEWithLogitsLoss(pos_weight=pos_weight)

criterion(output, target)