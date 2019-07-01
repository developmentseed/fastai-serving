import sys
import time
from io import BytesIO

from fastai.vision import load_learner, open_image
import torch

learner = load_learner('fixtures')

num_image = int(sys.argv[1])
tensor = torch.randn(num_image, 3, 224, 224)

t1 = time.time()
learner.pred_batch(batch=(tensor, torch.randn(0)))

t2 = time.time()
print(f'{t2 - t1} seconds, {(t2 - t1) / num_image} per image')
