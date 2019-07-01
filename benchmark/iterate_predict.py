import sys
import time
from io import BytesIO

from PIL import Image
from fastai.vision import load_learner, open_image
import numpy as np

learner = load_learner('fixtures')

num_image = int(sys.argv[1])
# create random arrays -> images -> bytes -> fast.ai images
arrays = (np.random.randn(num_image, 224, 224, 3) * 255).astype(np.uint8)
images = [Image.fromarray(arrays[i]) for i in range(num_image)]
byts = [BytesIO() for i in range(num_image)]
for idx, img in enumerate(images):
    img.save(byts[idx], format='png')

fimages = [open_image(b) for b in byts]

t1 = time.time()
for fimg in fimages:
    learner.predict(fimg)

t2 = time.time()
print(f'{t2 - t1} seconds, {(t2 - t1) / num_image} per image')
