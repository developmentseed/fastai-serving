import sys
import time
import json
from io import BytesIO
from base64 import b64encode, b64decode

from PIL import Image
from fastai.vision import load_learner, open_image, pil2tensor
import numpy as np
import torch

learner = load_learner('fixtures/classification')


num_image = int(sys.argv[1])
arrays = (np.random.randn(num_image, 224, 224, 3) * 255).astype(np.uint8)

images = [Image.fromarray(arrays[i]) for i in range(num_image)]
byte_arrays = [BytesIO() for i in range(num_image)]
for idx, byt in enumerate(byte_arrays):
    images[idx].save(byt, format='png')

b64_strs = [b64encode(byt.getvalue()).decode('utf-8') for byt in byte_arrays]
instances = [{'image_bytes': {'b64': b64_str}} for b64_str in b64_strs]
payload = json.dumps(dict(instances=instances))


def api_batch_predict(payload):
    instances = json.loads(payload)['instances']
    img_bytes = [b64decode(inst['image_bytes']['b64']) for inst in instances]
    tensors = [pil2tensor(Image.open(BytesIO(byts)), dtype=np.float32).div_(255) for byts in img_bytes]

    # batch predict, dummy labels for the second argument
    dummy_labels = torch.zeros(len(tensors))
    tensor_stack = torch.stack(tensors)
    if torch.cuda.is_available():
        tensor_stack = tensor_stack.cuda()
    learner.pred_batch(batch=(tensor_stack, dummy_labels))

def api_batch_tfm_predict(payload):
    instances = json.loads(payload)['instances']
    img_bytes = [b64decode(inst['image_bytes']['b64']) for inst in instances]
    tensors = [pil2tensor(Image.open(BytesIO(byts)), dtype=np.float32).div_(255) for byts in img_bytes]
    tfm_tensors = [learner.data.valid_dl.tfms[0]((tensor, torch.zeros(0)))[0] for tensor in tensors]

    # batch predict, dummy labels for the second argument
    dummy_labels = torch.zeros(len(tfm_tensors))
    tensor_stack = torch.stack(tfm_tensors)
    if torch.cuda.is_available():
        tensor_stack = tensor_stack.cuda()
    learner.pred_batch(batch=(tensor_stack, dummy_labels))

def api_iterate_predict(payload):
    instances = json.loads(payload)['instances']
    img_bytes = [b64decode(inst['image_bytes']['b64']) for inst in instances]

    # batch predict, dummy labels for the second argument
    for byts in img_bytes:
        fimg = open_image(BytesIO(byts))
        learner.predict(fimg)

t1 = time.time()
api_iterate_predict(payload)
t2 = time.time()
print('Iteration')
print(f'{t2 - t1} seconds, {(t2 - t1) / num_image} per image')

t1 = time.time()
api_batch_predict(payload)
t2 = time.time()
print('Batch')
print(f'{t2 - t1} seconds, {(t2 - t1) / num_image} per image')

t1 = time.time()
api_batch_tfm_predict(payload)
t2 = time.time()
print('Batch w/transform')
print(f'{t2 - t1} seconds, {(t2 - t1) / num_image} per image')
