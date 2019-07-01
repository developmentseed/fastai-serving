import sys
import json
from base64 import b64encode, b64decode
from io import BytesIO

import aiohttp
import asyncio
import uvicorn
import torch
from fastai.vision import pil2tensor, load_learner
from starlette.applications import Starlette
from starlette.middleware.cors import CORSMiddleware
from starlette.responses import JSONResponse
from PIL import Image
import numpy as np

app = Starlette()
app.add_middleware(CORSMiddleware, allow_origins=['*'], allow_headers=['X-Requested-With', 'Content-Type'])

async def setup_learner():
    learner = load_learner('model')
    return learner

loop = asyncio.get_event_loop()
tasks = [asyncio.ensure_future(setup_learner())]
learner = loop.run_until_complete(asyncio.gather(*tasks))[0]
loop.close()

@app.route('/analyze:predict', methods=['POST'])
async def analyze(request):
    data = await request.body()
    instances = json.loads(data.decode('utf-8'))['instances']

    # convert from image bytes to images to tensors
    img_bytes = [b64decode(inst['image_bytes']['b64']) for inst in instances]
    tensors = [pil2tensor(Image.open(BytesIO(byts)), dtype=np.float32).div_(255) for byts in img_bytes]

    # batch predict, dummy labels for the second argument
    dummy_labels = torch.zeros(len(tensors))
    tensor_stack = torch.stack(tensors)
    if torch.cuda.is_available():
        tensor_stack = tensor_stack.cuda()
    prediction = learner.pred_batch(batch=(tensor_stack, dummy_labels))

    # find the maximum value along the prediction axis
    classes = np.argmax(np.array(prediction), axis=1)
    return JSONResponse(dict(predictions=classes.tolist()))


if __name__ == '__main__':
    if 'serve' in sys.argv:
        uvicorn.run(app=app, host='0.0.0.0', port=8501, log_level="info")
