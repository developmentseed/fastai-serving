# fastai serving

A Docker image for serving [fastai](https://www.fast.ai/) models, mimicking the API of [Tensorflow Serving](https://github.com/tensorflow/serving).  It is designed for running batch inference at scale. It is not optimized for performance (but it's not [that slow](benchmark)).

## Build

First, export a fastai `Learner` with [`.export`](https://docs.fast.ai/basic_train.html#Learner.export). Assuming that this file is in `model_dir`, you can build the serving image like so:

```
# docker build -f Dockerfile.[cpu/gpu] --build-arg MODEL_DIR=./model_dir -t <org>/<image>:<tag> .`
```

If you require additional utils files for loading the model with [`load_learner`](https://docs.fast.ai/basic_train.html#load_learner), you can mount an additional directory at build time with:

```
# docker build -f Dockerfile.[cpu/gpu] --build-arg MODEL_DIR=./model_dir --build-arg UTILS_DIR=./utils -t org/image:tag .`
```

## Run

```
docker run --rm -p 8501:8501 -t org/image:tag .
```

## Use

The API currently has two endpoints:

### `POST /analyze:predict`

Accepts a JSON request in the form:

```js
{
  "instances": [
    {
      "image_bytes": {
        "b64": "[b64_string]"
      }
    }
  ],
  ...
}
```

where each `b64_string` is a base-64 encoded string representing the model input.

### `GET /analyze`

Returns an HTTP Status of `200` as long as the API is running (health check).

### Limitations, Motivation, and Future Directions

- This was written so fastai models could be used with [chip-n-scale](https://github.com/developmentseed/chip-n-scale-queue-arranger), an orchestration pipeline for running machine learning inference at scale. It has only been tested in that context.
- It has only been tested with a few CNN models.
- It only uses the [first transform from the validation data loader](https://github.com/developmentseed/fastai-serving/blob/master/src/server.py#L50) to transform input data.
- **Comparison to TensorFlow Serving**: This repo currently only implements a single replica of a TensorFlow serving endpoint and doesn't have any of the additional features that it supports (multiple models, gRPC support, batching scheduler, etc.). We're happy to accept pull requests which increase the functionality in this regard.
- **Pytorch JIT**: A [popular guide to deploying PyTorch models](https://medium.com/datadriveninvestor/deploy-your-pytorch-model-to-production-f69460192217#1bc6) (of which fastai models are a subset), shows how to create a `traced_script_module` for faster inference. Future iterations of this repo may explore these methods to improve inference times.

## Acknowledgments

- The code for `server.py` is taken almost entirely from the [fastai example](https://github.com/render-examples/fastai-v3) for [Render](https://render.com/). The primary addition is the batch inference code which can provide significant speed-ups compared to single image prediction.
- This work was undertaken in partnership with our friends at [Sinergise](https://www.sinergise.com/) and funded by the [European Space Agency](https://www.esa.int/ESA), specifically [Phi Lab](http://blogs.esa.int/philab/)
