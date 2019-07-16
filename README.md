# Fast.ai serving

Docker image for serving fast.ai models, mimicking the API of [Tensorflow Serving](https://github.com/tensorflow/serving). It is not optimized for performance.

## Build

```
# docker build -f Dockerfile.[cpu/gpu] --build-arg MODEL_DIR=./model_dir -t org/image:tag .`
```

If you require additional utils files for loading the model with `load_learner`, you can mount an additional directory at build time with:

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

```json
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
