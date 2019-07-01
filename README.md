# Fast.ai serving

Docker image for serving fast.ai models, mimicking the API of Tensorflow Serving. It is not optimized for performance.

## Build

```
docker build -t fastai/serving .
```

## Run

```
docker run --rm -p 8501:8501 -v $PWD/model_folder:/workdir/model -t fastai/serving
```
