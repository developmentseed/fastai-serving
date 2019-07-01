FROM python:3.7.3-stretch

RUN pip install torch==1.0.1 torchvision==0.2.2 pillow fastai aiohttp asyncio uvicorn starlette

WORKDIR /workdir

EXPOSE 8501

COPY src /workdir/src

ENTRYPOINT ["python3.7", "src/server.py", "serve"]
