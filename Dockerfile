# git dan git-lfs
FROM alpine/git as lfs

WORKDIR /src

RUN git clone -b staging https://github.com/AnimaLink/ml-deploy.git .

RUN git lfs pull

FROM python:3.10-alpine

WORKDIR /usr/src/app

COPY --from=lfs /src .

RUN pip install -r requirements.txt

EXPOSE 8080

CMD ["python", "app.py"]
