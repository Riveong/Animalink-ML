FROM python:3.10-alpine

WORKDIR /usr/src/app

COPY requirements.txt requirements.txt
RUN pip install -r requirements.txt

COPY . ./

EXPOSE 8080

CMD ["python", "app.py"]
