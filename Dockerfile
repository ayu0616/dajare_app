FROM python:3.10

RUN apt-get update && \
    apt-get install mecab libmecab-dev mecab-ipadic-utf8 sudo -y && \
    apt-get clean && \
    rm -rf /var/lib/apt/lists/* && \
    git clone --depth 1 https://github.com/neologd/mecab-ipadic-neologd.git && \
    cd mecab-ipadic-neologd && \
    bin/install-mecab-ipadic-neologd -n -y

WORKDIR /app

COPY . /app

RUN pip install --upgrade pip && \
pip install -r requirements.txt

EXPOSE 3000

CMD ["python", "-m", "flask", "run", "--host=0.0.0.0", "--port=3000"]