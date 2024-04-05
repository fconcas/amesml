FROM python:3.11-slim as builder

WORKDIR /app

ENV PYTHONDONTWRITEBYTECODE 1
ENV PYTHONUNBUFFERED 1

COPY requirements.txt .

RUN pip wheel \
	--no-cache-dir \
	--no-deps \
	--wheel-dir /app/wheels \
	-r requirements.txt


FROM python:3.11-slim

WORKDIR /app

RUN apt-get update \
	&& apt-get install -y --no-install-recommends \
	&& apt-get install -y curl \
	&& apt-get install libgomp1 \
	&& apt-get clean \
	&& rm -rf /var/lib/apt/lists/*

COPY --from=builder /app/wheels /wheels
COPY --from=builder /app/requirements.txt .

RUN pip install --no-cache /wheels/* \
	&& rm -rf /wheels

COPY . ./

CMD python app.py -a 0.0.0.0 -p 5000
