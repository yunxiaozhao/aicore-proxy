FROM python:3.12-slim

WORKDIR /app
RUN pip install --no-cache-dir flask requests gunicorn gevent packaging
COPY config.py proxy.py app.py models.py ./
EXPOSE 6655

CMD ["gunicorn", "--bind", "0.0.0.0:6655", "--timeout", "600", "--worker-class", "gevent", "app:app"]
