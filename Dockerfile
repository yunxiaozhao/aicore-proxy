FROM python:3.12-slim

WORKDIR /app
RUN pip install --no-cache-dir flask requests gunicorn
COPY proxy.py .
EXPOSE 6655

CMD ["gunicorn", "--bind", "0.0.0.0:6655", "proxy:app"]
