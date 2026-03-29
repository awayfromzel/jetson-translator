FROM python:3.11-slim

WORKDIR /app

# Copy requirements
COPY tts_service/requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Copy service code
COPY tts_service /app/tts_service

# Run FastAPI app
CMD ["uvicorn", "tts_service.serve:app", "--host", "0.0.0.0", "--port", "5005"]
