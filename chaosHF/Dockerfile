FROM python:3.11-slim
WORKDIR /app

COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

COPY app.py .
COPY src/ ./src/
COPY openenv.yaml inference.py ./

EXPOSE 7860
HEALTHCHECK --interval=10s --timeout=5s --start-period=2s --retries=2 \
  CMD python -c "from src.environment import MLPipelineEnv; MLPipelineEnv().reset(); print('READY')" || exit 1

CMD sh -c "python app.py; tail -f /dev/null"
#checking if it updated
 
 