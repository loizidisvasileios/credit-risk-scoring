# The API and the dashboard in one container, so that the scoring service can
# be started without a virtual environment or a dataset.
#
# The Python minor version and the three library versions below are the ones
# recorded in models/model_metadata.json by the run that wrote the model. A
# pickled estimator is tied to the library that created it, so they are pinned
# exactly here while pyproject.toml keeps the ranges a developer needs.
FROM python:3.13-slim

# LightGBM links against OpenMP, which the slim image does not carry.
RUN apt-get update \
    && apt-get install -y --no-install-recommends libgomp1 \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /srv

# Only what the service needs: the package, the app and the configuration.
# README.md comes along because pyproject.toml declares it as the description.
COPY pyproject.toml README.md ./
COPY src/ src/
COPY app/ app/
COPY config/ config/

RUN pip install --no-cache-dir -e ".[app]" \
    "scikit-learn==1.7.2" \
    "lightgbm==4.7.0" \
    "xgboost==3.4.1"

# The dashboard reaches the API inside the container and listens on the port
# the platform publishes. 7860 is the port Hugging Face Spaces expects.
ENV CREDIT_API_URL=http://127.0.0.1:8000 \
    CREDIT_DASHBOARD_HOST=0.0.0.0 \
    CREDIT_DASHBOARD_PORT=7860

EXPOSE 7860

# The API runs in the background and the dashboard in the foreground: the
# foreground process is the one whose exit stops the container, and the
# dashboard is the part a visitor talks to. `exec` hands it the shell's process
# so that a stop signal reaches Python instead of the shell.
CMD ["sh", "-c", "uvicorn app.api:app --host 127.0.0.1 --port 8000 & exec python app/dashboard.py"]
