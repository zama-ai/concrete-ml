FROM python:3.9-slim

WORKDIR /project

# Copy Python dependencies, configuration files and Python server
COPY requirements_server.txt utils_server.py server_glwe.py machine_configuration.sh .

# Copy remote modules
COPY deployment_float/compiled_models.tar.gz deployment_float/compiled_models.tar.gz

RUN chmod +x machine_configuration.sh

RUN apt-get update && \
    apt-get install -y net-tools && \
    pip install --upgrade pip && \
    pip install --no-cache-dir -r requirements_server.txt

    # Add a non-root user and group
RUN groupadd -g 10001 grp02 && \
    useradd -u 10000 -g grp02 usr01 && \
    chown -R usr01:grp02 /project

# Expose necessary ports
EXPOSE 80
EXPOSE 8000
EXPOSE 5000

# Switch to the non-root user
USER usr01:grp02

CMD ["python", "server_glwe.py"]
