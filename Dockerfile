FROM nrel/energyplus:25.1.0

USER root
WORKDIR /app

RUN apt-get update \
    && apt-get install -y --no-install-recommends python3 python3-pip ca-certificates \
    && rm -rf /var/lib/apt/lists/*

RUN python3 -m pip install --no-cache-dir fastmcp==3.2.4

COPY worker /app/worker

ENV PYTHONPATH=/app:/EnergyPlus-25.1.0-68a4a7c774-Linux-Ubuntu22.04-x86_64
ENV ENERGYPLUS_DIR=/EnergyPlus-25.1.0-68a4a7c774-Linux-Ubuntu22.04-x86_64
ENV PATH=/EnergyPlus-25.1.0-68a4a7c774-Linux-Ubuntu22.04-x86_64:${PATH}

ENTRYPOINT []
CMD ["python3", "-m", "worker.run", "--help"]
