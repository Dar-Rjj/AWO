FROM python@sha256:2d97f6910b16bd338d3060f261f53f144965f755599aab1acda1e13cf1731b1b

LABEL org.awo.profile="python3.9-numpy2.0.2" \
      org.awo.base.digest="sha256:2d97f6910b16bd338d3060f261f53f144965f755599aab1acda1e13cf1731b1b" \
      org.awo.numpy.sha256="f26b258c385842546006213344c50655ff1555a9338e2e5e02a0756dc3e803dd"

COPY requirements.txt /tmp/awo-sandbox-requirements.txt
RUN pip install --no-cache-dir --require-hashes -r /tmp/awo-sandbox-requirements.txt \
    && rm /tmp/awo-sandbox-requirements.txt \
    && python -c "import numpy; assert numpy.__version__ == '2.0.2'"
