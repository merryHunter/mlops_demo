FROM postgres:15

# Install build dependencies
RUN apt-get update && apt-get install -y \
    build-essential \
    git \
    postgresql-server-dev-15 \
    && rm -rf /var/lib/apt/lists/*

# Clone and build pg_parquet
RUN git clone https://github.com/pgvector/pg_parquet.git \
    && cd pg_parquet \
    && make \
    && make install

# Clean up
RUN apt-get purge -y build-essential git postgresql-server-dev-15 \
    && apt-get autoremove -y \
    && apt-get clean \
    && rm -rf /var/lib/apt/lists/* /pg_parquet 