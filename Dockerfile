FROM node:24-bookworm
RUN apt-get update -qq && apt-get install -y -qq libprimesieve-dev primesieve > /dev/null && rm -rf /var/lib/apt/lists/*
