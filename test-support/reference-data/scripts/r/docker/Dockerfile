FROM rocker/r-ver:4.6.1@sha256:4e6e6696ff54a86625c7f80046478e67014a2184e942b85eeff43db7b992d93e

ARG CRAN_SNAPSHOT=2026-07-01
ARG E1071_VERSION=1.7-17

LABEL org.opencontainers.image.title="R reference values generator"
LABEL org.opencontainers.image.description="Independent R and e1071 stats reference generator"
LABEL ref.cran.snapshot="${CRAN_SNAPSHOT}"
LABEL ref.e1071.version="${E1071_VERSION}"

ENV CRAN_SNAPSHOT=${CRAN_SNAPSHOT} \
    E1071_VERSION=${E1071_VERSION} \
    RENV_CONFIG_CACHE_ENABLED=FALSE \
    TZ=UTC \
    LC_ALL=C \
    OPENBLAS_NUM_THREADS=1 \
    OMP_NUM_THREADS=1

WORKDIR /reference/scripts/r
COPY .Rprofile renv.lock ./

# Bootstrap renv from the dated snapshot, then let the lockfile restore the
# complete package environment into the image's library, outside mounted source.
RUN Rscript -e 'install.packages("renv"); renv::restore(prompt = FALSE)'

COPY common.R moments.R dispersion.R quantiles.R association.R regression.R generate.R ./
RUN Rscript -e 'source("common.R"); check_environment()'
CMD ["Rscript", "generate.R"]
