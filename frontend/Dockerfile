# syntax=docker/dockerfile:1.7
FROM node:22-alpine AS build
WORKDIR /app
COPY package.json package-lock.json ./
RUN npm ci
COPY . .
# Baked at build time: Vite inlines VITE_* into the bundle, so changing the API
# base URL means rebuilding this image (or using the nginx proxy path below).
ARG VITE_API_BASE=/api
ENV VITE_API_BASE=${VITE_API_BASE}
RUN npm run build

FROM nginx:1.27-alpine AS runtime
COPY --from=build /app/dist /usr/share/nginx/html
# A plain config, not a template: nginx no longer handles the API key (see
# nginx.conf), so there is nothing to substitute at start.
COPY nginx.conf /etc/nginx/conf.d/default.conf
EXPOSE 80
# 127.0.0.1, not localhost: localhost resolves to ::1 first and nginx listens
# only on IPv4, so the probe would get ECONNREFUSED against a healthy server.
HEALTHCHECK --interval=30s --timeout=5s --retries=3 \
    CMD wget -qO- http://127.0.0.1/ >/dev/null || exit 1
