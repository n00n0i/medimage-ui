FROM nginx:alpine
RUN apk add --no-cache openssl && \
    mkdir -p /etc/nginx/ssl && \
    openssl req -x509 -nodes -days 3650 -newkey rsa:2048 \
      -keyout /etc/nginx/ssl/key.pem \
      -out /etc/nginx/ssl/cert.pem \
      -subj "/CN=h-forge/O=H-Forge/C=TH"
COPY dist/ /usr/share/nginx/html/
# Use sed to substitute __BACKEND_URL__ at container start (avoids envsubst conflicts)
COPY nginx.conf /etc/nginx/nginx.conf.tmpl
EXPOSE 8083
CMD sh -c "sed \
  -e 's|__BACKEND_URL__|'\"${BACKEND_URL:-http://localhost:8000}\"'|g' \
  -e 's|__LS_USER__|'\"${LS_USER:-admin@medimage.local}\"'|g' \
  -e 's|__LS_PASSWORD__|'\"${LS_PASSWORD:-}\"'|g' \
  /etc/nginx/nginx.conf.tmpl > /etc/nginx/conf.d/default.conf && nginx -g 'daemon off;'"