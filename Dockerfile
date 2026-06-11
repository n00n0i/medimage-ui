FROM nginx:alpine
RUN apk add --no-cache openssl && \
    mkdir -p /etc/nginx/ssl && \
    openssl req -x509 -nodes -days 3650 -newkey rsa:2048 \
      -keyout /etc/nginx/ssl/key.pem \
      -out /etc/nginx/ssl/cert.pem \
      -subj "/CN=100.68.3.42" \
      -addext "subjectAltName=IP:100.68.3.42,DNS:localhost,DNS:h-forge"
COPY dist/ /usr/share/nginx/html/
COPY nginx.conf /etc/nginx/nginx.conf.tmpl
EXPOSE 8083
EXPOSE 80
CMD sh -c "sed \
  -e 's|__BACKEND_URL__|'\"${BACKEND_URL:-http://localhost:8000}\"'|g' \
  -e 's|__LS_USER__|'\"${LS_USER:-admin@medimage.local}\"'|g' \
  -e 's|__LS_PASSWORD__|'\"${LS_PASSWORD:-}\"'|g' \
  /etc/nginx/nginx.conf.tmpl > /etc/nginx/conf.d/default.conf && \
  sed -e 's|listen 8083 ssl;|listen 80;|g' \
      -e '/error_page 497/d' \
      -e '/ssl_certificate/d' \
      -e '/ssl_certificate_key/d' \
      -e '/ssl_protocols/d' \
      -e '/ssl_ciphers/d' \
  /etc/nginx/conf.d/default.conf > /etc/nginx/conf.d/80-http.conf && \
  nginx -g 'daemon off;'"