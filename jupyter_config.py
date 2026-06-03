c.ServerApp.base_url = '/jupyter/'
c.ServerApp.allow_origin = '*'
c.ServerApp.disable_check_xsrf = True
c.ServerApp.tornado_settings = {
    'headers': {
        'Content-Security-Policy': "frame-ancestors *",
        'X-Frame-Options': '',
    }
}
