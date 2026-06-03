c.ServerApp.base_url = '/jupyter/'
c.ServerApp.allow_origin = '*'
c.ServerApp.disable_check_xsrf = True
c.IdentityProvider.token = 'medimage2026'
c.ServerApp.password = ''
c.ServerApp.password_required = False
c.ServerApp.tornado_settings = {
    'headers': {
        'Content-Security-Policy': "frame-ancestors *",
        'X-Frame-Options': '',
    }
}
