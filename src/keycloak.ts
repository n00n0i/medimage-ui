import Keycloak from 'keycloak-js'

// Keycloak is proxied through nginx at /kc — same origin, no mixed-content issues
const keycloak = new Keycloak({
  url: `${window.location.origin}/kc`,
  realm: 'h-forge',
  clientId: 'h-forge-ui',
})

export default keycloak
