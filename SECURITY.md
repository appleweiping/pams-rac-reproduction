# Security and private infrastructure

Do not report private credentials, dataset access tokens or server passwords
in a public issue. Use the repository owner's private coordination channel.

The repository intentionally stores no SSH host, username, password or private
key. Server inventory is captured in run manifests only as non-secret hardware
and software properties. Any suspected credential exposure should be reported
privately to the repository owner and removed from Git history before release.
