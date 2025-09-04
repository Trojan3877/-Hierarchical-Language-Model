# 🔒 Security Policy

## Supported Versions
We use [Semantic Versioning](https://semver.org/).  
The following versions are currently supported with security updates:

| Version | Supported |
|---------|-----------|
| 1.x.x   | ✅ |
| < 1.0   | ❌ |

---

## Reporting a Vulnerability
If you discover a security vulnerability in this project, please help us by responsibly disclosing it:

1. **Do not** open a public GitHub issue describing the vulnerability.
2. Instead, email the maintainer directly at: **corey.leath@example.com** (replace with your real contact).
3. Include:
   - A detailed description of the vulnerability
   - Steps to reproduce it
   - Any suggested fixes or mitigations

We will:
- Acknowledge receipt within **48 hours**
- Provide a status update within **7 days**
- Release a fix or mitigation plan as quickly as possible

---

## Security Best Practices
To keep deployments secure:
- Always use the latest stable release
- Run containers in isolated environments
- Use environment variables (`.env`) for secrets, never commit them
- Rotate API keys regularly
- Apply principle of least privilege for user roles

---

## Scope
This project is for **educational and research purposes**.  
It is **not intended for production use without additional security hardening**, such as:
- Authentication & authorization layers on the API
- HTTPS/TLS setup
- Cloud IAM role management
