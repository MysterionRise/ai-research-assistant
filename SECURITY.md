# Security Policy

## Supported Versions

The following versions of ARIA are currently being supported with security updates:

| Version | Supported          |
| ------- | ------------------ |
| 0.1.x   | :white_check_mark: |

## Reporting a Vulnerability

We take the security of ARIA seriously. If you believe you have found a security vulnerability, please report it to us as described below.

### How to Report

**Please do not report security vulnerabilities through public GitHub issues.**

Instead, please report them via email to:

**security@aria-research.ai** (or your designated security email)

### What to Include

Please include the following information in your report:

- Type of vulnerability (e.g., SQL injection, XSS, authentication bypass)
- Full paths of source file(s) related to the vulnerability
- Location of the affected source code (tag/branch/commit or direct URL)
- Step-by-step instructions to reproduce the issue
- Proof-of-concept or exploit code (if possible)
- Impact of the vulnerability and how an attacker might exploit it

### Response Timeline

- **Initial Response**: Within 48 hours of receiving your report
- **Status Update**: Within 7 days with an assessment of the vulnerability
- **Resolution Timeline**: Based on severity:
  - Critical: 7 days
  - High: 14 days
  - Medium: 30 days
  - Low: 90 days

### What to Expect

- We will acknowledge receipt of your vulnerability report
- We will send you regular updates about our progress
- We will notify you when the vulnerability is fixed
- We will publicly acknowledge your responsible disclosure (unless you prefer to remain anonymous)

### Scope

The following are in scope for security reports:

- ARIA core application (`src/aria/`)
- API endpoints and authentication
- Data handling and storage
- Third-party integrations
- Infrastructure configurations

### Out of Scope

- Vulnerabilities in third-party dependencies (report directly to maintainers)
- Social engineering attacks
- Denial of service attacks
- Physical security

## Security Best Practices

When deploying ARIA:

1. **Never commit secrets** - Use environment variables and secret management tools
2. **Enable audit logging** - Required for 21 CFR Part 11 compliance
3. **Use TLS everywhere** - Encrypt all data in transit
4. **Rotate API keys regularly** - Implement key rotation policies
5. **Limit network access** - Use firewalls and VPCs appropriately

## Acknowledgments

We appreciate the security research community and will acknowledge all valid reports.
