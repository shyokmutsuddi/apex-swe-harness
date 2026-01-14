# Security Policy

## Supported Versions

| Version | Supported          |
| ------- | ------------------ |
| 1.0.x   | :white_check_mark: |

## Reporting a Vulnerability

We take the security of APEX SWE Harness seriously. If you discover a security vulnerability, please follow these steps:

### How to Report

1. **DO NOT** open a public issue
2. Email security details to: [your-security-email@example.com]
3. Include:
   - Description of the vulnerability
   - Steps to reproduce
   - Potential impact
   - Suggested fix (if any)

### What to Expect

- **Acknowledgment**: Within 48 hours
- **Initial Assessment**: Within 1 week
- **Status Updates**: Every 2 weeks until resolved
- **Resolution**: Timeline depends on severity and complexity

### Disclosure Policy

- We follow coordinated disclosure
- Security advisories published after fix is available
- Credit given to reporters (unless anonymity requested)

## Security Best Practices

When using APEX SWE Harness:

### API Keys and Secrets

- **Never commit** API keys or secrets to version control
- Use environment variables for all credentials
- Rotate API keys regularly
- Use different keys for dev/staging/production

### Docker Security

- Keep Docker daemon updated
- Use minimal base images
- Scan containers for vulnerabilities
- Limit container privileges

### Task Execution

- Review task definitions before execution
- Run in isolated environments when possible
- Monitor resource usage
- Set appropriate timeouts

### Network Security

- Use HTTPS for all API communications
- Validate SSL certificates
- Implement rate limiting
- Monitor for unusual activity

## Known Security Considerations

### Task Execution Environment

Tasks run with access to:
- Docker daemon
- Network resources
- File system (within containers)

**Mitigation**: Use dedicated execution environments and network isolation

### API Credentials

Multiple AI model providers require API keys:
- Claude (Anthropic)
- Gemini (Google)
- XAI
- Fireworks

**Mitigation**: Use environment variables and secret management tools

### Generated Artifacts

Status CSV files and reports may contain:
- Task names
- Execution times
- Error messages

**Mitigation**: Store artifacts in secure locations and review before sharing

## Security Updates

Security updates are released as:
- Patch versions (1.0.x) for critical issues
- Minor versions (1.x.0) for important updates
- Security advisories for all vulnerabilities

Subscribe to repository releases for notifications.
