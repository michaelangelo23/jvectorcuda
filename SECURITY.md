# Security Policy

## Supported Versions

| Version | Supported          |
| ------- | ------------------ |
| 1.0.x   | :white_check_mark: |

## Reporting a Vulnerability

If you discover a security vulnerability in JVectorCUDA, please report it responsibly:

1. **Do NOT open a public GitHub issue** for security vulnerabilities
2. Email details to the maintainer (or use GitHub's private vulnerability reporting)
3. Include:
   - Description of the vulnerability
   - Steps to reproduce
   - Potential impact
   - Suggested fix (if any)

## Response Timeline

- **Initial response**: Within 48 hours
- **Status update**: Within 7 days
- **Fix timeline**: Depends on severity (critical: ASAP, high: 2 weeks, medium: 1 month)

## Security Measures

This project uses:

- **GitHub CodeQL** - Static analysis for security vulnerabilities
- **Dependabot** - Automated dependency updates
- **Snyk** (planned) - Native library vulnerability scanning

## Known Security Considerations

### JNI/Native Code
JVectorCUDA uses JCuda which loads native CUDA libraries. This requires:
- `--enable-native-access=ALL-UNNAMED` JVM flag
- Trusted CUDA installation from NVIDIA

### Memory Safety
GPU memory operations use native pointers. The library includes:
- Bounds checking before GPU memory access
- Proper resource cleanup in `close()` methods with try-catch for each resource
- CUDA error checking with descriptive exceptions via `checkCudaResult()`
- Null validation for vector arrays before processing
- Input validation for NaN/Infinity values

## Security Updates

Security updates will be released as patch versions (e.g., 1.0.1) and announced via GitHub releases.
