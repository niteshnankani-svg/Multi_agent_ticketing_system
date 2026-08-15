# Backend & Server Troubleshooting Guide

## 500 Internal Server Errors
A 500 error usually means the server encountered an unexpected problem.
First step: try again after 60 seconds, as this often resolves temporary
load issues. If it persists, check our status page for ongoing incidents.

## Connection Timeouts
Connection timeouts are typically caused by network congestion or a
temporary server overload. Clearing your browser cache and retrying
usually resolves this. If using an API, requests should include a
retry with exponential backoff.

## File Upload Failures
Uploads fail most commonly due to file size limits (max 25MB per file)
or unsupported file types (we support PDF, CSV, PNG, JPG, DOCX).
Large files should be compressed before uploading.

## Login / Authentication Issues
If login fails repeatedly, this is often due to an expired session
token. Logging out completely and back in resolves most cases.
Persistent login failures across multiple devices may indicate
an account security lock, which requires manual review.

## Database Connection Errors
These are typically transient and resolve within a few minutes.
If a database error persists beyond 15 minutes, it likely indicates
an active incident, and the status page should be checked immediately.

## Planned Maintenance
Scheduled maintenance windows are announced 48 hours in advance via
email and the status page. Service may be intermittently unavailable
during these windows, typically between 2-4 AM UTC.
