@echo off
REM ==========================================================
REM  AgentKnowledgeHub API startup script
REM  Project : resolved relative to this script (%~dp0python)
REM ==========================================================

cd /d "%~dp0python"

set PYTHONIOENCODING=utf-8

echo ========================================
echo   Starting AgentKnowledgeHub API
echo   URL  : http://127.0.0.1:8080
echo   Docs : http://127.0.0.1:8080/docs
echo   Key  : dev-key-1  (admin: dev-admin-key-1)
echo   Stop : stop-api.bat
echo   Press Ctrl+C to stop
echo ========================================
echo.
echo   NOTE: No external services required.
echo         Embedded Chroma + SQLite - no Docker, no external DB.
echo.
echo   If the port is already taken, this script reports who holds it
echo   instead of failing after half-initialising. See docs/runbook.md
echo.

python -m api.main
