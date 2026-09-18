@echo off
REM ==========================================================
REM  AgentKnowledgeHub API - stop the instance on the API port
REM ==========================================================
setlocal enabledelayedexpansion

if "%API_PORT%"=="" (
  set PORT=8080
) else (
  set PORT=%API_PORT%
)

echo Stopping AgentKnowledgeHub API on port %PORT% ...

set FOUND=0
for /f "tokens=5" %%p in ('netstat -ano -p TCP ^| findstr "LISTENING" ^| findstr ":%PORT% "') do (
  set FOUND=1
  echo   killing PID %%p
  taskkill /PID %%p /F
  if errorlevel 1 (
    echo.
    echo   [FAILED] could not kill PID %%p
    echo   It is probably running elevated ^(an Administrator shell^).
    echo   Right-click Command Prompt -^> "Run as administrator", then:
    echo       taskkill /PID %%p /F
  )
)

if "!FOUND!"=="0" (
  echo   nothing is listening on port %PORT% - already stopped.
)

endlocal
