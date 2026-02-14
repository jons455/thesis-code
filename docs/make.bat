@echo off
REM Minimal makefile for Sphinx documentation on Windows

set SPHINXOPTS=
set SPHINXBUILD=sphinx-build
set SOURCEDIR=source
set BUILDDIR=build

if "%1"=="" goto help
if "%1"=="help" goto help
if "%1"=="html" goto html
if "%1"=="clean" goto clean
goto :eof

:help
	@echo.
	@echo Available targets:
	@echo   html    - Build HTML documentation
	@echo   clean   - Remove build directory
	@echo   help    - Show this help message
	@echo.
	goto :eof

:html
	%SPHINXBUILD% -b html %SPHINXOPTS% %SOURCEDIR% %BUILDDIR%\html
	@echo.
	@echo Build finished. The HTML pages are in %BUILDDIR%\html.
	goto :eof

:clean
	if exist %BUILDDIR% rmdir /s /q %BUILDDIR%
	@echo Clean finished.
	goto :eof
