# Codes by Vision
import os

def display_banner():
    banner = """
\033[1;36m██████╗ ███████╗███████╗██████╗ ████████╗██████╗  █████╗  ██████╗███████╗
██╔══██╗██╔════╝██╔════╝██╔══██╗╚══██╔══╝██╔══██╗██╔══██╗██╔════╝██╔════╝
██║  ██║█████╗  █████╗  ██████╔╝   ██║   ██████╔╝███████║██║     █████╗  
██║  ██║██╔══╝  ██╔══╝  ██╔═══╝    ██║   ██╔══██╗██╔══██║██║     ██╔══╝  
██████╔╝███████╗███████╗██║        ██║   ██║  ██║██║  ██║╚██████╗███████╗
╚═════╝ ╚══════╝╚══════╝╚═╝        ╚═╝   ╚═╝  ╚═╝╚═╝  ╚═╝ ╚═════╝╚══════╝\033[0m

\033[1;37mDeepTrace: Forensic Tracing & Deepfake Detection\033[0m
\033[1;32mMade by Vision | GitHub: vision-dev1\033[0m
"""
    print(banner)

def print_status(message):
    print(f"[\033[1;34m*\033[0m] {message}")

def print_success(message):
    print(f"[\033[1;32m+\033[0m] {message}")

def print_error(message):
    print(f"[\033[1;31m!\033[0m] {message}")

def print_warning(message):
    print(f"[\033[1;33m-\033[0m] {message}")
