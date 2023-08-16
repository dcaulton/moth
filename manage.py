#!/usr/bin/env python
"""Django's command-line utility for administrative tasks."""
import os
import sys


def main():
    if len(sys.argv) >=2 and sys.argv[1] == 'test':
        os.environ.setdefault("DJANGO_SETTINGS_MODULE", "server.test_settings")
    else:
        os.environ.setdefault("DJANGO_SETTINGS_MODULE", "server.settings")
    os.environ.setdefault("OMP_THREAD_LIMIT", "1")
    try:
        from django.core.management import execute_from_command_line
    except ImportError as exc:
        raise ImportError(
            "Couldn't import Django. Are you sure it's installed and "
            "available on your PYTHONPATH environment variable? Did you "
            "forget to activate a virtual environment?"
        ) from exc
    execute_from_command_line(sys.argv)


if __name__ == "__main__":
    main()