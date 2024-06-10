import sys

def error(msg: str, exit_status: int = 1):
    """ Prints error to stderr and exits with failure code """
    print()
    print(f"ERROR: {msg}", file=sys.stderr)
    exit(exit_status)
