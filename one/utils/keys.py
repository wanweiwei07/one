"""Single-key prompts for the step-by-step tools: 'g' to go, 'q' / Esc /
Ctrl+C to quit -- read straight off the console, no Enter.
"""

_QUIT_KEYS = ('q', '\x1b', '\x03')     # q, Esc, Ctrl+C

try:
    import msvcrt

    def _read_key():
        return msvcrt.getwch().lower()

    def _quit_pressed():
        """Non-blocking: True when q / Esc / Ctrl+C is waiting."""
        while msvcrt.kbhit():
            if msvcrt.getwch().lower() in _QUIT_KEYS:
                return True
        return False
except ImportError:                     # not Windows: fall back to a line
    def _read_key():
        return input().strip().lower()[:1]

    def _quit_pressed():
        return False                    # Ctrl+C still raises there


def _wait_for_g(prompt):
    """True on 'g', False on 'q' / Esc / Ctrl+C. Anything else is ignored.

    ``msvcrt.getwch`` hands Ctrl+C over as a plain '\\x03' instead of raising
    KeyboardInterrupt, so it is caught here explicitly.
    """
    print(f"\n{prompt} -- 'g' to run, 'q' to quit", flush=True)
    while True:
        key = _read_key()
        if key == 'g':
            return True
        if key in _QUIT_KEYS:
            return False
