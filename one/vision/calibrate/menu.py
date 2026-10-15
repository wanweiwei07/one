"""The step menu a camera's ``calib.py`` shows when it is run: numbered
steps, one is picked on the console and run."""


def run(title, steps):
    """``steps``: ``[(label, fn), ...]``, ``fn() -> bool``. Lists them, runs
    the one picked; Enter quits. True when the step succeeded."""
    print(title)
    for k, (label, _) in enumerate(steps, 1):
        print(f'  {k}  {label}')
    while True:
        ans = input('step? number, Enter quits: ').strip()
        if not ans:
            return True
        if ans.isdigit() and 1 <= int(ans) <= len(steps):
            return bool(steps[int(ans) - 1][1]())
