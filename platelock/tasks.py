"""User-facing repair intentions; old workflows retain their explicit settings."""
TASKS = (
    'Advanced settings', 'Correct framing', 'Repair timing',
    'Repair timing + framing', 'Reconstruct missing footage (experimental)',
)


def resolve_task(task, method, position):
    if task not in TASKS:
        raise ValueError(f'Unknown repair task: {task}')
    return {
        'Correct framing': ('None', 'Correct framing (scale + position)'),
        'Repair timing': ('RIFE', 'Measure only'),
        'Repair timing + framing': ('RIFE', 'Correct framing (scale + position)'),
        'Reconstruct missing footage (experimental)': ('LTX', 'Measure only'),
    }.get(task, (method, position))


def require_framing_only_plan(task, summary):
    """Legacy guard for a renderer that would apply the proposed timing plan.

    Studio's frame-preserving framing-only renderer does not use that plan.
    """
    if task != 'Correct framing':
        return
    changes = (summary['interpolated'] or summary['missing_plate_frames'] or
               summary['dropped_gen_frames'] or summary.get('offset_median', 0))
    if changes:
        raise ValueError('This shot also needs timing edits. Choose Repair timing + framing; '
                         'Correct framing never silently changes the frame sequence.')
