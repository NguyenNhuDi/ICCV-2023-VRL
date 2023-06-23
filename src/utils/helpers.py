import numpy as np

def get_case_from_path(path:str) -> str:
    """
    Given a file path, extracts the case name.
    """
    if 'case' in path:
        return [name.split('_')[1] for name in path.split('/') if 'case' in name][0]
    else:
        return [name for name in path.split('/') if '0' in name or '1' in name][0]