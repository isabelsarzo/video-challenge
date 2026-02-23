import os

def parse_segment_name(segment_name: str):
    """
    Parse a segment name of the form:
    child_xx_yy.npy

    Args
    - segment_name : str (e.g. "child_xx_yy.npy")
    
    Returns
    - child_n : str (e.g. "child_xx")
    - segment_idx : int (e.g. yy)
    """
    # Remove extension
    name = segment_name.replace(".npy", "")
    parts = name.split("_")

    segment_idx = int(parts[-1])
    child_n = "_".join(parts[:-1])

    return child_n, segment_idx
