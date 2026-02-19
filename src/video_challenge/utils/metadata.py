import os
from pathlib import Path

def list_parquet_files(dir: Path, prefix: str = None) -> list[str]:
    """
    Retrieves a list of all parquet filenames in a specified directory.
    If "prefix" is specified, the list will only contain the filenames that start
    with "prefix".

    Args:
        dir (Path): 
            The path to the directory containing the parquet files.
        prefix (str, optional): 
            The prefix that the filenames must start with. Default = None.
    Returns:
        list[str]: 
            A list of parquet filenames that start with 'prefix'.
    """
    allFiles = os.listdir(dir)
    all_parquets = [file for file in allFiles if file.endswith('.parquet')]
    if prefix is not None:
        return [file for file in all_parquets if file.startswith(prefix)]
    else:
        return all_parquets