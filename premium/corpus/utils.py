import os 

class LocalData(object):
    """premium/localdata manager"""
    def __init__(self, file_name: str) -> None:
        self.file_name = file_name

    def fullpath(self) -> str:
        """Get path of file from "premium/localdata/" folder
        """
        current_dir = os.path.dirname(os.path.realpath(__file__))
        parent_dir = os.path.dirname(current_dir)
        return os.path.join(parent_dir, f'localdata/{self.file_name}')