#!/usr/bin/env python
try:
    import yaml
except ImportError:
    print('Please install yaml')
    
class Struct:
    def __init__(self, **entries):
        self.__dict__.update(entries)
    
    def __str__(self) -> str:
        return str(self.__dict__)
        
def load_yaml(path)->Struct:
    with open(path) as f:
        return Struct(**yaml.safe_load(f))
        