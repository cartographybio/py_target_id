# utils/__init__.py
import os
import importlib

# Auto-import all .py files in this directory
_current_dir = os.path.dirname(__file__)
for filename in os.listdir(_current_dir):
    if filename.endswith('.py') and not filename.startswith('_'):
        module_name = filename[:-3]
        module = importlib.import_module(f'.{module_name}', package=__name__)
        # Import all public names
        for name in dir(module):
            if not name.startswith('_'):
                globals()[name] = getattr(module, name)