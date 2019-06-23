from pathlib import Path

import json_schema_to_class

current_dir: Path = Path(__file__).parent
json_schema_to_class.generate_dir(
    schema_dir=current_dir.parent / 'schema',
    output_dir=current_dir / 'build'
)

if __name__ != '__main__':
    from .build import *  # noqa: F403

    del json_schema_to_class
    del current_dir
    del Path
