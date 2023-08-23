import json
import shutil
import os
import click
import simplejson as json
import yaml

def read(file: str) -> map:
    """
    Read a given file as a map
    """
    if 'yml' in file:
        with open(file) as file:
            return yaml.safe_load(file)
    elif 'json' in file:
        with open(file) as file:
            return json.load(file)
    else:
        raise NameError("extension not supported. womp womp")

def dump(content, file: str, **kwargs) -> None:
    """
    Dump given map
    """
    if 'yml' in file:
        with open(file, 'w+') as file:
            yaml.dump(content, file, **kwargs)
    elif 'json' in file:
        with open(file, 'w+') as file:
            json.dump(content, file, **kwargs)

@click.command()
@click.option('--working_dir', help="Output folder for json files architecture", required=True)
@click.option('--outlines', help="Json outlines. Ex: a.json b.json", multiple=True, required=True)
@click.option('--replacement_combo', help="Start with the replacement, followed by options. Ex: 'version [b4, b5, b6]'", required=True)
def main(working_dir: str, outlines: list[str], replacement_combo: str) -> None:
    output_root = f"model_definitions/{working_dir}"
    to_replace = replacement_combo[:replacement_combo.find(' ')]
    replacement_options = eval(replacement_combo[replacement_combo.find(' '):])
    
    outline_loaded = []
    for outline in outlines:
        outline_loaded.append(read(outline))
    
    for option in replacement_options:
        #Generate files with this replacement
        for path, c_outline in zip(outlines, outline_loaded):
            #Do for all outlines
            c_outline = str(c_outline)
            c_outline = c_outline.replace(f"*{to_replace}*", str(option))
            #get name from path
            new_name = path.split('/')[-1].split('.')[0]
            new_name = f"{new_name}_{option}"
            #get extension from path
            extension = path.split('.')[-1]
            dump(eval(c_outline), f"{output_root}/{new_name}.{extension}", indent=4)


if __name__ == "__main__":
    main()