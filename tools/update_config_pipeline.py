import re

cfg_path = 'projects/BEVFusion/configs/bevfusion_lidar-cam-radar_geometry_enhanced.py'
with open(cfg_path, 'r') as f:
    content = f.read()

# The block to find and replace:
# dict(
#     [...]
#     type='LoadRadarPointsFromFile',
#     [...]
#     type='LoadRadarPointsFromMultiSweeps',
#     [...])

# It's better to use regex to find dict(backend_args=..., coord_type='LIDAR', load_dim=18, type='LoadRadarPointsFromFile'...) up to sweeps_num=5...
block_pattern = re.compile(
    r"\s*dict\(\s*(backend_args=None,\s*)?coord_type='LIDAR',\s*load_dim=18,\s*type='LoadRadarPointsFromFile',\s*use_dim=\[\s*0,\s*1,\s*2,\s*5,\s*8,\s*9,\s*16,\s*17,?\s*\]\),\s*dict\(\s*backend_args=None,\s*close_radius=1\.0,\s*load_dim=18,\s*pad_empty_sweeps=True,\s*remove_close=True,\s*sweeps_num=5,\s*(test_mode=True,\s*)?type='LoadRadarPointsFromMultiSweeps',\s*use_dim=\[\s*0,\s*1,\s*2,\s*5,\s*8,\s*9,\s*16,\s*17,?\s*\]\),", 
    re.MULTILINE
)

# New block preserving the indent of the match but simpler
def replacer(match):
    # Find the identation of the first matched 'dict('
    text = match.group(0)
    indent = text[:len(text) - len(text.lstrip('\n\r\t '))]
    indent = indent.replace('\n', '').replace('\r', '')
    if not indent:
        indent = "\n    "
        
    new_block = (f"{indent}dict(\n"
                 f"{indent}    type='LoadPreprocessedRadarPoints',\n"
                 f"{indent}    data_root='data/nuscenes',\n"
                 f"{indent}    preprocessed_dir='radar_multisweep_preprocessed',\n"
                 f"{indent}    coord_type='LIDAR',\n"
                 f"{indent}    use_dim=[0, 1, 2, 5, 8, 9, 16, 17]),")
    return new_block

new_content = block_pattern.sub(replacer, content)

with open(cfg_path, 'w') as f:
    f.write(new_content)

print(f"Replaced {len(block_pattern.findall(content))} occurrences.")
