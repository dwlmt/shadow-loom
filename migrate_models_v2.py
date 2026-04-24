"""
Migration script v2: Convert old Location format to new format.
Uses AST-aware parsing via bracket counting instead of regex.
"""
import re
import os

MODEL_DIR = "tests/test_plot_models"


def find_matching_close(content, start_pos, open_char='(', close_char=')'):
    """Find the matching closing bracket position."""
    depth = 0
    for i in range(start_pos, len(content)):
        if content[i] == open_char:
            depth += 1
        elif content[i] == close_char:
            depth -= 1
            if depth == 0:
                return i
    return -1


def extract_location_blocks(content):
    """Extract all Location(...) blocks with their dict keys."""
    blocks = []
    pattern = re.compile(r'"(LOC_[^"]+)":\s*Location\(')
    for match in pattern.finditer(content):
        loc_id = match.group(1)
        paren_start = content.index('(', match.start() + len(match.group(1)))
        paren_end = find_matching_close(content, paren_start)
        if paren_end == -1:
            continue
        # Include possible trailing comma
        trail_end = paren_end + 1
        if trail_end < len(content) and content[trail_end] == ',':
            trail_end += 1
        
        full_start = match.start()
        full_text = content[full_start:trail_end]
        blocks.append({
            'loc_id': loc_id,
            'start': full_start,
            'end': trail_end,
            'text': full_text,
            'inner': content[paren_start+1:paren_end],
        })
    return blocks


def parse_location_fields(inner_text):
    """Parse fields from inside Location(...)."""
    # name
    name_match = re.search(r'name="([^"]*)"', inner_text)
    name = name_match.group(1) if name_match else "Unknown"
    
    # connected_locations
    conn_match = re.search(r'connected_locations=\[([^\]]*)\]', inner_text)
    connected = []
    if conn_match:
        connected = re.findall(r'"(LOC_[^"]+)"', conn_match.group(1))
    
    # ambient_states with AmbientVector
    ambient_dict = {}
    amb_pattern = re.compile(r'"(\w+)":\s*AmbientVector\(value=([\d.]+),\s*volatility=([\d.]+)\)')
    for am in amb_pattern.finditer(inner_text):
        ambient_dict[am.group(1)] = (am.group(2), am.group(3))
    
    # constants
    constants = []
    const_match = re.search(r'constants=\[([^\]]*)\]', inner_text)
    if const_match and const_match.group(1).strip():
        constants = re.findall(r'"([^"]*)"', const_match.group(1))
    
    return name, connected, ambient_dict, constants


def build_new_location(loc_id, name, ambient_dict, constants, indent):
    """Build new Location definition string."""
    inner = indent + "    "
    
    desc = name
    if constants:
        desc = f"{name} ({', '.join(constants)})"
    
    # Build ambient_state string
    if ambient_dict:
        items = []
        for k, (val, vol) in ambient_dict.items():
            items.append(f'"{k}": {{"value": {val}, "volatility": {vol}}}')
        ambient_str = "{" + ", ".join(items) + "}"
    else:
        ambient_str = "{}"
    
    return (
        f'{indent}"{loc_id}": Location(\n'
        f'{inner}name="{name}",\n'
        f'{inner}description="{desc}",\n'
        f'{inner}ambient_state={ambient_str},\n'
        f'{indent}),'
    )


def migrate_file(filepath):
    with open(filepath, 'r') as f:
        content = f.read()
    
    # 1. Update imports
    content = content.replace(
        "    WorldStateV1, Location, NarrativeObject, Entity, EventNode,\n"
        "    CausalEdge, RelationshipEdge, TraitVector, AmbientVector,\n"
        "    Affordance, Belief,",
        "    WorldStateV1, Location, NarrativeObject, Entity, EventNode,\n"
        "    CausalEdge, SpatialEdge, RelationshipEdge, TraitVector,\n"
        "    Affordance, Belief,"
    )
    
    # 2. Extract all Location blocks
    blocks = extract_location_blocks(content)
    
    # 3. Collect spatial edges
    spatial_edges = set()
    for block in blocks:
        name, connected, ambient_dict, constants = parse_location_fields(block['inner'])
        for tgt in connected:
            pair = tuple(sorted([block['loc_id'], tgt]))
            spatial_edges.add(pair)
    
    # 4. Replace Location blocks (reverse order to preserve positions)
    for block in reversed(blocks):
        name, connected, ambient_dict, constants = parse_location_fields(block['inner'])
        
        # Detect indentation from original text
        line_start = content.rfind('\n', 0, block['start']) + 1
        indent = content[line_start:block['start']]
        if not indent.strip() == '':
            indent = "        "  # default 2-level indent
        
        new_loc = build_new_location(block['loc_id'], name, ambient_dict, constants, indent)
        content = content[:block['start']] + new_loc + content[block['end']:]
    
    # 5. Add spatial_topology before social_topology
    if spatial_edges:
        se_lines = ["    spatial_topology=["]
        for src, tgt in sorted(spatial_edges):
            se_lines.append(f'        SpatialEdge(source_id="{src}", target_id="{tgt}"),')
        se_lines.append("    ],")
        se_lines.append("")
        se_block = "\n".join(se_lines)
        
        content = content.replace(
            "    social_topology=[",
            se_block + "    social_topology=["
        )
    
    with open(filepath, 'w') as f:
        f.write(content)
    
    print(f"Migrated: {filepath} ({len(spatial_edges)} spatial edges, {len(blocks)} locations)")


if __name__ == "__main__":
    for fname in sorted(os.listdir(MODEL_DIR)):
        if fname.endswith('.py') and fname != '__init__.py':
            fpath = os.path.join(MODEL_DIR, fname)
            migrate_file(fpath)
    print("Done!")
