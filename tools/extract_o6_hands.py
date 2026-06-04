"""One-off: carve the lh_/rh_ LinkerHand O6 subtrees out of the L1 h0602.urdf
into two standalone EE URDFs (o6_left.urdf / o6_right.urdf).

- root link of each hand becomes lh_hand_base_link / rh_hand_base_link
- the mount joint (left/right_linkerhand_mount_joint) is dropped (the mount
  pose lives in code now, via MechBase.mount)
- mesh paths rewritten from ../meshes/o6/<side>/X -> ../meshes/<side>/X
"""
import os
import xml.etree.ElementTree as ET

HERE = os.path.dirname(os.path.abspath(__file__))
SRC = os.path.join(HERE, '..', 'one', 'robots', 'humanoids', 'linx', 'l1',
                   'urdf', 'h0602.urdf')
OUT_DIR = os.path.join(HERE, '..', 'one', 'robots', 'end_effectors',
                       'linkerhand', 'o6', 'urdf')


def extract(prefix, side):
    tree = ET.parse(SRC)
    root = tree.getroot()  # <robot>
    keep = []
    # materials for this hand
    for el in root.findall('material'):
        if el.get('name', '').startswith(prefix):
            keep.append(el)
    # links belonging to this hand
    for el in root.findall('link'):
        if el.get('name', '').startswith(prefix):
            keep.append(el)
    # joints whose child is one of this hand's links (drops the mount joint,
    # whose parent is *_arm_link_6 and child is *_hand_base_link -- but its
    # child DOES start with prefix, so exclude mount joints explicitly).
    for el in root.findall('joint'):
        child = el.find('child')
        if child is not None and child.get('link', '').startswith(prefix):
            if el.get('type') == 'fixed' and 'mount' in el.get('name', ''):
                continue
            keep.append(el)
    # rewrite mesh paths
    for el in keep:
        for mesh in el.iter('mesh'):
            fn = mesh.get('filename', '')
            mesh.set('filename', fn.replace(f'../meshes/o6/{side}/',
                                            f'../meshes/{side}/'))
    new_robot = ET.Element('robot', {'name': f'linkerhand_o6_{side}'})
    new_robot.extend(keep)
    ET.indent(new_robot, space='  ')
    out = os.path.join(OUT_DIR, f'o6_{side}.urdf')
    ET.ElementTree(new_robot).write(out, encoding='utf-8', xml_declaration=True)
    n_lnk = sum(1 for e in keep if e.tag == 'link')
    n_jnt = sum(1 for e in keep if e.tag == 'joint')
    print(f'{side}: wrote {out}  ({n_lnk} links, {n_jnt} joints)')


extract('lh_', 'left')
extract('rh_', 'right')
