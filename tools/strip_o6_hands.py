"""One-off: strip the Linkerbot O6 hands out of the L1 h0602 body URDF.

Removes every <link>/<joint>/<material> named lh_*/rh_* plus the two
*_linkerhand_mount_joint fixed joints, leaving the body ending at
left_arm_link_6 / right_arm_link_6 (the new flange tips). The hands now live
as standalone EEs under end_effectors/linkerbot/o6 and are mounted in code.
"""
import os
import shutil
import xml.etree.ElementTree as ET

HERE = os.path.dirname(os.path.abspath(__file__))
URDF = os.path.join(HERE, '..', 'one', 'robots', 'humanoids', 'linx', 'l1',
                    'urdf', 'h0602.urdf')


def is_hand(name):
    return name.startswith('lh_') or name.startswith('rh_')


tree = ET.parse(URDF)
root = tree.getroot()

removed = {'material': 0, 'link': 0, 'joint': 0}
for tag in ('material', 'link', 'joint'):
    for el in list(root.findall(tag)):
        name = el.get('name', '')
        drop = is_hand(name)
        if tag == 'joint' and 'linkerhand_mount_joint' in name:
            drop = True
        if drop:
            root.remove(el)
            removed[tag] += 1

# back up once, then overwrite
bak = URDF + '.with_hands.bak'
if not os.path.exists(bak):
    shutil.copy2(URDF, bak)
ET.indent(root, space='  ')
tree.write(URDF, encoding='utf-8', xml_declaration=True)
print('removed:', removed)
print('backup:', os.path.abspath(bak))
