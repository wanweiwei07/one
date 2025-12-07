import os
import re
import xacro
from urdf_parser_py.urdf import URDF
from xacro import substitution_args

def auto_scan_packages(main_file, search_root):
    """
    Scan main_file and auto-detect "$(find pkg)" patterns,
    then search under search_root for matching folders.
    Returns: dict pkg_name → path
    """
    # Read xacro text
    with open(main_file, "r") as f:
        text = f.read()
    # Regex: match $(find xxx)
    pkg_names = set(re.findall(r"\$\(find\s+([^\)]+)\)", text))
    mapping = {}
    # Search directory recursively
    for root, dirs, files in os.walk(search_root):
        for pkg in pkg_names:
            if pkg in root:
                mapping[pkg] = root.replace("\\", "/")
                return mapping

def load_robot_from_xacro(main_file, base_dir=None, mappings=None):
    """
    Load a robot URDF from a xacro or urdf file.
    :param main_file: path to .xacro or .urdf
    :param paramers: paramers mappings, e.g. {"model": "cobotta"}
    """

    # absolute path
    main_file = os.path.abspath(main_file)
    if base_dir is None:
        raise RuntimeError("Package base_dir must be specified for loading xacro files.")
    known_packages = auto_scan_packages(main_file, base_dir)
    # local resolver for $(find ...)
    def eval_find(pkg_name):
        if pkg_name in known_packages:
            return known_packages[pkg_name]
        if ("control" in pkg_name
                or "gazebo" in pkg_name
                or pkg_name.endswith("_moveit_config")):
            return ""
        raise RuntimeError(f"Unknown ROS package in $(find): {pkg_name}")

    # override xacro's find
    substitution_args._eval_find = eval_find
    if main_file.endswith(".xacro"):
        doc = xacro.process_file(main_file, mappings=mappings or {})
        xml = doc.toxml()
    else:
        xml = open(main_file).read()
    return URDF.from_xml_string(xml)
