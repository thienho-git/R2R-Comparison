"""
Read the configuration of the reference script (original/xs_test_r2r_optimization_for_target_matrix.py) without
running it: its configr2r class and the `case` of the selected geometry are extracted with ast and executed. Only the
configuration statements run (no vamtoolbox, no torch, no file access).

Geometry: the script runs the case given as its first argument (default "tpms_diamond_skel0"). Here the geometry is
the one set by --geometry (setGeometry), else that default; run.py passes it to the script, so both use the same one.
"""

import ast

import numpy as np

from paths import REFERENCE_SCRIPT

_geometry = None #set by --geometry (setGeometry); None = the script's default


def _parse(script):
    return ast.parse(script.read_text(encoding="utf-8"))


def _cases(tree):
    match = next(n for n in tree.body if isinstance(n, ast.Match))
    return {c.pattern.value.value: c for c in match.cases if isinstance(c.pattern, ast.MatchValue)}


def defaultGeometry(script=REFERENCE_SCRIPT):
    """The script's default geometry: default of its parser.add_argument("geometry", ...)."""
    call = next(n for n in ast.walk(_parse(script)) if isinstance(n, ast.Call) and getattr(n.func, "attr", None) == "add_argument"
                and n.args and getattr(n.args[0], "value", None) == "geometry")
    return next(k.value.value for k in call.keywords if k.arg == "default")


def geometries(script=REFERENCE_SCRIPT):
    """The geometry (case) names of the script."""
    return list(_cases(_parse(script)))


def activeGeometry(script=REFERENCE_SCRIPT):
    return _geometry or defaultGeometry(script)


def setGeometry(name):
    """Select a geometry (case) of the script for this process; fails on an unknown name."""
    global _geometry
    if name is not None and name not in geometries():
        raise ValueError(f"Geometry '{name}' is not a case of the reference script. Choose one of: {geometries()}.")
    _geometry = name


def _activeCase(tree, script):
    geometry = activeGeometry(script)
    cases = _cases(tree)
    if geometry not in cases:
        raise ValueError(f"Geometry '{geometry}' is not a case of the reference script. Choose one of: {list(cases)}.")
    return geometry, cases[geometry]


def _isAssignTo(statement, name):
    return isinstance(statement, ast.Assign) and any(getattr(t, "id", None) == name for t in statement.targets)


def referenceSpec(script=REFERENCE_SCRIPT):
    """
    Returns (config, tpms) for the reference script's active geometry:
        config : configr2r instance (alpha_mat, n_sur, n_mat, r_i, tau, l_max, deg_inc, f_rho, f_l, N_rho, N_l, N_z, ...)
        tpms   : dict(unit_cell_size, t, kind, unsupported) of its vam.util.buildtargetsR2R.tpms(...) target, or None
                 when the target is not a TPMS (e.g. loaded from a .mat file). `unsupported` lists the tpms arguments
                 that targets/tpms.py does not reproduce (e.g. p_cubic).
    """
    tree = _parse(script)
    config_class = next(n for n in tree.body if isinstance(n, ast.ClassDef) and n.name == "configr2r")
    geometry, case = _activeCase(tree, script)

    namespace = {"np": np, "geometry": geometry}
    #configr2r and the plain assignments of the case before the target is built (config = ..., unit_cell_size = ...);
    #assignments that call anything other than configr2r (e.g. data = loadmat(...)) are skipped
    statements = [config_class]
    tpms_call = None
    for statement in case.body:
        if _isAssignTo(statement, "a"):
            call = statement.value
            if isinstance(call, ast.Call) and getattr(call.func, "attr", None) == "tpms":
                tpms_call = call
            break
        calls = [n for n in ast.walk(statement) if isinstance(n, ast.Call)]
        if isinstance(statement, ast.Assign) and all(getattr(c.func, "id", None) == "configr2r" for c in calls):
            statements.append(statement)
    exec(compile(ast.Module(body=statements, type_ignores=[]), str(script), "exec"), namespace)
    config = namespace["config"]

    tpms = None
    if tpms_call is not None:
        keywords = {k.arg: eval(compile(ast.Expression(k.value), str(script), "eval"), namespace) for k in tpms_call.keywords}
        unsupported = sorted(set(keywords) - {"f_rho", "f_l", "unit_cell_size", "t", "anisotropy_factor", "type"})
        tpms = dict(unit_cell_size=keywords.get("unit_cell_size", 0.2), t=keywords.get("t", [-1]),
                    kind=keywords.get("type", "schwarzp"), unsupported=unsupported)
    return config, tpms


def referenceConfig(script=REFERENCE_SCRIPT):
    return referenceSpec(script)[0]


def addGeometryArgument(parser):
    """--geometry option of the entry points: selects the reference script's case."""
    parser.add_argument("--geometry", default=None,
                        help="reference script geometry (case name, see main.py --list-geometries); default: the script's default")


def applyGeometry(args):
    if args.geometry is not None:
        setGeometry(args.geometry)
