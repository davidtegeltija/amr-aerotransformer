"""
========================================================================
Aerodynamic force and moment coefficients from wing surface fields.
========================================================================

Turns a surface solution into the integrated coefficients an aerodynamicist
actually reads: lift, drag and pitching moment. The field metrics in
``src/evaluation/metrics.py`` score the prediction pointwise; these functions
score what the prediction *means*, which is the quantity a surrogate is
ultimately judged on.

Data layout
-----------
The SuperWing samples this project trains on carry a structured surface mesh of
``[3, 128, 256]``, read by ``WingDataset`` as ``[H, W, C]``:

    geometry (input)  : (x, y, z) node coordinates of the wing surface
    solution (target) : (cp, cf_tau, cf_z) — pressure coefficient, chordwise
                        skin-friction magnitude, and spanwise skin friction

Node quantities live on the ``[..., I, J, 3]`` mesh, cell quantities on the
``[..., I-1, J-1]`` grid of quadrilateral cells it spans. ``I`` indexes spanwise
stations, ``J`` indexes points around a section (chordwise).

Typical use
-----------
``skin_friction_to_xyz`` lifts the two friction channels into a 3-D vector using
the local surface tangent, then ``wind_force_coefficients`` and
``moment_coefficients`` integrate over the surface::

    cf = skin_friction_to_xyz(geometry, solution[..., 1:])
    cd, cl, cz = wind_force_coefficients(geometry, angle_of_attack, cp, cf) / ref_area
    cmz = moment_coefficients(geometry, cp, cf, ref_point)[..., 2] / ref_area / ref_chord

Both integrals are unnormalized: divide by the reference area (and reference
chord for moments) to get comparable coefficients. Those references, along with
the solver's own CL/CD/CMz, sit in columns 4-8 of the SuperWing index file, so a
run can be checked against them directly.

Note the target channels are stored pre-scaled by ``(1, 150, 300)``; undo that
scaling before integrating, or the friction contribution is off by two orders of
magnitude.
"""

import torch


# ---------------------------------------------------------------------------
# Rotation between body axes and wind axes
# ---------------------------------------------------------------------------

# The base rotation matrix:
#  / (1, 0)  (0, 1) \
#  \ (0,-1)  (1, 0) /
# To rotate a vector (x_o, y_o) from the origin frame (o) to the target frame
# (t), contract this basis with the origin x unit-vector expressed in the target
# frame. For example, to transfer a force (f_x, f_y) to drag and lift:
#   - the target frame (along the freestream) is the origin frame (along the
#     chord) rotated by the angle of attack, counter-clockwise
#   - the x unit-vector in the target frame is /  cos(AoA) \
#                                              \ -sin(AoA) /
#   - thus ( Drag, Lift ) = ( f_x, f_y ) .  / (1, 0)  (0, 1) \  .  /  cos(AoA) \
#                                           \ (0,-1)  (1, 0) /     \ -sin(AoA) /
ROTATION_MATRIX = torch.Tensor([[[1.0, 0], [0, 1.0]], [[0, -1.0], [1.0, 0]]])


def _freestream_unit_vector(angle_of_attack: torch.Tensor) -> torch.Tensor:
    """Freestream direction as a unit vector in the body frame.

    Args:
        angle_of_attack: Angles of attack in degrees, shape ``[B]``.

    Returns:
        Tensor ``[B, 2]`` holding ``(cos(aoa), -sin(aoa))`` per sample.
    """
    angle_of_attack = torch.deg2rad(angle_of_attack)
    return torch.cat((torch.cos(angle_of_attack).unsqueeze(1),
                      -torch.sin(angle_of_attack).unsqueeze(1)), dim=1)


def rotate_to_wind_axes(forces_xy: torch.Tensor, angle_of_attack: torch.Tensor) -> torch.Tensor:
    """Rotate in-plane body-axis forces into wind axes, i.e. (Fx, Fy) -> (CD, CL).

    Args:
        forces_xy: Body-axis force components ``(Fx, Fy)``, shape ``[B, 2]``.
        angle_of_attack: Angles of attack in degrees, shape ``[B]``.

    Returns:
        Tensor ``[B, 2]`` holding ``(CD, CL)`` per sample.
    """
    return torch.einsum('bp,prs,bs->br', forces_xy,
                        ROTATION_MATRIX.to(forces_xy.device),
                        _freestream_unit_vector(angle_of_attack).to(forces_xy.device))


# ---------------------------------------------------------------------------
# Surface mesh geometry
# ---------------------------------------------------------------------------

def surface_cell_normals_and_areas(geometry: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
    """Outward unit normal and area of every quadrilateral surface cell.

    Each cell is spanned by four neighbouring mesh nodes. Its normal comes from
    the cross product of the two diagonals; its area from the two triangles the
    diagonals cut it into, which stays exact for a non-planar quad.

    Args:
        geometry: Node coordinates ``(x, y, z)``, shape ``[..., I, J, 3]``.

    Returns:
        Tuple of the unit normals ``[..., I-1, J-1, 3]`` and the cell areas
        ``[..., I-1, J-1]``.
    """
    # Corner nodes of each cell (p0, p1, p2, p3)
    p0 = geometry[..., :-1, :-1, :]     # SW
    p1 = geometry[..., :-1, 1:, :]      # SE
    p2 = geometry[..., 1:, 1:, :]       # NW
    p3 = geometry[..., 1:, :-1, :]      # NE

    normals = torch.cross(p2 - p0, p3 - p1, dim=-1)
    areas = 0.5 * (torch.linalg.norm(torch.cross(p1 - p0, p2 - p0, dim=-1), dim=-1)
                   + torch.linalg.norm(torch.cross(p2 - p0, p3 - p0, dim=-1), dim=-1))

    normals = normals / (torch.linalg.norm(normals, dim=-1, keepdim=True) + 1e-20)
    return normals, areas


def section_tangents_and_normals(section: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
    """Unit tangent and normal of every segment of a 2-D section outline.

    The normal is the tangent rotated a quarter turn counter-clockwise, so the
    pair forms a right-handed frame walking along the outline.

    Args:
        section: Outline node coordinates ``(x, y)``, shape ``[..., J, 2]``.

    Returns:
        Tuple of the unit tangents and unit normals, both ``[..., J-1, 2]``,
        defined at segment centres.
    """
    tangents = section[..., 1:, :] - section[..., :-1, :]
    tangents = tangents / (torch.linalg.norm(tangents, dim=-1, keepdim=True) + 1e-20)
    normals = torch.cat((-tangents[..., [1]], tangents[..., [0]]), dim=-1)

    return tangents, normals


def skin_friction_to_xyz(geometry: torch.Tensor, cf: torch.Tensor) -> torch.Tensor:
    """Lift the two stored skin-friction channels into a 3-D friction vector.

    The solution files carry friction as a chordwise magnitude plus a spanwise
    component, which only becomes a vector once the local surface direction is
    known. The chordwise part is laid along the section tangent, averaged onto
    cell centres in the spanwise direction, and the spanwise part rides along as
    the z component.

    Args:
        geometry: Node coordinates ``(x, y, z)``, shape ``[..., I, J, 3]``.
        cf: Friction components ``(cf_tau, cf_z)``, shape ``[..., I-1, J-1, 2]``.

    Returns:
        Friction coefficient vectors ``[..., I-1, J-1, 3]``.
    """
    tangents, _ = section_tangents_and_normals(geometry[..., [0, 1]])
    # Section tangents sit on node rows; average adjacent rows onto cell centres.
    tangents = 0.5 * (tangents[..., 1:, :, :] + tangents[..., :-1, :, :])

    return torch.cat((cf[..., [0]] * tangents, cf[..., [1]]), dim=-1)


# ---------------------------------------------------------------------------
# Surface integration of forces and moments
# ---------------------------------------------------------------------------

def cell_force_coefficients(
    geometry: torch.Tensor | list[torch.Tensor],
    cp: torch.Tensor,
    cf: torch.Tensor = None
) -> torch.Tensor:
    """Force coefficient contributed by each individual surface cell.

    Pressure acts along the cell normal, friction along the surface, so the
    friction vector is projected onto the tangent plane before it is added.

    Args:
        geometry: Node coordinates ``(x, y, z)``, shape ``[..., I, J, 3]``, or a
            precomputed ``[normals, areas]`` pair from
            ``surface_cell_normals_and_areas`` when integrating many solutions
            over the same mesh.
        cp: Pressure coefficients ``Cp = (p - p_inf) / 0.5 * rho * U_inf^2``,
            shape ``[..., I-1, J-1]``.
        cf: Friction coefficients ``Cf = (tau @ n) / 0.5 * rho * U_inf^2``,
            shape ``[..., I-1, J-1, 3]``. Omit for an inviscid (pressure-only)
            integral.

    Returns:
        Per-cell force coefficients ``(dCx, dCy, dCz)``, shape
        ``[..., I-1, J-1, 3]``.
    """
    if isinstance(geometry, list):
        normals, areas = geometry
    else:
        normals, areas = surface_cell_normals_and_areas(geometry)

    forces = cp[..., None] * normals * areas[..., None]

    if not (cf is None or len(cf) == 0):
        shear = (cf - torch.sum(cf * normals, dim=-1, keepdim=True) * normals) * areas[..., None]
        forces = forces + shear

    return forces


def body_force_coefficients(
    geometry: torch.Tensor | list[torch.Tensor],
    cp: torch.Tensor,
    cf: torch.Tensor = None
) -> torch.Tensor:
    """Total force coefficients in body axes, integrated over the surface.

    Args:
        geometry: Node coordinates ``(x, y, z)``, shape ``[..., I, J, 3]``, or a
            precomputed ``[normals, areas]`` pair.
        cp: Pressure coefficients, shape ``[..., I-1, J-1]``.
        cf: Friction coefficients, shape ``[..., I-1, J-1, 3]``. Optional.

    Returns:
        Force coefficients ``(CX, CY, CZ)``, shape ``[..., 3]``, unnormalized by
        the reference area.
    """
    return torch.sum(cell_force_coefficients(geometry, cp, cf), dim=(-3, -2))


def wind_force_coefficients(
    geometry: torch.Tensor | list[torch.Tensor],
    angle_of_attack: torch.Tensor,
    cp: torch.Tensor,
    cf: torch.Tensor = None
) -> torch.Tensor:
    """Total force coefficients in wind axes: drag, lift and side force.

    The same surface integral as ``body_force_coefficients``, with the in-plane
    components rotated by the angle of attack so they read as drag and lift.

    Args:
        geometry: Node coordinates ``(x, y, z)``, shape ``[..., I, J, 3]``, or a
            precomputed ``[normals, areas]`` pair.
        angle_of_attack: Angles of attack in degrees, shape ``[...]``.
        cp: Pressure coefficients, shape ``[..., I-1, J-1]``.
        cf: Friction coefficients, shape ``[..., I-1, J-1, 3]``. Optional.

    Returns:
        Force coefficients ``(CD, CL, CZ)``, shape ``[..., 3]``, unnormalized by
        the reference area.
    """
    forces = body_force_coefficients(geometry, cp, cf)
    drag_lift = rotate_to_wind_axes(forces[..., :2], angle_of_attack)
    return torch.cat((drag_lift, forces[..., 2:]), dim=-1)


def moment_coefficients(
    geometry: torch.Tensor,
    cp: torch.Tensor,
    cf: torch.Tensor = None,
    ref_point: torch.Tensor = None
) -> torch.Tensor:
    """Total moment coefficients about a reference point.

    Each cell force is applied at the cell centre and crossed with its lever arm
    from the reference point. The third component ``CMz`` is the pitching moment
    reported for the SuperWing cases.

    Args:
        geometry: Node coordinates ``(x, y, z)``, shape ``[..., I, J, 3]``.
        cp: Pressure coefficients, shape ``[..., I-1, J-1]``.
        cf: Friction coefficients, shape ``[..., I-1, J-1, 3]``. Optional.
        ref_point: Moment reference point ``(x, y, z)``, shape ``[3]``. Defaults
            to the quarter-chord point ``(0.25, 0, 0)``.

    Returns:
        Moment coefficients ``(CMx, CMy, CMz)``, shape ``[..., 3]``,
        unnormalized by the reference area and chord.
    """
    if ref_point is None:
        ref_point = torch.Tensor([0.25, 0.0, 0.0])

    forces = cell_force_coefficients(geometry, cp, cf)
    lever_arms = 0.25 * (geometry[..., :-1, :-1, :] + geometry[..., :-1, 1:, :]
                         + geometry[..., 1:, 1:, :] + geometry[..., 1:, :-1, :]) - ref_point.to(geometry.device)

    return torch.sum(torch.cross(lever_arms, forces, dim=-1), dim=(-3, -2))
