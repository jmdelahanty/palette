#!/usr/bin/env python3
"""
Visual Angle (Angular Size) Visualizer — v5 (mm/deg readouts)
-------------------------------------------------------------
Adds live readouts of mm (units) per degree for:
- Arc (exact)               : d(R*V)/dV = R * pi/180
- Chord (exact)             : d(2R sin(V/2))/dV = R cos(V/2) * pi/180
- Plane (exact)             : d(2n tan(V/2))/dV = n sec^2(V/2) * pi/180
Also shows the small-angle approximation for plane: (pi/180) * n.

Units: whatever units you choose for R and n (e.g., mm) — the readout
is in those units per degree.
"""

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.widgets import Slider, Button
from matplotlib.patches import Circle, Arc
import math

def visual_angle(S, D):
    S = max(1e-12, S); D = max(1e-12, D)
    return 2.0 * math.degrees(math.atan(S/(2.0*D)))

def retinal_sizes(V_deg, R, n):
    V = math.radians(V_deg)
    arc = R * V
    chord = 2.0 * R * math.sin(V/2.0)
    plane = 2.0 * n * math.tan(V/2.0)
    return arc, chord, plane

def retina_center(R, n):
    return (-n - R, 0.0)

def line_circle_intersections(m, xc, yc, R):
    a = 1.0 + m*m
    b = -2.0 * xc
    c = xc*xc + yc*yc - R*R
    disc = b*b - 4*a*c
    if disc < 0: return None
    sd = math.sqrt(disc)
    x1 = (-b + sd)/(2*a); x2 = (-b - sd)/(2*a)
    if x1 > x2: x1, x2 = x2, x1
    return (x1, m*x1), (x2, m*x2)

def make_plot():
    S0, D0, R0, n0 = 1.0, 10.0, 1.2, 1.7
    Vgoal0 = 175.0
    EP_frac0, Iris_frac0, Pupil0 = 0.25, 0.22, 0.35

    fig, ax = plt.subplots(figsize=(10, 6.4))
    plt.subplots_adjust(left=0.10, bottom=0.48)

    # Scene elements (similar to v4, shortened for brevity)
    obj_line, = ax.plot([D0, D0], [-S0/2, S0/2], lw=2, color='tab:orange')
    ray_top_fwd, = ax.plot([0, D0], [0, S0/2], lw=1, color='tab:green')
    ray_bot_fwd, = ax.plot([0, D0], [0, -S0/2], lw=1, color='tab:red')
    ray_top_back, = ax.plot([], [], lw=1, linestyle=':', color='tab:gray')
    ray_bot_back, = ax.plot([], [], lw=1, linestyle=':', color='tab:gray')

    retina_circle = Circle((0,0), 1.0, fill=False, lw=2, color='black')
    ax.add_patch(retina_circle)
    plane_line, = ax.plot([], [], lw=1, linestyle='--', color='tab:purple')
    plane_img, = ax.plot([], [], lw=3, color='tab:purple')

    # nodal
    ax.plot([0], [0], marker='o', color='tab:blue')
    ax.text(0, 0, " O (nodal)", va='bottom', ha='left', fontsize=9)

    # entrance pupil & iris (schematic)
    EP_pt, = ax.plot([], [], marker='o', mfc='none', mec='k')
    EP_label = ax.text(0, 0, "", fontsize=9)
    iris_line, = ax.plot([], [], lw=1.5, color='k')
    iris_aperture, = ax.plot([], [], lw=3, color='k')
    iris_label = ax.text(0, 0, "", fontsize=9)

    chord_line, = ax.plot([], [], lw=2, color='goldenrod')
    a_pt, = ax.plot([], [], marker='o', color='teal')
    b_pt, = ax.plot([], [], marker='o', color='teal')
    arc_patch = Arc((0,0), 2.0, 2.0, angle=0, theta1=0, theta2=0, lw=3, color='goldenrod')
    ax.add_patch(arc_patch)

    wedge = Arc((0,0), 0.8, 0.8, angle=0, theta1=0, theta2=0, lw=2, color='gray')
    ax.add_patch(wedge)
    ax.text(0.5, 0.92, "180° limit (collision)", transform=ax.transAxes, ha='center', va='center', fontsize=9)
    angle_text = ax.text(0.02, 0.98, "", transform=ax.transAxes, va='top', ha='left')

    ax.set_aspect('equal', adjustable='box')
    ax.set_xlabel("Distance (D)")
    ax.set_ylabel("Height")
    ax.set_title("Visual Angle with mm/° Readouts")

    def set_bounds(S, D, R, n):
        left = -n - 1.4*R - 0.25*max(D, 1.0)
        right = D + 0.35*max(D, 1.0)
        half_y = max(1.2, 0.8*S, 1.3*R)
        ax.set_xlim(left, right)
        ax.set_ylim(-half_y, half_y)
        fig.canvas.draw_idle()

    # Sliders
    ax_S = plt.axes([0.12, 0.39, 0.76, 0.03])
    ax_D = plt.axes([0.12, 0.35, 0.76, 0.03])
    ax_R = plt.axes([0.12, 0.31, 0.35, 0.03])
    ax_n = plt.axes([0.53, 0.31, 0.35, 0.03])
    ax_EP = plt.axes([0.12, 0.27, 0.35, 0.03])
    ax_Iris = plt.axes([0.53, 0.27, 0.35, 0.03])
    ax_Pupil = plt.axes([0.12, 0.23, 0.35, 0.03])
    ax_Vg = plt.axes([0.12, 0.19, 0.76, 0.03])

    s_S = Slider(ax_S, 'S (size)', 1e-3, 20.0, valinit=S0, valstep=1e-3)
    s_D = Slider(ax_D, 'D (distance)', 1e-3, 100.0, valinit=D0, valstep=1e-3)
    s_R = Slider(ax_R, 'R (retina R)', 0.2, 6.0, valinit=R0, valstep=1e-3)
    s_n = Slider(ax_n, 'n (nodal)', 0.3, 6.0, valinit=n0, valstep=1e-3)
    s_EP = Slider(ax_EP, 'EP_frac (×n ahead of O)', 0.0, 0.9, valinit=EP_frac0, valstep=1e-3)
    s_Iris = Slider(ax_Iris, 'Iris_frac (×n ahead of O)', 0.0, 0.9, valinit=Iris_frac0, valstep=1e-3)
    s_Pupil = Slider(ax_Pupil, 'Pupil radius (glyph)', 0.05, 1.5, valinit=Pupil0, valstep=1e-3)
    s_Vg = Slider(ax_Vg, 'V_goal (deg)', 60.0, 179.0, valinit=Vgoal0, valstep=0.1)

    # Readouts (bottom of figure)
    readout_sizes = fig.text(0.5, 0.12, "", ha='center', va='bottom', fontsize=9)
    readout_mmdeg = fig.text(0.5, 0.08, "", ha='center', va='bottom', fontsize=9)
    goal_out = fig.text(0.5, 0.04, "", ha='center', va='bottom', fontsize=9)

    def update(val=None):
        S = s_S.val; D = s_D.val; R = s_R.val; n = s_n.val
        ep_frac = s_EP.val; iris_frac = s_Iris.val; pupil_r = s_Pupil.val
        Vg = s_Vg.val

        yh = S/2.0
        obj_line.set_data([D, D], [-yh, yh])
        ray_top_fwd.set_data([0, D], [0, yh])
        ray_bot_fwd.set_data([0, D], [0, -yh])

        V = visual_angle(S, D)
        angle_text.set_text(f"V ≈ {V:.2f}°")
        theta = math.degrees(math.atan2(yh, D))
        wedge.theta1 = -theta; wedge.theta2 = theta

        # Retina and plane
        xc, yc = retina_center(R, n)
        retina_circle.center = (xc, yc); retina_circle.radius = R
        plane_line.set_data([-n, -n], [-max(R,S), max(R,S)])

        arc_len, chord_len, plane_len = retinal_sizes(V, R, n)
        plane_img.set_data([-n, -n], [-plane_len/2.0, plane_len/2.0])

        # Intersections back to retina
        m = (S/2.0)/D
        top_pts = line_circle_intersections(m, xc, yc, R)
        bot_pts = line_circle_intersections(-m, xc, yc, R)
        if top_pts and bot_pts:
            top = max(top_pts, key=lambda p: p[0])
            bot = max(bot_pts, key=lambda p: p[0])
            ray_top_back.set_data([0, top[0]], [0, top[1]])
            ray_bot_back.set_data([0, bot[0]], [0, bot[1]])
            a_pt.set_data([bot[0]], [bot[1]])
            b_pt.set_data([top[0]], [top[1]])
            chord_line.set_data([bot[0], top[0]], [bot[1], top[1]])

            def theta_deg(pt): return math.degrees(math.atan2(pt[1]-yc, pt[0]-xc))
            th_a = (theta_deg(bot) + 360) % 360
            th_b = (theta_deg(top) + 360) % 360
            delta = (th_b - th_a + 540) % 360 - 180
            arc_patch.center = (xc, yc); arc_patch.width = 2*R; arc_patch.height = 2*R
            arc_patch.theta1 = th_a; arc_patch.theta2 = th_a + delta
        else:
            ray_top_back.set_data([], []); ray_bot_back.set_data([], [])
            a_pt.set_data([], []); b_pt.set_data([], [])
            chord_line.set_data([], []); arc_patch.theta1 = 0; arc_patch.theta2 = 0

        # Entrance pupil & iris schematic positions
        x_ep = ep_frac * n; EP_pt.set_data([x_ep], [0])
        EP_label.set_text(f" EP (entrance pupil) x={x_ep:.2f}")
        EP_label.set_position((x_ep, 0.05))
        x_iris = iris_frac * n
        iris_line.set_data([x_iris, x_iris], [-max(R,S)*0.9, max(R,S)*0.9])
        iris_aperture.set_data([x_iris, x_iris], [-pupil_r/2, pupil_r/2])
        iris_label.set_text(" Iris plane")
        iris_label.set_position((x_iris, max(R,S)*0.95))

        # Readout: sizes
        readout_sizes.set_text(f"Retina image — arc: {arc_len:.3f}, chord: {chord_len:.3f}, plane: {plane_len:.3f} (units)")

        # Readout: mm/deg (exact + small-angle for plane)
        V_rad = math.radians(V)
        mmdeg_arc_exact   = R * (math.pi/180.0)  # exact
        mmdeg_chord_exact = R * math.cos(V_rad/2.0) * (math.pi/180.0)
        mmdeg_plane_exact = n * (1.0 / math.cos(V_rad/2.0))**2 * (math.pi/180.0)  # n sec^2(V/2) * pi/180
        mmdeg_plane_small = n * (math.pi/180.0)

        readout_mmdeg.set_text(
            "mm/° — arc (exact): {:.4f} | chord (exact): {:.4f} | plane (exact): {:.4f} | plane (small-angle): {:.4f}".format(
                mmdeg_arc_exact, mmdeg_chord_exact, mmdeg_plane_exact, mmdeg_plane_small
            )
        )

        # Goal distance
        Vg_rad = math.radians(Vg); D_target = S / (2.0 * max(1e-12, math.tan(Vg_rad/2.0)))
        goal_out.set_text(f"For V_goal={Vg:.1f}°, need D_target ≈ {D_target:.4f}")

        set_bounds(S, D, R, n)

    s_S.on_changed(update); s_D.on_changed(update); s_R.on_changed(update); s_n.on_changed(update)
    s_EP.on_changed(update); s_Iris.on_changed(update); s_Pupil.on_changed(update); s_Vg.on_changed(update)

    # Buttons
    ax_reset = plt.axes([0.12, 0.10, 0.18, 0.06])
    ax_loom = plt.axes([0.35, 0.10, 0.18, 0.06])
    ax_loom_goal = plt.axes([0.58, 0.10, 0.30, 0.06])

    b_reset = Button(ax_reset, 'Reset')
    b_loom = Button(ax_loom, 'Animate Loom')
    b_loom_goal = Button(ax_loom_goal, 'Loom to V_goal')

    def on_reset(event):
        s_S.reset(); s_D.reset(); s_R.reset(); s_n.reset()
        s_EP.reset(); s_Iris.reset(); s_Pupil.reset(); s_Vg.reset()
        update()

    def on_loom(event):
        D = s_D.val
        for _ in range(360):
            if not plt.fignum_exists(fig.number): break
            D = max(1e-6, D * 0.97)
            s_D.set_val(D); update(); plt.pause(0.02)

    def on_loom_goal(event):
        S = s_S.val; D = s_D.val; Vg = s_Vg.val
        Vg_rad = math.radians(Vg); D_target = S / (2.0 * max(1e-12, math.tan(Vg_rad/2.0)))
        while D > D_target and plt.fignum_exists(fig.number):
            D = max(D_target, D - max(0.001, 0.05*D))
            s_D.set_val(D); update(); plt.pause(0.02)

    b_reset.on_clicked(on_reset); b_loom.on_clicked(on_loom); b_loom_goal.on_clicked(on_loom_goal)

    update()
    plt.show()

if __name__ == "__main__":
    make_plot()