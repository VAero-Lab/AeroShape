"""Visualization of point distribution (clustering) laws.

Illustrates how different clustering functions distribute points in [0, 1],
and how they affect spanwise section placement and chordwise profile
point density on a NACA 2412 airfoil.

Uses matplotlib only — no OCC or vedo required.
"""

import numpy as np
import matplotlib.pyplot as plt
from aeroshape.core.clustering import (
    uniform, cosine, half_cosine_start, half_cosine_end,
    tanh_one_sided, tanh_two_sided, exponential, vinokur,
)


def main():
    n = 31  # number of points for distribution comparison

    # ── Distribution laws to compare ────────────────────────────────
    laws = [
        ("Uniform",              uniform),
        ("Cosine",               cosine),
        ("Half-cosine (start)",  half_cosine_start),
        ("Half-cosine (end)",    half_cosine_end),
        ("Tanh one-sided (1.5)", tanh_one_sided(1.5)),
        ("Tanh two-sided (1.5)", tanh_two_sided(1.5)),
        ("Tanh two-sided (2.5)", tanh_two_sided(2.5)),
        ("Exponential (1.3)",    exponential(1.3)),
        ("Vinokur (0.005, 0.005)", vinokur(0.005, 0.005)),
    ]

    # ── Figure 1: 1-D distribution comparison ──────────────────────
    fig, axes = plt.subplots(3, 3, figsize=(14, 9))
    fig.suptitle("Point Distribution Laws — n = %d points in [0, 1]" % n,
                 fontsize=14, fontweight="bold")

    for ax, (name, law) in zip(axes.flat, laws):
        t = law(n)
        dt = np.diff(t)

        # Top: point positions on [0, 1]
        ax.plot(t, np.zeros_like(t), "k|", markersize=12, markeredgewidth=1.5)
        ax.set_xlim(-0.02, 1.02)
        ax.set_ylim(-0.35, 0.35)
        ax.set_title(name, fontsize=10)
        ax.set_yticks([])
        ax.set_xlabel("t", fontsize=9)

        # Overlay local spacing as bar chart
        ax2 = ax.twinx()
        midpoints = 0.5 * (t[:-1] + t[1:])
        ax2.bar(midpoints, dt, width=dt * 0.8, alpha=0.3, color="steelblue",
                edgecolor="steelblue", linewidth=0.5)
        ax2.set_ylim(0, max(dt) * 2.2)
        ax2.set_ylabel("$\\Delta t$", fontsize=9, color="steelblue")
        ax2.tick_params(axis="y", labelcolor="steelblue", labelsize=7)

    plt.tight_layout(rect=[0, 0, 1, 0.95])

    # ── Figure 2: Spanwise section placement ───────────────────────
    span = 10.0  # semi-span [m]
    n_sec = 15
    span_laws = [
        ("Uniform",              uniform),
        ("Cosine",               cosine),
        ("Tanh two-sided (2.0)", tanh_two_sided(2.0)),
        ("Half-cosine (start)",  half_cosine_start),
    ]

    fig2, ax_span = plt.subplots(figsize=(12, 4))
    fig2.suptitle("Spanwise Section Placement — %d sections over %.0f m"
                  % (n_sec, span), fontsize=13, fontweight="bold")

    colors = ["#1f77b4", "#ff7f0e", "#2ca02c", "#d62728"]
    for i, (name, law) in enumerate(span_laws):
        y_pos = law(n_sec) * span
        y_off = i * 0.4
        ax_span.plot(y_pos, np.full_like(y_pos, y_off), "|",
                     markersize=25, markeredgewidth=2, color=colors[i],
                     label=name)
        ax_span.axhline(y_off, color=colors[i], alpha=0.15, linewidth=8)

    ax_span.set_xlabel("Spanwise position Y [m]", fontsize=11)
    ax_span.set_yticks([i * 0.4 for i in range(len(span_laws))])
    ax_span.set_yticklabels([name for name, _ in span_laws], fontsize=10)
    ax_span.set_xlim(-0.3, span + 0.3)
    ax_span.legend(loc="upper center", ncol=4, fontsize=9,
                   bbox_to_anchor=(0.5, -0.18))
    plt.tight_layout(rect=[0, 0.08, 1, 0.95])

    # ── Figure 3: Chordwise point placement on NACA 2412 ───────────
    from aeroshape.geometry.airfoils import AirfoilProfile

    n_chord = 61
    chord_laws = [
        ("Uniform",              uniform),
        ("Cosine",               cosine),
        ("Tanh two-sided (1.5)", tanh_two_sided(1.5)),
        ("Half-cosine (start)",  half_cosine_start),
    ]

    fig3, axes3 = plt.subplots(2, 2, figsize=(13, 7))
    fig3.suptitle("Chordwise Point Distribution on NACA 2412 — %d points"
                  % n_chord, fontsize=13, fontweight="bold")

    # Generate high-res reference airfoil
    ref = AirfoilProfile.from_naca4("2412", num_points=500)

    for ax, (name, law) in zip(axes3.flat, chord_laws):
        # Resample with clustering
        profile = AirfoilProfile.from_naca4("2412", num_points=200)
        dx = np.diff(profile.x)
        dz = np.diff(profile.z)
        ds = np.sqrt(dx**2 + dz**2)
        s = np.concatenate(([0.0], np.cumsum(ds)))
        s_norm = s / s[-1]
        s_new = law(n_chord)
        x_pts = np.interp(s_new, s_norm, profile.x)
        z_pts = np.interp(s_new, s_norm, profile.z)

        # Plot reference shape
        ax.fill(ref.x, ref.z, alpha=0.08, color="gray")
        ax.plot(ref.x, ref.z, "-", color="gray", linewidth=0.5, alpha=0.5)
        # Plot clustered points
        ax.plot(x_pts, z_pts, "o-", markersize=3, linewidth=0.8,
                color=colors[chord_laws.index((name, law))],
                markerfacecolor="white", markeredgewidth=0.8)
        ax.set_title(name, fontsize=11)
        ax.set_aspect("equal")
        ax.set_xlabel("x/c")
        ax.set_ylabel("z/c")
        ax.set_xlim(-0.05, 1.05)

    plt.tight_layout(rect=[0, 0, 1, 0.95])

    plt.show()


if __name__ == "__main__":
    main()
