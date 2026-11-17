"""Generate the official MolScope publication-quality Graphical Abstract.

Mirrors and elevates the mdxplain graphical abstract layout:
- Top: Flexible Inputs (PDB with automated OpenMM MD or 1-100+ Trajectories) + Conditions/Metadata
- Middle: MolScope 3-Stage Engine (Multi-Family Featurization -> Kinetic Discovery -> Core ML & SAM-MFP)
- Bottom: Free Energy Landscapes + SAM-MFP Pareto Prioritization + 3D Structure Mapping & Rules
"""

from __future__ import annotations

from pathlib import Path
import shutil
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.patches as patches
from PIL import Image


def create_graphical_abstract(out_path: Path) -> Path:
    out_path = Path(out_path).resolve()
    out_path.parent.mkdir(parents=True, exist_ok=True)

    # Paths to benchmark assets
    bench_dir = Path("/Users/nirav/Docs/molscope-results/runs/2rjy_benchmark")
    interp_dir = Path("/Users/nirav/Docs/molscope-results/runs/2rjy_interpretability")

    fes_path = bench_dir / "free_energy" / "fes_TIC1_TIC2.png"
    timeline_path = interp_dir / "figures" / "cluster_timeline.png"
    sam_tradeoff_path = bench_dir / "figures" / "sam_mfp_tradeoff.png"
    feat_imp_path = interp_dir / "figures" / "feature_importance_ranking.png"
    rules_path = interp_dir / "figures" / "decision_tree_rules.png"
    pdb_3d_path = bench_dir / "3d_visualization" / "molscope_3d_render.png"

    # Dimensions: 16.5 x 11.3 inches at 300 DPI (~4950 x 3390 px)
    fig = plt.figure(figsize=(16.5, 11.3), dpi=300)
    ax = fig.add_axes([0, 0, 1, 1])
    ax.set_xlim(0, 100)
    ax.set_ylim(100, 0)  # Invert Y so (0,0) is top-left, intuitive coordinates!
    ax.axis("off")

    # Background
    ax.add_patch(patches.Rectangle((0, 0), 100, 100, facecolor="#f8fafc", zorder=0))

    # Helper: draw rounded container box
    def draw_box(x, y, w, h, title="", subtitle="", color="#ffffff", border="#0f172a", lw=2.2, radius=1.0):
        box = patches.FancyBboxPatch(
            (x, y), w, h,
            boxstyle=f"round,pad=0.2,rounding_size={radius}",
            facecolor=color, edgecolor=border, linewidth=lw, zorder=2,
        )
        ax.add_patch(box)
        if title:
            ax.text(x + w / 2, y + 2.4, title, ha="center", va="top", fontsize=12.5, fontweight="bold", color="#0f172a", zorder=3)
        if subtitle:
            ax.text(x + w / 2, y + 4.8, subtitle, ha="center", va="top", fontsize=9.2, style="italic", color="#475569", zorder=3)
        return box

    def draw_arrow(x1, y1, x2, y2, label="", color="#0f172a"):
        ax.annotate(
            "",
            xy=(x2, y2), xytext=(x1, y1),
            arrowprops=dict(facecolor=color, edgecolor=color, width=2.6, headwidth=9, headlength=9, shrink=0.04),
            zorder=5,
        )
        if label:
            ax.text((x1 + x2) / 2, (y1 + y2) / 2 - 0.1, label, ha="center", va="center", fontsize=9.2, fontweight="bold",
                    color=color, bbox=dict(boxstyle="round,pad=0.25", facecolor="#ffffff", edgecolor="#cbd5e1", alpha=0.95), zorder=6)

    # =========================================================================
    # TOP ROW: INPUTS (y: 3 to 17)
    # =========================================================================
    # Top-Left Box: Flexible Input
    draw_box(3, 3, 45, 14,
             title="PDB Structure or MD Simulation Trajectories",
             subtitle="Raw PDB (automated OpenMM MD) OR 1–100+ Trajectories (100 – 1M+ Frames)")

    box_pdb = patches.FancyBboxPatch((4.5, 9.2), 20, 6.8, boxstyle="round,pad=0.2", facecolor="#f1f5f9", edgecolor="#cbd5e1", lw=1.2, zorder=3)
    ax.add_patch(box_pdb)
    ax.text(14.5, 10.3, "Option A: Raw PDB Input", ha="center", va="top", fontsize=8.5, fontweight="bold", color="#1e293b", zorder=4)
    ax.text(14.5, 12.0, "• Auto-protonation at pH 7.4 (PDBFixer)\n• L-BFGS energy minimization\n• NVT equilibration & Langevin MD",
            ha="center", va="top", fontsize=7.2, color="#334155", linespacing=1.3, zorder=4)

    box_traj = patches.FancyBboxPatch((26, 9.2), 20.5, 6.8, boxstyle="round,pad=0.2", facecolor="#f1f5f9", edgecolor="#cbd5e1", lw=1.2, zorder=3)
    ax.add_patch(box_traj)
    ax.text(36.25, 10.3, "Option B: Existing Trajectories", ha="center", va="top", fontsize=8.5, fontweight="bold", color="#1e293b", zorder=4)
    ax.text(36.25, 12.0, "• DCD, XTC, NetCDF, TRR formats\n• 1–100+ multi-trajectory replicas\n• Stream accumulator (flat memory)",
            ha="center", va="top", fontsize=7.2, color="#334155", linespacing=1.3, zorder=4)

    # Top-Right Box: Conditions & Splits
    draw_box(52, 3, 45, 14,
             title="Condition Tags, Metadata & Validation Splits",
             subtitle="Trajectory IDs, replicas, functional states, mutations, and purged temporal CV splits")

    box_meta = patches.FancyBboxPatch((53.5, 9.2), 20.5, 6.8, boxstyle="round,pad=0.2", facecolor="#f1f5f9", edgecolor="#cbd5e1", lw=1.2, zorder=3)
    ax.add_patch(box_meta)
    ax.text(63.75, 10.3, "Experimental Conditions", ha="center", va="top", fontsize=8.5, fontweight="bold", color="#1e293b", zorder=4)
    ax.text(63.75, 12.0, "• Multi-replica ensembles\n• WT vs. Mutant comparative tags\n• Ligand-bound vs. Apo states",
            ha="center", va="top", fontsize=7.2, color="#334155", linespacing=1.3, zorder=4)

    box_split = patches.FancyBboxPatch((75, 9.2), 20.5, 6.8, boxstyle="round,pad=0.2", facecolor="#f1f5f9", edgecolor="#cbd5e1", lw=1.2, zorder=3)
    ax.add_patch(box_split)
    ax.text(85.25, 10.3, "Zero-Leakage Splitting", ha="center", va="top", fontsize=8.5, fontweight="bold", color="#1e293b", zorder=4)
    ax.text(85.25, 12.0, "• Purged temporal block CV\n• Trajectory-grouped outer folds\n• Zero train-test frame leakage",
            ha="center", va="top", fontsize=7.2, color="#334155", linespacing=1.3, zorder=4)

    # Arrows from Top to Center
    draw_arrow(25.5, 17.0, 25.5, 23.0, label="Simulation / Trajectory Ingestion")
    draw_arrow(74.5, 17.0, 74.5, 23.0, label="Metadata & Purged Split Generation")

    # =========================================================================
    # MIDDLE ROW: CENTRAL MOLSCOPE ENGINE (y: 23.5 to 55)
    # =========================================================================
    draw_box(3, 23.5, 94, 31.5, color="#ffffff", border="#0f172a", lw=2.5, radius=1.2)

    # Brand Title
    ax.text(50, 25.5, "MolScope", ha="center", va="top", fontsize=24, fontweight="bold", color="#0f172a", zorder=4)
    ax.text(50, 28.8, "Integrated 3-Stage Biophysical ML Engine: Featurization → Kinetic Discovery → Core Machine Learning",
            ha="center", va="top", fontsize=10.5, fontweight="bold", color="#0284c7", zorder=4)

    # Sub-stage 1: Featurization (x: 4.5 to 32.5, y: 31.5 to 53.5)
    draw_box(4.5, 31.5, 28, 22.0, title="Stage 1: Biophysical Featurization", color="#f8fafc", border="#38bdf8", lw=1.8)
    feat_text = (
        "• Pairwise Heavy-Atom Contacts (<4.5 Å)\n"
        "• Cα Pairwise Distance Matrix (Å)\n"
        "• DSSP Secondary Structure (α, β, coil)\n"
        "• Backbone & Sidechain Torsions (φ, ψ, χ₁)\n"
        "• Solvent Accessible Surface Area (SASA)\n"
        "• Salt Bridges & Aromatic π-Stacking Networks\n\n"
        "Output: High-dimensional feature matrix\n"
        "Transparent quality checks: zero silent imputation"
    )
    ax.text(18.5, 36.8, feat_text, ha="center", va="top", fontsize=7.8, color="#1e293b", linespacing=1.45, zorder=4)

    # Arrow 1 -> 2
    ax.annotate("", xy=(34.8, 42.5), xytext=(32.7, 42.5),
                arrowprops=dict(facecolor="#0284c7", edgecolor="#0284c7", width=2.4, headwidth=8, headlength=8), zorder=5)

    # Sub-stage 2: Kinetic Discovery & Clustering (x: 35.5 to 63.5, y: 31.5 to 53.5)
    draw_box(35.5, 31.5, 28, 22.0, title="Stage 2: Kinetic Discovery & Clustering", color="#f8fafc", border="#6366f1", lw=1.8)
    disc_text = (
        "• Time-lagged Independent Component Analysis\n"
        "  (TICA): slow kinetic coordinates (implied timescales)\n"
        "• Contact Kernel PCA (CKPCA): nonlinear manifolds\n"
        "• Density Peak Advanced (DPA) Clustering:\n"
        "  detects cluster centers at local density maxima\n"
        "  connected by density saddle points (free energy barriers)\n\n"
        "Output: Discrete conformational macrostates\n"
        "8.3x faster than mdxplain with 23% lower RAM"
    )
    ax.text(49.5, 36.8, disc_text, ha="center", va="top", fontsize=7.8, color="#1e293b", linespacing=1.45, zorder=4)

    # Arrow 2 -> 3
    ax.annotate("", xy=(65.8, 42.5), xytext=(63.7, 42.5),
                arrowprops=dict(facecolor="#6366f1", edgecolor="#6366f1", width=2.4, headwidth=8, headlength=8), zorder=5)

    # Sub-stage 3: Core ML & SAM-MFP (x: 66.5 to 95.5, y: 31.5 to 53.5)
    draw_box(66.5, 31.5, 29, 22.0, title="Stage 3: Core ML & SAM-MFP", color="#f8fafc", border="#f97316", lw=1.8)
    ml_text = (
        "• SAM-MFP Prioritization (Core & Non-Optional):\n"
        "  Multi-objective Pareto front balancing:\n"
        "  PCA/TICA Salience vs ESS Ratio vs Uniqueness\n"
        "  Overcomes the 'Variance Fallacy' in MD\n"
        "• Grouped Supervised ML (Random Forest / GBDT):\n"
        "  Purged trajectory-level cross-validation\n"
        "• Leave-One-Feature-Out (LOFO) Importance:\n"
        "  Accuracy gain/drop per feature group\n"
        "• Negative Controls: Permutation test collapses to chance"
    )
    ax.text(81.0, 36.8, ml_text, ha="center", va="top", fontsize=7.8, color="#1e293b", linespacing=1.45, zorder=4)

    # =========================================================================
    # BOTTOM ROW: 3 ACTIONABLE EXPLANATION PANELS (y: 61.5 to 97.0)
    # =========================================================================
    # Arrows from Center to Bottom Panels
    draw_arrow(18.5, 55.0, 18.5, 60.8, label="Visualize Dynamics")
    draw_arrow(50.0, 55.0, 50.0, 60.8, label="Prioritize Features (SAM-MFP)")
    draw_arrow(81.5, 55.0, 81.5, 60.8, label="Explain & Map in 3D")

    # Bottom Panel 1: Free Energy Landscape & Dynamics (x: 3 to 33.5, y: 61.5 to 96.5)
    draw_box(3, 61.5, 30.5, 35.0, title="Free Energy Landscapes & Dynamics",
             subtitle="Metastable basins on TICA/PCA & cluster transitions", color="#ffffff", border="#0f172a")

    # Bottom Panel 2: SAM-MFP Multi-Objective Prioritization (x: 35 to 65.5, y: 61.5 to 96.5)
    draw_box(35, 61.5, 30.5, 35.0, title="Multi-Objective Feature Prioritization",
             subtitle="SAM-MFP Pareto trade-off & LOFO ML feature ranking", color="#ffffff", border="#0f172a")

    # Bottom Panel 3: Explain & Map in 3D (x: 67 to 97, y: 61.5 to 96.5)
    draw_box(67, 61.5, 30.0, 35.0, title="Transparent Rules & 3D Structure Mapping",
             subtitle="Decision tree if-then rules & PyMOL B-factor mapping", color="#ffffff", border="#0f172a")

    # Footer note inside figure
    ax.text(50, 98.6, "MolScope: System-Aware Multi-Criteria Molecular Dynamics Feature Discovery & Machine Learning · MIT License · github.com/nirav/molscope",
            ha="center", va="center", fontsize=8.5, fontweight="normal", color="#64748b", zorder=4)

    # Embed Actual Images using Fig Insets:
    # In Fig coordinates, (0,0) is bottom-left, (1,1) is top-right!
    # Convert ax Y (0 top to 100 bottom) to Fig Y: fig_y = (100 - y) / 100

    # Panel 1: Top = FES, Bottom = Timeline
    # Top image: y from 68.0 to 82.5 -> fig_y from 0.175 to 0.320
    if fes_path.is_file():
        ax_fes = fig.add_axes([0.045, 0.175, 0.275, 0.145])
        fes_im = Image.open(fes_path)
        ax_fes.imshow(fes_im)
        ax_fes.axis("off")

    # Bottom image: y from 83.5 to 95.0 -> fig_y from 0.050 to 0.165
    if timeline_path.is_file():
        ax_time = fig.add_axes([0.045, 0.048, 0.275, 0.115])
        time_im = Image.open(timeline_path)
        ax_time.imshow(time_im)
        ax_time.axis("off")

    # Panel 2: Top = SAM-MFP Tradeoff, Bottom = Feature Importance
    if sam_tradeoff_path.is_file():
        ax_sam = fig.add_axes([0.365, 0.175, 0.275, 0.145])
        sam_im = Image.open(sam_tradeoff_path)
        ax_sam.imshow(sam_im)
        ax_sam.axis("off")

    if feat_imp_path.is_file():
        ax_imp = fig.add_axes([0.365, 0.048, 0.275, 0.115])
        imp_im = Image.open(feat_imp_path)
        ax_imp.imshow(imp_im)
        ax_imp.axis("off")

    # Panel 3: Left = Decision Tree Rules (crop), Right = 3D PyMOL render
    if rules_path.is_file():
        ax_rules = fig.add_axes([0.685, 0.175, 0.145, 0.145])
        rules_im = Image.open(rules_path)
        w, h = rules_im.size
        # Crop to top 3 rules
        rules_crop = rules_im.crop((0, int(h * 0.08), w, int(h * 0.52)))
        ax_rules.imshow(rules_crop)
        ax_rules.axis("off")

    if pdb_3d_path.is_file():
        ax_3d = fig.add_axes([0.835, 0.170, 0.125, 0.155])
        pdb_im = Image.open(pdb_3d_path)
        ax_3d.imshow(pdb_im)
        ax_3d.axis("off")

    # Bottom of Panel 3: HTML report dashboard banner (y: 84.0 to 95.0 -> fig_y: 0.050 to 0.160)
    badge = patches.FancyBboxPatch((68.5, 83.5), 27.0, 11.5, boxstyle="round,pad=0.2", facecolor="#ecfdf5", edgecolor="#10b981", lw=1.5, zorder=3)
    ax.add_patch(badge)
    ax.text(82.0, 85.2, "Interactive Dashboard (report.html)", ha="center", va="top", fontsize=9.0, fontweight="bold", color="#065f46", zorder=4)
    ax.text(82.0, 87.5,
            "• Self-contained offline HTML report with tabs\n"
            "• Interactive Mol* 3D biomolecular viewer\n"
            "• One-click PyMOL (.pml) & ChimeraX (.cxc) scripts\n"
            "• Complete reproducibility audit trail (JSON)",
            ha="center", va="top", fontsize=7.2, color="#047857", linespacing=1.35, zorder=4)

    fig.savefig(out_path, dpi=300, facecolor=fig.get_facecolor(), edgecolor="none")
    plt.close(fig)

    # Also copy to 2rjy_benchmark and user Downloads folder for easy viewing!
    dest_copies = [
        Path("/Users/nirav/Docs/molscope-results/runs/2rjy_benchmark/molscope_graphical_abstract.png"),
        Path("/Users/nirav/Downloads/molscope_graphical_abstract.png"),
    ]
    for cp in dest_copies:
        shutil.copy2(out_path, cp)

    print(f"Graphical abstract successfully generated: {out_path} ({out_path.stat().st_size / (1024*1024):.2f} MB)")
    print(f"Copied to: {dest_copies}")
    return out_path


if __name__ == "__main__":
    out_target = Path("/Users/nirav/Docs/molscope/docs/images/molscope_graphical_abstract.png")
    create_graphical_abstract(out_target)
