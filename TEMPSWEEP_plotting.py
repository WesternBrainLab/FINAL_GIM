
# ═══════════════════════════════════════════════════════════════════════════
# STEP 2 : TEMPERATURE SWEEP
# ═══════════════════════════════════════════════════════════════════════════
sweep = run_temperature_sweep(
   label    = "single sweep",
   t_min    = T_MIN,
   t_max    = T_MAX,
   t_steps  = T_STEPS,
   alpha    = alpha_star,
   save_dir = TEMP_SWEEP_DIR
)


# ── NaN guard ─────────────────────────────────────────────────────────────
corr_arr      = np.array(sweep.corr_ar_total)
spec_heat_arr = np.array(sweep.spec_heat_ar)
suscept_arr   = np.array(sweep.suscept_ar)
T_global = sweep.T_global


n_nan = np.sum(np.isnan(corr_arr))
print(f"NaN correlations in sweep: {n_nan}/{len(corr_arr)}")


# override T_crit and T_best using cleaned arrays
T_suscept_peak = sweep.T_global[stable_peak_index(suscept_arr)]
T_spec_heat_peak = sweep.T_global[stable_peak_index(spec_heat_arr)]
crit_idx = stable_peak_index(spec_heat_arr)
best_idx = np.nanargmax(corr_arr)
T_crit    = sweep.T_global[crit_idx]
T_best    = sweep.T_global[best_idx]
best_corr = np.nanmax(corr_arr)


# also patch the sweep object so downstream code is consistent
sweep.crit_temp  = T_crit
sweep.best_temp  = T_best
sweep.best_corr  = best_corr
sweep.best_ising = sweep.ising_ar[best_idx]
sweep.crit_ising = sweep.ising_ar[crit_idx]


print(f"\nSusceptibility peak temperature        : {T_suscept_peak:.4f}")
print(f"Specific heat peak temperature         : {T_spec_heat_peak:.4f}")
print(f"Critical temperature (specific heat)   : {T_crit:.4f}")
print(
   "Peak detection                         : "
   f"prominence>={PEAK_PROMINENCE_FRACTION:.2f} of curve range, "
   f"edge points ignored={PEAK_IGNORE_EDGE_POINTS}"
)
print(f"Best-match temperature (peak r)        : {T_best:.4f}")
print(f"Best Pearson r                         : {best_corr:.4f}")




# ── observables ───────────────────────────────────────────────────────────
avg_energy = np.array(sweep.avg_energy_ar)
avg_energy_sd = np.array(sweep.avg_energy_sd_ar)
avg_mag = np.array(sweep.avg_mag_ar)
avg_mag_sd = np.array(sweep.avg_mag_sd_ar)
suscept = np.array(sweep.suscept_ar)
suscept_sd = np.array(sweep.suscept_sd_ar)
spec_heat = np.array(sweep.spec_heat_ar)
spec_heat_sd = np.array(sweep.spec_heat_sd_ar)




# ── Figure 1: E, |M|, susceptibility, specific heat vs T ──────────────────
fig1, axes1 = plt.subplots(2, 2, figsize=(13, 9), constrained_layout=True)
fig1.suptitle(
   f"Ising model — temperature sweep  |  alpha = {alpha_star:.3f}",
   fontsize=14,
   fontweight="bold"
)

# reference lines shown on every panel
ref_lines = [
   (T_suscept_peak,   RED,    "--", rf"$T_{{\chi\,peak}}$ = {T_suscept_peak:.2f}"),
   (T_spec_heat_peak, SD_BAND, "--", rf"$T_{{C\,peak}}$ = {T_spec_heat_peak:.2f}"),
   (T_best,           AMBER,   ":",  rf"$T_{{best}}$ = {T_best:.2f}"),
]

panels = [
   (axes1[0, 0], avg_energy, avg_energy_sd, r"average energy $\langle E \rangle$", "Energy vs T"),
   (axes1[0, 1], avg_mag, avg_mag_sd, r"average $|M|$", "|Magnetization| vs T"),
   (axes1[1, 1], suscept, suscept_sd, r"susceptibility $\chi$", "Susceptibility vs T"),
   (axes1[1, 0], spec_heat, spec_heat_sd, r"specific heat $C$", "Specific Heat vs T"),
]

for ax, data, sd, ylabel, title in panels:
   data_plot, sd_plot = temperature_mean_and_sd_band(data, sd)

   ax.plot(T_global, data_plot, color=BLUE, lw=2.0)
   ax.fill_between(T_global, data_plot - sd_plot, data_plot + sd_plot, color=SD_BAND, alpha=0.28, linewidth=0)

   for temp, color, ls, label in ref_lines:
      ax.axvline(temp, color=color, linestyle=ls, lw=1.6, label=label)

   ax.set_xlabel("global temperature  T", fontsize=11)
   ax.set_ylabel(ylabel, fontsize=11)
   ax.set_title(title, fontsize=12)
   ax.legend(fontsize=8, framealpha=0.3)
   ax.spines[["top", "right"]].set_visible(False)

plt.savefig(RESULTS_DIR / "temperature_sweep_3.png", dpi=150, bbox_inches="tight")
plt.close(fig1)
print("Saved: temperature_sweep_3.png")


# ── Figure 2: correlation vs T ────────────────────────────────────────────
fig_corr, ax_corr = plt.subplots(figsize=(7, 4), constrained_layout=True)

corr_total = np.array(sweep.corr_ar_total)
corr_total_sd = np.array(sweep.corr_sd_ar_total)
corr_total_plot, corr_total_sd_plot = temperature_mean_and_sd_band(corr_total, corr_total_sd)

ax_corr.plot(
   T_global,
   corr_total_plot,
   color=BLUE,
   lw=2.0,
   label="avg FC"
)
ax_corr.fill_between(
   T_global,
   corr_total_plot - corr_total_sd_plot,
   corr_total_plot + corr_total_sd_plot,
   color=SD_BAND,
   alpha=0.28,
   linewidth=0,
   label="standard deviation"
)


ax_corr.axvline(T_crit, color=RED, linestyle="--", lw=1.5, label=f"T_crit = {T_crit:.2f}")
ax_corr.axvline(T_best, color=AMBER, linestyle=":", lw=1.5, label=f"T_best = {T_best:.2f}")


ax_corr.set_xlabel("Global temperature  T", fontsize=11)
ax_corr.set_ylabel("Pearson r  (sim Pearson FC vs emp Pearson FC)", fontsize=11)
ax_corr.set_title("Correlation vs Temperature", fontsize=12)
ax_corr.legend(fontsize=9, framealpha=0.3)
ax_corr.spines[["top", "right"]].set_visible(False)


plt.savefig(RESULTS_DIR / "correlation_vs_T_3.png", dpi=150, bbox_inches="tight")
plt.close(fig_corr)
print("Saved: correlation_vs_T_3.png")




# ═══════════════════════════════════════════════════════════════════════════
# STEP 3 : MATRIX COMPARISON  (T_best) — Pearson FC only
# ════
print("STEP 3 : MATRIX COMPARISON  (T_best, Pearson FC)")
print("=" * 65)


best_gd = sweep.best_ising
sim_FC  = best_gd.FC.copy()
Jij_mat = best_gd.Jij.copy()


set_fc_diagonal(sim_FC)


sim_FC_vec = clean_vec(fc_compare_vec(sim_FC))


r_best    = safe_pearson(sim_FC_vec, rho_emp_vec)
dist_best = np.linalg.norm(sim_FC_vec - rho_emp_vec)
diss_best = 1.0 - r_best


print(f"sim FC neg fraction : {np.mean(sim_FC_vec  < 0):.4f}")
print(f"emp FC neg fraction : {np.mean(rho_emp_vec < 0):.4f}")
print(f"sim FC range        : {sim_FC_vec.min():.4f} → {sim_FC_vec.max():.4f}")
print(f"emp FC range        : {rho_emp_vec.min():.4f} → {rho_emp_vec.max():.4f}")
print(f"r              = {r_best:.4f}")
print(f"eucl. distance = {dist_best:.4f}")
print(f"dissimilarity  = {diss_best:.4f}")




# ── color normalization ──────────────────────────────────────────────────
# Use one fixed shared norm for simulated and empirical FC.
fc_lim = 0.5
fc_norm = TwoSlopeNorm(vmin=-fc_lim, vcenter=0, vmax=fc_lim)


# Use separate norm for Jij because it may have a different scale.
j_offdiag = Jij_mat[~np.eye(Jij_mat.shape[0], dtype=bool)]
j_lim = np.percentile(np.abs(j_offdiag), 99)


if not np.isfinite(j_lim) or j_lim < 0.05:
   j_lim = 0.2


j_norm = TwoSlopeNorm(vmin=-j_lim, vcenter=0, vmax=j_lim)


print(f"FC color limit  : ±{fc_lim:.4f}")
print(f"Jij color limit : ±{j_lim:.4f}")




# ── matrix figure ────────────────────────────────────────────────────────
fig3, axes3 = plt.subplots(1, 3, figsize=(15, 5), constrained_layout=True)


fig3.suptitle(
   f"Matrix comparison  |  T_best={T_best:.2f}  |  alpha={alpha_star:.2f}  |  r={r_best:.4f}  |  threshold={THRESHOLD:g}",
   fontsize=13,
   fontweight="bold"
)


matrix_panels = [
   (sim_FC,  f"Simulated Pearson FC\n(T={T_best:.2f}, alpha={alpha_star:.2f})", fc_norm),
   (rho_emp, "Empirical Pearson FC", fc_norm),
   (Jij_mat, "Structural connectivity  $J_{ij}$", j_norm),
]


for ax, (mat, title, norm_to_use) in zip(axes3, matrix_panels):
   im = ax.matshow(mat, cmap="RdBu_r", norm=norm_to_use)
   ax.set_title(title, fontsize=11, pad=12)
   ax.set_xlabel("region", fontsize=9)
   ax.set_ylabel("region", fontsize=9)
   plt.colorbar(im, ax=ax, fraction=0.046, pad=0.04)


plt.savefig(RESULTS_DIR / "matrix_comparison_3.png", dpi=150, bbox_inches="tight")
plt.close(fig3)




# ── scatter: sim vs emp ──────────────────────────────────────────────────
fig3s, ax3s = plt.subplots(figsize=(6, 5), constrained_layout=True)


ax3s.scatter(
   rho_emp_vec,
   sim_FC_vec,
   s=2,
   alpha=0.3,
   color=BLUE,
   rasterized=True
)


m, b = np.polyfit(rho_emp_vec, sim_FC_vec, 1)
x_line = np.linspace(rho_emp_vec.min(), rho_emp_vec.max(), 200)


ax3s.plot(x_line, m * x_line + b, color="black", lw=1.5, linestyle="--")


ax3s.set_xlabel("empirical Pearson FC", fontsize=11)
ax3s.set_ylabel("simulated Pearson FC", fontsize=11)
ax3s.set_title(f"Sim vs Emp Pearson FC  (r = {r_best:.4f})", fontsize=12)
ax3s.spines[["top", "right"]].set_visible(False)


plt.savefig(RESULTS_DIR / "scatter_sim_vs_emp_3.png", dpi=150, bbox_inches="tight")
plt.close(fig3s)


print("Saved: matrix_comparison_3.png, scatter_sim_vs_emp_3.png")




# ── additional matrix comparisons after Tcrit ────────────────────────────
post_crit_indices = np.where(T_global > T_crit)[0]
best_idx = int(np.nanargmax(corr_arr))
post_crit_indices = post_crit_indices[post_crit_indices != best_idx]
post_crit_indices = evenly_spaced_indices(post_crit_indices, N_POST_CRIT_MATRICES)


if len(post_crit_indices) > 0:
   fig3_post, axes3_post = plt.subplots(
       len(post_crit_indices),
       3,
       figsize=(15, 3.8 * len(post_crit_indices)),
       constrained_layout=True,
       squeeze=False,
   )


   fig3_post.suptitle(
       f"Post-critical matrix comparisons  |  Tcrit={T_crit:.2f}  |  alpha={alpha_star:.2f}",
       fontsize=13,
       fontweight="bold"
   )


   print("\nPost-critical matrix comparisons:")


   for row, idx in enumerate(post_crit_indices):
       T_here = T_global[idx]
       gd_here = sweep.ising_ar[idx]
       sim_here = gd_here.FC.copy()
       set_fc_diagonal(sim_here)


       sim_here_vec = clean_vec(fc_compare_vec(sim_here))
       r_here = safe_pearson(sim_here_vec, rho_emp_vec)
       dist_here = np.linalg.norm(sim_here_vec - rho_emp_vec)


       print(f"  T={T_here:.4f}  r={r_here:.4f}  dist={dist_here:.4f}")


       row_panels = [
           (sim_here, f"Simulated Pearson FC\nT={T_here:.2f}, r={r_here:.4f}", fc_norm),
           (rho_emp, "Empirical Pearson FC", fc_norm),
           (Jij_mat, "Structural connectivity  $J_{ij}$", j_norm),
       ]


       for ax, (mat, title, norm_to_use) in zip(axes3_post[row], row_panels):
           im = ax.matshow(mat, cmap="RdBu_r", norm=norm_to_use)
           ax.set_title(title, fontsize=10, pad=10)
           ax.set_xlabel("region", fontsize=8)
           ax.set_ylabel("region", fontsize=8)
           plt.colorbar(im, ax=ax, fraction=0.046, pad=0.04)


   plt.savefig(RESULTS_DIR / "matrix_comparisons_post_Tcrit_3.png", dpi=150, bbox_inches="tight")
   plt.close(fig3_post)
   print("Saved: matrix_comparisons_post_Tcrit_3.png")
else:
   print("No post-critical temperatures available for extra matrix comparisons.")

