"""Reference table at the main setting (preset `hotspot`, d = 2.2 Mbps, 30 seeds).

Rows: All-on, All-on + MLB rule, the rule-based pair (dead-zone MLB), and two privileged
references: All-on + CIO oracle and the oracle reference (future-aware OracleES +
CioOracleMLB seeing s_{t-1}).  The threshold rho_th of the rule-based pair and the load
margin of the oracle are the best values in the file's sweep for which the average V meets
that file's V_max (best_par).  Learned methods get empty rows.  Every method must be evaluated on the same episodes: chunk k in 0..5 is
RanEnv(make_config('hotspot'), n_envs=5, seed=1000 * k).

    python reference_table.py                                  # files of the study
    python reference_table.py --probe results/probe_hotspot_d2.2.json   # one probe run
"""
import json

import numpy as np

def best_par(path, fam, demand=2.2):
    """Parameter of family fam with the largest energy saving among those whose average V
    meets the file's V_max (the least-violating one if none does)."""
    D = json.load(open(path))
    cname = next(c for c, kw in D["cfgs"].items() if kw.get("demand_mbps") == demand)
    pars = sorted({r["par"] for r in D["rows"] if r["config"] == cname and r["fam"] == fam})
    res = {p: collect(path, fam, p, demand) for p in pars}
    v_max = D["cfgs"][cname].get("v_max", V_MAX)
    ok = [p for p in pars if res[p]["V"] <= v_max]
    return max(ok, key=lambda p: res[p]["ES"]) if ok else min(pars, key=lambda p: res[p]["V"])


def _resolve(rows):
    out = []
    for lab, f, fam, par in rows:
        if par == "best":
            par = best_par(f, fam)
            lab = f"{lab}, {'ρ_th' if fam.startswith('rule') else 'margin'} = {par:g}"
        out.append((lab, f, fam, par))
    return out


def sources(probe=None):
    """(TOP, ORACLE) row sources; with probe, all rows come from that probe.py output.
    Rule-based threshold and oracle margin are chosen against the file's own V_max."""
    if probe is None:
        return _resolve(TOP), _resolve(ORACLE)
    top = [("All-on", probe, "allon", None),
           ("All-on + MLB rule (dead zone)", probe, "allon_ruleT", None),
           ("Rule-based (dead zone)", probe, "ruleT", "best")]
    ora = [("All-on + CIO oracle", probe, "allon_cio", None),
           ("Oracle reference", probe, "orC", "best")]
    return _resolve(top), _resolve(ora)


TOP = [  # (label, file, family, parameter)
    ("All-on", "results/hotspot_nonadj_d2.0_2.2.json", "allon", None),
    ("All-on + MLB rule (dead zone)", "results/hotspot_nonadj_d2.0_2.2.json", "allon_ruleT", None),
    ("Rule-based (dead zone)", "results/hotspot_nonadj_d2.0_2.2.json", "ruleT", "best"),
]
PENDING = ["ES-only learning", "Learned ES, MLB off", "ACCORD", "Centralized"]
ORACLE = [
    ("All-on + CIO oracle", "results/hotspot_nonadj_d2.0_2.2.json", "allon_cio", None),
    ("Oracle reference", "results/simultaneous_prev.json", "orC", "best"),
]
V_MAX = 0.05                     # fallback when a probe file does not record v_max


def collect(path, fam, par, demand=2.2):
    D = json.load(open(path))
    cname = next(c for c, kw in D["cfgs"].items() if kw.get("demand_mbps") == demand)
    v_max = D["cfgs"][cname].get("v_max", V_MAX)         # feasibility against the probe's own V_max
    groups = {}
    for r in sorted(D["rows"], key=lambda r: r["chunk"]):
        if r["config"] == cname:
            g = groups.setdefault((r["fam"], r["par"]), {})
            for k in ("E_sum", "V", "H", "O", "Xi", "jain", "n_off", "n_short"):
                g.setdefault(k, []).extend(r[k])
    g = {k: np.array(v) for k, v in groups[(fam, par)].items()}
    e0 = np.array(groups[("allon", None)]["E_sum"])
    return dict(ES=100 * np.mean(1 - g["E_sum"] / e0), V=g["V"].mean(),
                ok=int((g["V"] <= v_max).sum()), n=len(g["V"]), O=g["O"].mean(),
                H=g["H"].mean(), Xi=g["Xi"].mean(), jain=g["jain"].mean(),
                short=g["n_short"].sum() / max(g["n_off"].sum(), 1))


def main(probe=None):
    top, ora = sources(probe)
    head = ["Method", "Energy saving [%]", "V̄", "Episodes with V ≤ V_max",
            "Ō", "H̄", "Ξ_T", "Short-lived deactivations", "Jain"]
    md = ["| " + " | ".join(head) + " |", "|" + "---|" * len(head)]
    tex = []
    for lab, f, fam, par in top:
        m = collect(f, fam, par)
        md.append(_md(lab, m))
        tex.append(_tex(lab, m))
    for lab in PENDING:
        md.append(f"| {lab} |" + " |" * (len(head) - 1))
        tex.append(f"{lab} & & & & & & & & \\\\")
    tex.append("\\midrule")
    for lab, f, fam, par in ora:
        m = collect(f, fam, par)
        md.append(_md(lab, m))
        tex.append(_tex(lab + "$^\\dagger$", m))
    open("results/reference_d2.2.md", "w").write("\n".join(md) + "\n")
    open("results/reference_d2.2.tex", "w").write("\n".join(tex) + "\n")
    print("\n".join(md))


def _short(m, tex=False):
    if m["Xi"] == 0:
        return "--"
    return f"{100 * m['short']:.0f}" + ("\\%" if tex else " %")


def _md(lab, m):
    return (f"| {lab} | {m['ES']:.1f} | {m['V']:.4f} | {m['ok']}/{m['n']} | {m['O']:.4f} | "
            f"{m['H']:.4f} | {m['Xi']:.1f} | {_short(m)} | {m['jain']:.3f} |")


def _tex(lab, m):
    return (f"{lab} & {m['ES']:.1f} & {m['V']:.3f} & {m['ok']}/{m['n']} & {m['O']:.3f} & "
            f"{m['H']:.3f} & {m['Xi']:.1f} & {_short(m, True)} & {m['jain']:.3f} \\\\")


if __name__ == "__main__":
    import argparse
    ap = argparse.ArgumentParser()
    ap.add_argument("--probe", default=None, help="probe.py output with every family at 2.2 Mbps")
    main(ap.parse_args().probe)
