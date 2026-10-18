#!/usr/bin/env python3
"""Move-variable QUBO (Section 4): y[i,B] = 1 moves replica i to node B; all zeros is the current placement."""
import math
from collections import defaultdict

from qubo_index_placement import QUBO
from paper_qubo_verify import RESOURCES, footprint, capacity
from paper_qubo_experiment import _add_squared


def yv(rep_name, dest):
    return f"y|{rep_name}|{dest}"


def _add_product(q, t1, c1, t2, c2, w):
    for v1, a1 in t1:
        for v2, a2 in t2:
            if v1 == v2:
                q.add_linear(v1, w * a1 * a2)
            else:
                q.add_quadratic(v1, v2, w * a1 * a2)
    for v1, a1 in t1:
        q.add_linear(v1, w * a1 * c2)
    for v2, a2 in t2:
        q.add_linear(v2, w * a2 * c1)
    q.add_offset(w * c1 * c2)


def load_expr(inst, P0, res, node_name):
    nodes = [n.name for n in inst.usable_nodes()]
    terms = []
    used_cur = 0.0
    for rep in inst.replicas:
        a = footprint(rep, res)
        home = P0.get(rep.name)
        if home == node_name:
            used_cur += a
            if a:
                for B in nodes:
                    if B != home:
                        terms.append((yv(rep.name, B), -a))
        elif a:
            terms.append((yv(rep.name, node_name), +a))
    return terms, used_cur


def stay_expr(inst, P0, rep):
    nodes = [n.name for n in inst.usable_nodes()]
    home = P0.get(rep.name)
    return [(yv(rep.name, B), -1.0) for B in nodes if B != home], 1.0


def on_node_expr(inst, P0, rep, node_name):
    if P0.get(rep.name) == node_name:
        return stay_expr(inst, P0, rep)
    return [(yv(rep.name, node_name), 1.0)], 0.0


def new_nodes_of(inst, P0):
    held = {P0.get(r.name) for r in inst.replicas}
    return {n.name for n in inst.usable_nodes() if n.name not in held}


def movement_terms(inst, P0, w_move, nodes, new_nodes=None):
    excluded = {n.name for n in inst.nodes if n.excluded}
    new = set(new_nodes if new_nodes is not None else new_nodes_of(inst, P0))
    terms, charged = [], 0.0
    for rep in inst.replicas:
        home = P0.get(rep.name)
        if home in excluded:
            continue
        charged += w_move[rep.name]
        for B in nodes:
            if B != home and B not in new:
                terms.append((yv(rep.name, B), w_move[rep.name]))
    return terms, charged


def build_move_qubo(inst, P0, lam, gamma=0.0, w_move=None,
                    capacity_mode="slack", kappa=1.0, resources=RESOURCES,
                    beta=None, move_budget=None, move_kappa=1.0, move_slack_bits=0,
                    move_quad=False, onedest_mult=1.0,
                    res_weights=None, cap_slack_bits=0, new_nodes=None):
    q = QUBO()
    nodes = [n.name for n in inst.usable_nodes()]
    node_obj = {n.name: n for n in inst.usable_nodes()}
    N = len(nodes)
    reps = inst.replicas
    if w_move is None:
        w_move = {r.name: footprint(r, "d") for r in reps}
    wmap = {"d": inst.alpha_d, "m": inst.alpha_m, "r": inst.alpha_r}
    nslack = 0

    active = [r for r in resources if wmap[r] != 0]
    rw = {r: (res_weights or {}).get(r, 1.0) for r in active}
    wsum = sum(rw.values()) or 1.0
    for res in active:
        wr = wmap[res]
        total = sum(footprint(r, res) for r in reps)
        mean = total / N
        if beta is None:
            coef = wr
        else:
            coef = beta * rw[res] / (wsum * N * max(mean, 1e-12) ** 2)
        for nm in nodes:
            t, c = load_expr(inst, P0, res, nm)
            _add_squared(q, t, c - mean, coef)

    if move_budget is not None:
        mterms, charged = movement_terms(inst, P0, w_move, nodes, new_nodes)
        mmax = move_budget * charged
        if move_slack_bits:
            g = max(mmax / (2 ** move_slack_bits - 1), 1e-9)
            mterms = mterms + [(f"ms|{b}", g * (2 ** b)) for b in range(move_slack_bits)]
            nslack += move_slack_bits
            mlam = move_kappa * kappa * lam / max(mmax, 1e-9) ** 2
            _add_squared(q, mterms, -mmax, mlam)
        else:
            delta = 0.10 * mmax
            mlam2 = move_kappa * kappa * lam / max(mmax, 1e-9) ** 2
            mlam1 = 2.0 * mlam2 * delta
            _add_squared(q, mterms, -mmax, mlam2)
            for v, a in mterms:
                q.add_linear(v, mlam1 * a)
            q.add_offset(mlam1 * (-mmax))
    elif gamma:
        mterms, charged = movement_terms(inst, P0, w_move, nodes, new_nodes)
        if move_quad:
            wtot = max(charged, 1e-9)
            _add_squared(q, mterms, 0.0, gamma / (wtot * wtot))
        else:
            for v, a in mterms:
                q.add_linear(v, gamma * a)

    ejected = {n.name for n in inst.nodes if n.excluded}
    n_forced = 0
    if ejected:
        for rep in reps:
            if P0.get(rep.name) in ejected:
                _add_squared(q, [(yv(rep.name, B), 1.0) for B in nodes], -1.0, lam)
                n_forced += 1

    for rep in reps:
        home = P0.get(rep.name)
        ds = [B for B in nodes if B != home]
        for a in range(len(ds)):
            for b in range(a + 1, len(ds)):
                q.add_quadratic(yv(rep.name, ds[a]), yv(rep.name, ds[b]),
                                onedest_mult * lam)

    by_logical = defaultdict(list)
    for rep in reps:
        by_logical[rep.logical].append(rep)
    for grp in by_logical.values():
        for a in range(len(grp)):
            for b in range(a + 1, len(grp)):
                for nm in nodes:
                    t1, c1 = on_node_expr(inst, P0, grp[a], nm)
                    t2, c2 = on_node_expr(inst, P0, grp[b], nm)
                    _add_product(q, t1, c1, t2, c2, lam)

    groups_map = defaultdict(list)
    for n in inst.usable_nodes():
        groups_map[n.group].append(n.name)
    if len(groups_map) > 1:
        for grp in by_logical.values():
            R, G = len(grp), len(groups_map)
            for gnodes in groups_map.values():
                terms, const = [], 0.0
                for rep in grp:
                    for nm in gnodes:
                        t, c = on_node_expr(inst, P0, rep, nm)
                        terms.extend(t)
                        const += c
                _add_squared(q, terms, const - float(R) / G, lam)

    for nm in nodes:
        for res in resources:
            cap = capacity(node_obj[nm], res)
            t, c = load_expr(inst, P0, res, nm)
            if capacity_mode == "slack":
                if cap_slack_bits:
                    g = max(cap / (2.0 ** cap_slack_bits - 1), 1e-12)
                    terms = [(v, -a) for v, a in t]
                    for k in range(cap_slack_bits):
                        terms.append((f"sm|{res}|{nm}|{k}", -(g * 2.0 ** k)))
                        nslack += 1
                else:
                    kbits = (int(math.floor(math.log2(cap))) + 1) if cap >= 1 else 1
                    terms = [(v, -a) for v, a in t]
                    for k in range(kbits):
                        terms.append((f"sm|{res}|{nm}|{k}", -(2.0 ** k)))
                        nslack += 1
                _add_squared(q, terms, cap - c, kappa * lam / max(cap, 1e-9) ** 2)
            elif capacity_mode == "unbalanced":
                delta = 0.10 * cap
                lam2 = kappa * lam / max(cap, 1e-9) ** 2
                lam1 = 2.0 * lam2 * delta
                _add_squared(q, t, c - cap, lam2)
                for v, a in t:
                    q.add_linear(v, lam1 * a)
                q.add_offset(lam1 * (c - cap))
            else:
                raise ValueError(capacity_mode)

    return q, {"nodes": nodes, "reps": reps, "n_slack": nslack, "P0": P0,
               "ejected": sorted(ejected), "n_forced": n_forced}


def decode_move(sample, inst, P0):
    nodes = [n.name for n in inst.usable_nodes()]
    ejected = {n.name for n in inst.nodes if n.excluded}
    place, conflicts = {}, 0
    for rep in inst.replicas:
        home = P0.get(rep.name)
        chosen = [B for B in nodes if B != home and int(sample.get(yv(rep.name, B), 0)) == 1]
        if len(chosen) == 1:
            place[rep.name] = chosen[0]
        elif not chosen:
            if home in ejected:
                conflicts += 1
                continue
            place[rep.name] = home
        else:
            conflicts += 1
            place[rep.name] = chosen[0]
    return place, conflicts
