#!/usr/bin/env python3
"""Detect self-overlap (border self-intersections) in a track map.

Replicates the C++ geometry: centripetal Catmull-Rom centerline
(CENTERLINE_SUBSEGMENTS=10) + miter-offset borders (half-width, miter limit 3).
Reports clusters of border-segment intersections, which is exactly where the
rendered track ribbon crosses itself.

Usage: python3 tools/check_map_overlap.py maps/map1_chicanes_infernais.json
"""
import json, sys, math

N = 10          # CENTERLINE_SUBSEGMENTS
MITER_LIMIT = 3.0


def catmull(p0, p1, p2, p3, u):
    def knot(ti, a, b):
        d = math.sqrt(math.hypot(b[0]-a[0], b[1]-a[1]))  # alpha=0.5
        return ti + max(d, 1e-6)
    t0 = 0.0
    t1 = knot(t0, p0, p1)
    t2 = knot(t1, p1, p2)
    t3 = knot(t2, p2, p3)
    t = t1 + u*(t2-t1)
    def lerp(a, b, w): return (a[0]+(b[0]-a[0])*w, a[1]+(b[1]-a[1])*w)
    A1 = lerp(p0, p1, (t-t0)/(t1-t0))
    A2 = lerp(p1, p2, (t-t1)/(t2-t1))
    A3 = lerp(p2, p3, (t-t2)/(t3-t2))
    B1 = lerp(A1, A2, (t-t0)/(t2-t0))
    B2 = lerp(A2, A3, (t-t1)/(t3-t1))
    return lerp(B1, B2, (t-t1)/(t2-t1))


def build_centerline(wps, closed):
    n = len(wps)
    def wp(i):
        if closed: return wps[((i % n)+n) % n]
        if i < 0:  return (2*wps[0][0]-wps[1][0], 2*wps[0][1]-wps[1][1])
        if i >= n: return (2*wps[n-1][0]-wps[n-2][0], 2*wps[n-1][1]-wps[n-2][1])
        return wps[i]
    cl = []
    trechos = n if closed else n-1
    for i in range(trechos):
        p0, p1, p2, p3 = wp(i-1), wp(i), wp(i+1), wp(i+2)
        for k in range(N):
            cl.append(catmull(p0, p1, p2, p3, k/N))
    cl.append(wps[0] if closed else wps[n-1])
    return cl


def normd(v):
    l = math.hypot(v[0], v[1])
    return (v[0]/l, v[1]/l) if l > 1e-12 else (0.0, 0.0)


def perp(v): return (-v[1], v[0])


def build_borders(cl, hw, closed):
    n = len(cl)
    def prev(i): return i-1 if i > 0 else (n-2 if closed else 0)
    def nxt(i):  return i+1 if i+1 < n else (1 if closed else n-1)
    L, R = [], []
    for i in range(n):
        if not closed and i == 0:
            d1 = d2 = normd((cl[1][0]-cl[0][0], cl[1][1]-cl[0][1]))
        elif not closed and i == n-1:
            d1 = d2 = normd((cl[n-1][0]-cl[n-2][0], cl[n-1][1]-cl[n-2][1]))
        else:
            d1 = normd((cl[i][0]-cl[prev(i)][0], cl[i][1]-cl[prev(i)][1]))
            d2 = normd((cl[nxt(i)][0]-cl[i][0], cl[nxt(i)][1]-cl[i][1]))
        n1, n2 = perp(d1), perp(d2)
        miter = (n1[0]+n2[0], n1[1]+n2[1])
        ml2 = miter[0]**2 + miter[1]**2
        if ml2 < 1e-6:
            off = (n2[0]*hw, n2[1]*hw)
        else:
            m = (miter[0]/math.sqrt(ml2), miter[1]/math.sqrt(ml2))
            dot = m[0]*n1[0]+m[1]*n1[1]
            if dot < 1e-4:
                off = (n2[0]*hw, n2[1]*hw)
            else:
                s = min(hw/dot, hw*MITER_LIMIT)
                off = (m[0]*s, m[1]*s)
        L.append((cl[i][0]+off[0], cl[i][1]+off[1]))
        R.append((cl[i][0]-off[0], cl[i][1]-off[1]))
    return L, R


def seg_intersect(p1, p2, p3, p4):
    d1 = (p2[0]-p1[0], p2[1]-p1[1])
    d2 = (p4[0]-p3[0], p4[1]-p3[1])
    den = d1[0]*d2[1]-d1[1]*d2[0]
    if abs(den) < 1e-12: return None
    t = ((p3[0]-p1[0])*d2[1]-(p3[1]-p1[1])*d2[0])/den
    u = ((p3[0]-p1[0])*d1[1]-(p3[1]-p1[1])*d1[0])/den
    if 0 <= t <= 1 and 0 <= u <= 1:
        return (p1[0]+t*d1[0], p1[1]+t*d1[1])
    return None


def render(path, out, cl, L, R, clusters):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    fig, ax = plt.subplots(figsize=(9, 7))
    ax.set_facecolor("#1e1e1e")
    # Ribbon fill
    poly = L + R[::-1]
    ax.fill([p[0] for p in poly], [p[1] for p in poly],
            color="#3a3a3a", zorder=1)
    ax.plot([p[0] for p in L], [p[1] for p in L], color="white", lw=1, zorder=3)
    ax.plot([p[0] for p in R], [p[1] for p in R], color="white", lw=1, zorder=3)
    ax.plot([p[0] for p in cl], [p[1] for p in cl], color="#888", lw=0.6, zorder=2)
    for c in clusters:
        cx = sum(h[0] for h in c)/len(c); cy = sum(h[1] for h in c)/len(c)
        ax.add_patch(plt.Circle((cx, cy), 28, color="#33ff99", fill=False, lw=2, zorder=5))
    ax.set_aspect("equal"); ax.invert_yaxis()
    ax.set_title(path.split("/")[-1] + f"  ({len(clusters)} overlap regions)")
    fig.tight_layout(); fig.savefig(out, dpi=110); plt.close(fig)
    print(f"  rendered -> {out}")


def main():
    path = sys.argv[1]
    j = json.load(open(path))
    wps = [(float(a), float(b)) for a, b in j["waypoints"]]
    closed = j["closed"]
    hw = j["track_width"]/2.0
    cl = build_centerline(wps, closed)
    W = j["track_width"]

    # True ribbon overlap = the centerline passes within < W of a NON-adjacent
    # part of itself. (A simple miter notch at a corner does not trigger this;
    # only genuine doubling-back / folds where the road covers the road do.)
    # Arc-distance gate: ignore neighbours closer than `arc_gate` along the path.
    cum = [0.0]
    for i in range(1, len(cl)):
        cum.append(cum[-1] + math.hypot(cl[i][0]-cl[i-1][0], cl[i][1]-cl[i-1][1]))
    total = cum[-1]
    arc_gate = W * 1.2

    hits = []
    for i in range(len(cl)):
        for k in range(i+1, len(cl)):
            da = cum[k] - cum[i]
            if closed: da = min(da, total - da)
            if da < arc_gate: continue
            d = math.hypot(cl[i][0]-cl[k][0], cl[i][1]-cl[k][1])
            if d < W:
                mx = (cl[i][0]+cl[k][0])/2
                my = (cl[i][1]+cl[k][1])/2
                hits.append((mx, my, d))

    # Border self-intersections: the actual rendered ribbon edges crossing each
    # other. Catches miter spikes at sharp tooth tips that the centerline metric
    # misses (the little inward "arrows" visible in the app). Reported as gap 0.
    L, R = build_borders(cl, hw, closed)
    segs = [("L", i, L[i], L[i+1]) for i in range(len(L)-1)] \
         + [("R", i, R[i], R[i+1]) for i in range(len(R)-1)]
    nL = len(L) - 1
    for a in range(len(segs)):
        ta, ia, pa1, pa2 = segs[a]
        for b in range(a+1, len(segs)):
            tb, ib, pb1, pb2 = segs[b]
            if ta == tb and abs(ia-ib) <= 1: continue
            if ta == tb and closed and {ia, ib} == {0, nL-1}: continue
            p = seg_intersect(pa1, pa2, pb1, pb2)
            if p: hits.append((p[0], p[1], 0.0))

    # Cluster nearby hits; report worst (smallest) gap per cluster.
    clusters = []
    for h in hits:
        for c in clusters:
            if math.hypot(h[0]-c[0][0], h[1]-c[0][1]) < 45:
                c.append(h); break
        else:
            clusters.append([h])

    print(f"{path}: {len(hits)} overlap samples in {len(clusters)} region(s) "
          f"(W={W:.0f}, gate={arc_gate:.0f})")
    for c in sorted(clusters, key=lambda c: min(h[2] for h in c)):
        cx = sum(h[0] for h in c)/len(c)
        cy = sum(h[1] for h in c)/len(c)
        worst = min(h[2] for h in c)
        print(f"  region ~({cx:.0f},{cy:.0f})  min gap {worst:.0f}px  ({len(c)} samples)")
    if len(sys.argv) > 2:
        render(path, sys.argv[2], cl, L, R, clusters)
    return len(clusters)


if __name__ == "__main__":
    sys.exit(1 if main() else 0)
