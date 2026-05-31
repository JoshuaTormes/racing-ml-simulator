#!/usr/bin/env python3
"""
Monitor ao vivo do treino do racing-ml-sim.

Le os CSVs que o TrainingSession escreve (precisa rodar o treino com --log-csv)
e redesenha as curvas sozinho a cada par de segundos:

  - held_out_log.csv  -> progresso (prog) nos mapas de validacao por geracao
  - training_log.csv   -> taxa de conclusao por mapa de treino + agg_best/agg_mean

Uso (com a janela atualizando ao vivo):
    python3 tools/watch_training.py out_3maps \
        --train-names map1_chicanes,map4_obstaculos,map5_tecnico \
        --val-names   map7_pesadelo,map8_caos_total

Os nomes sao opcionais (so deixam a legenda bonita). Sem eles, usa m0/m1/...

Snapshot estatico (salva PNG e sai, util pra headless/SSH):
    python3 tools/watch_training.py out_3maps --once --save curva.png
"""
import argparse
import os
import sys
import time

import pandas as pd

try:
    import matplotlib
    import matplotlib.pyplot as plt
except Exception as e:  # pragma: no cover
    print("matplotlib nao disponivel:", e, file=sys.stderr)
    sys.exit(1)


def safe_read_csv(path):
    """Le um CSV que pode estar sendo escrito agora (pula linha de cauda parcial)."""
    if not os.path.exists(path) or os.path.getsize(path) == 0:
        return None
    try:
        return pd.read_csv(path, on_bad_lines="skip")
    except Exception:
        return None


def names_or_indices(names, n, prefix):
    if names:
        out = list(names)
    else:
        out = []
    while len(out) < n:
        out.append(f"{prefix}{len(out)}")
    return out[:n]


def n_maps_training(cols):
    # 8 colunas base + 10 por mapa
    return max(0, (len(cols) - 8) // 10)


def n_maps_heldout(cols):
    # 1 coluna (gen) + 3 por mapa de validacao
    return max(0, (len(cols) - 1) // 3)


def draw(fig, axes, out_dir, train_names, val_names):
    ax_val, ax_comp, ax_agg = axes
    for ax in axes:
        ax.clear()

    status_bits = []

    # ---- Held-out (validacao): a curva principal ---------------------------
    held = safe_read_csv(os.path.join(out_dir, "held_out_log.csv"))
    if held is not None and len(held) > 0 and "gen" in held.columns:
        nv = n_maps_heldout(held.columns)
        vnames = names_or_indices(val_names, nv, "val m")
        g = held["gen"]
        for i in range(nv):
            col = f"m{i}_progress"
            if col in held.columns:
                y = pd.to_numeric(held[col], errors="coerce")
                ax_val.plot(g, y, marker="o", ms=4, label=vnames[i])
                if len(y.dropna()) > 0:
                    last = y.dropna().iloc[-1]
                    rcol = f"m{i}_done_reason"
                    reason = held[rcol].dropna().iloc[-1] if rcol in held.columns and len(held[rcol].dropna()) else "?"
                    status_bits.append(f"{vnames[i]}={last:.3f}({reason})")
        ax_val.axhline(0.99, color="green", ls="--", lw=0.8, alpha=0.5)
        ax_val.set_ylim(0, 1.02)
        ax_val.set_ylabel("prog (held-out)")
        ax_val.set_title("Progresso held-out (val: solido | test: tracejado)")
        ax_val.grid(alpha=0.3)
    else:
        ax_val.set_title("Aguardando held_out_log.csv ... (valida a cada 10 gens)")

    # ---- Test set (report-only): same panel, dashed ----------------------------
    test = safe_read_csv(os.path.join(out_dir, "test_log.csv"))
    if test is not None and len(test) > 0 and "gen" in test.columns:
        nt = n_maps_heldout(test.columns)
        tnames = names_or_indices(val_names[len(val_names):] if False else [], nt, "test m")
        g = test["gen"]
        for i in range(nt):
            col = f"m{i}_progress"
            if col in test.columns:
                y = pd.to_numeric(test[col], errors="coerce")
                ax_val.plot(g, y, marker="s", ms=3, ls="--", label=f"test m{i}")
                if len(y.dropna()) > 0:
                    status_bits.append(f"test m{i}={y.dropna().iloc[-1]:.3f}")
    if held is not None and len(held) > 0 and "gen" in held.columns:
        ax_val.legend(loc="lower right", fontsize=8)

    # ---- Taxa de conclusao por mapa de treino ------------------------------
    tr = safe_read_csv(os.path.join(out_dir, "training_log.csv"))
    if tr is not None and len(tr) > 0 and "gen" in tr.columns:
        nm = n_maps_training(tr.columns)
        tnames = names_or_indices(train_names, nm, "train m")
        g = tr["gen"]
        for i in range(nm):
            comp = pd.to_numeric(tr.get(f"m{i}_completed"), errors="coerce")
            col = pd.to_numeric(tr.get(f"m{i}_collision"), errors="coerce")
            sta = pd.to_numeric(tr.get(f"m{i}_stall"), errors="coerce")
            to = pd.to_numeric(tr.get(f"m{i}_timeout"), errors="coerce")
            if comp is None:
                continue
            pop = comp + col + sta + to
            frac = (comp / pop).where(pop > 0)
            ax_comp.plot(g, frac, label=tnames[i])
        ax_comp.set_ylim(0, 1.02)
        ax_comp.set_ylabel("frac. completou")
        ax_comp.set_title("Taxa de conclusao por mapa de treino")
        ax_comp.legend(loc="lower right", fontsize=8)
        ax_comp.grid(alpha=0.3)

        if "agg_best" in tr.columns:
            ax_agg.plot(g, pd.to_numeric(tr["agg_best"], errors="coerce"), label="agg_best", color="tab:green")
        if "agg_mean" in tr.columns:
            ax_agg.plot(g, pd.to_numeric(tr["agg_mean"], errors="coerce"), label="agg_mean", color="tab:gray")
        ax_agg.set_ylabel("agg (rank)")
        ax_agg.set_xlabel("geracao")
        ax_agg.set_title("Fitness agregado")
        ax_agg.legend(loc="lower right", fontsize=8)
        ax_agg.grid(alpha=0.3)

        status_bits.insert(0, f"gen={int(g.iloc[-1])}")
    else:
        ax_comp.set_title("Aguardando training_log.csv ...")

    fig.suptitle("  |  ".join(status_bits) if status_bits else "aguardando dados do treino...",
                 fontsize=10)
    fig.tight_layout(rect=(0, 0, 1, 0.97))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("out_dir", help="pasta passada em --out no treino (ex.: out_3maps)")
    ap.add_argument("--refresh", type=float, default=2.0, help="segundos entre atualizacoes")
    ap.add_argument("--train-names", default="", help="nomes dos mapas de treino, separados por virgula")
    ap.add_argument("--val-names", default="", help="nomes dos mapas de validacao, separados por virgula")
    ap.add_argument("--once", action="store_true", help="desenha uma vez e sai")
    ap.add_argument("--save", default="", help="salva o grafico em PNG (com --once)")
    args = ap.parse_args()

    train_names = [s for s in args.train_names.split(",") if s] if args.train_names else []
    val_names = [s for s in args.val_names.split(",") if s] if args.val_names else []

    fig, axes = plt.subplots(3, 1, figsize=(10, 9), sharex=True)

    if args.once:
        draw(fig, axes, args.out_dir, train_names, val_names)
        if args.save:
            fig.savefig(args.save, dpi=110)
            print("salvo em", args.save)
        else:
            plt.show()
        return

    plt.ion()
    plt.show(block=False)
    print(f"[watch] monitorando '{args.out_dir}' a cada {args.refresh}s. Ctrl+C pra sair.")
    try:
        while True:
            draw(fig, axes, args.out_dir, train_names, val_names)
            fig.canvas.draw_idle()
            plt.pause(args.refresh)
    except KeyboardInterrupt:
        print("\n[watch] encerrado.")


if __name__ == "__main__":
    main()
