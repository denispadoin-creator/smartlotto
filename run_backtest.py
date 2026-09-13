#!/usr/bin/env python3
"""
Esecuzione completa dell'analisi SmartLotto sull'archivio storico.

Uso:
    python run_backtest.py [percorso_archivio] [--warmup N] [--taglio AAAA-MM-GG]

Produce:
    - un report leggibile a video
    - report_backtest.json  (metriche complete)
    - report_backtest.txt   (stesso report, in testo)

Metodo: backtesting walk-forward a finestra espandente su tutte le ruote, con
confronto rigoroso contro la giocata casuale (probabilità teorica 5/90 per
l'estratto), più ricerca di formule con validazione out-of-sample.
"""

from __future__ import annotations
import sys, json, argparse, io
from datetime import datetime

from smartlotto import ingestion, regole, backtest as B, formule, RUOTE, RUOTE_NOMI
from smartlotto.backtest import P_ESTRATTO


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("archivio", nargs="?", default="data/archivio_lotto.txt")
    ap.add_argument("--warmup", type=int, default=300,
                    help="estrazioni iniziali usate solo per 'scaldare' lo stato")
    ap.add_argument("--taglio", default="2010-01-01",
                    help="data di separazione in-sample/out-of-sample per la ricerca formule")
    ap.add_argument("--regole", default=None,
                    help="elenco di chiavi regola separate da virgola (default: tutte). "
                         "Es: ritardatario,frequente_caldo,fibonacci")
    ap.add_argument("--categorie", default=None,
                    help="filtra per categoria: Matematiche,Numerologiche,Astrologiche")
    ap.add_argument("--elenca", action="store_true", help="stampa l'elenco delle regole e termina")
    args = ap.parse_args()

    if args.elenca:
        for cat, elenco in regole.regole_per_categoria().items():
            print(f"\n[{cat}]")
            for chiave, nome in elenco:
                print(f"  {chiave:28s} {nome}")
        return

    out = io.StringIO()
    def stampa(*a):
        s = " ".join(str(x) for x in a)
        print(s); out.write(s + "\n")

    stampa("=" * 70)
    stampa("SMARTLOTTO — ANALISI PREDITTIVA WALK-FORWARD (finestra espandente)")
    stampa("Eseguita il", datetime.now().strftime("%Y-%m-%d %H:%M"))
    stampa("=" * 70)

    df = ingestion.carica_archivio(args.archivio)
    rip = ingestion.riepilogo(df)
    stampa(f"\nArchivio: {rip['estrazioni_totali_righe']} righe (data,ruota) — "
           f"{rip['date_distinte']} date distinte")
    stampa(f"Periodo: {rip['prima_data']} -> {rip['ultima_data']}")
    stampa(f"Ruote: {', '.join(rip['ruote'])}")
    stampa(f"Probabilita' teorica dell'estratto su una ruota: 5/90 = {P_ESTRATTO*100:.4f}%")
    stampa(f"Warmup: {args.warmup} estrazioni  |  (le previsioni partono dopo il warmup)")

    # ------------------------------------------------------------------
    # BACKTEST WALK-FORWARD su tutte le ruote (regole selezionabili)
    # ------------------------------------------------------------------
    chiavi = None
    if args.regole:
        chiavi = [k.strip() for k in args.regole.split(",") if k.strip()]
    elif args.categorie:
        cats = {c.strip() for c in args.categorie.split(",")}
        chiavi = [r[0] for r in regole.REGISTRO if r[2] in cats]
    strategie = regole.costruisci_strategie(chiavi)
    stampa(f"\nRegole selezionate: {len(strategie)} — {', '.join(s.nome for s in strategie)}")
    stampa("\nEsecuzione walk-forward su tutte le ruote... (ogni previsione usa solo il passato)")

    per_ruota = []
    for r in RUOTE:
        sub = ingestion.matrice_ruota(df, r)
        if len(sub) <= args.warmup + 10:
            continue
        res = B.backtest_ruota(sub, strategie, warmup=args.warmup)
        per_ruota.append(res)
        stampa(f"  {RUOTE_NOMI[r]:10s}: {len(sub)} estrazioni analizzate")

    agg = B.aggrega(per_ruota)

    stampa("\n" + "-" * 70)
    stampa("RISULTATO AGGREGATO SU TUTTE LE RUOTE (estratto)")
    stampa("-" * 70)
    stampa(f"{'strategia':22s} {'giocate':>9s} {'successi':>9s} {'tasso%':>8s} "
           f"{'atteso%':>8s} {'z':>7s} {'p-value':>9s}")
    risultati_json = {}
    for nome, r in sorted(agg.items(), key=lambda kv: kv[1].statistica()["z"], reverse=True):
        st = r.statistica()
        atteso_pct = P_ESTRATTO * 100
        stampa(f"{nome:22s} {r.giocate:9d} {r.successi:9d} "
               f"{r.tasso_per_numero*100:8.4f} {atteso_pct:8.4f} "
               f"{st['z']:7.2f} {st['p_value']:9.4f}")
        risultati_json[nome] = {
            "giocate": r.giocate, "successi": r.successi,
            "tasso_per_numero": r.tasso_per_numero,
            "tasso_estrazioni_vincenti": r.tasso_vincenti,
            "attesi_dal_caso": st["attesi"], "z": st["z"], "p_value": st["p_value"],
        }

    stampa("\nLettura: 'tasso%' e' quanto la strategia ha azzeccato per numero giocato;")
    stampa("'atteso%' e' quanto avrebbe fatto una giocata a caso (5/90). Un z alto e un")
    stampa("p-value piccolo indicherebbero un vantaggio reale; z vicino a 0 = come il caso.")

    # ------------------------------------------------------------------
    # RICERCA FORMULE + VALIDAZIONE OUT-OF-SAMPLE
    # ------------------------------------------------------------------
    stampa("\n" + "-" * 70)
    stampa(f"MOTORE DI RICERCA FORMULE  (in-sample < {args.taglio} <= out-of-sample)")
    stampa("-" * 70)
    ff = formule.cerca_formule(df, args.taglio, top=10)
    stampa(f"Regole candidate esaminate: {ff['n_regole']}")
    stampa(f"Regole 'significative' in-sample a p<0.05: {ff['significative_p05']}")
    stampa(f"  ...ma per puro caso a p<0.05 ne attendiamo circa: {ff['attesi_falsi_positivi_p05']:.0f}")
    stampa(f"Regole che superano la soglia Bonferroni (p<{ff['soglia_bonferroni']:.2e}): {ff['significative_bonferroni']}")
    stampa(f"Correlazione rendimento passato vs futuro (tutte le regole): {ff['correlazione_in_out']:.3f}")
    stampa(f"  (vicina a 0 = il rendimento passato NON predice quello futuro)")
    stampa(f"\nLe 10 migliori regole nel passato, e come si comportano nel futuro:")
    stampa(f"{'regola':28s} {'tasso_in%':>10s} {'tasso_out%':>11s} {'caso%':>7s}")
    for r in ff["migliori_in_sample"]:
        tout = r["tasso_out"]
        tout_s = f"{tout*100:10.4f}" if tout == tout else "     n/a"
        stampa(f"{r['regola']:28s} {r['tasso_in']*100:10.4f} {tout_s:>11s} {P_ESTRATTO*100:7.4f}")
    stampa(f"\nDelle 10 regole 'migliori' nel passato, quelle che battono ancora il caso nel futuro: "
           f"{ff['migliori_che_reggono_out']}/10")

    risultati_json["_ricerca_formule"] = {
        k: v for k, v in ff.items() if k != "migliori_in_sample"
    }
    risultati_json["_ricerca_formule"]["migliori_in_sample"] = ff["migliori_in_sample"]
    risultati_json["_meta"] = {"periodo": [rip["prima_data"], rip["ultima_data"]],
                               "warmup": args.warmup, "taglio": args.taglio}

    with open("report_backtest.json", "w", encoding="utf-8") as fh:
        json.dump(risultati_json, fh, ensure_ascii=False, indent=2, default=float)
    with open("report_backtest.txt", "w", encoding="utf-8") as fh:
        fh.write(out.getvalue())

    stampa("\nSalvati: report_backtest.json, report_backtest.txt")
    stampa("=" * 70)


if __name__ == "__main__":
    main()
