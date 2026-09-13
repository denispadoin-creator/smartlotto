#!/usr/bin/env python3
"""
Ciclo settimanale automatico di SmartLotto.

Fa tre cose, in ordine:
  1) scarica le estrazioni mancanti e le aggiunge all'archivio;
  2) verifica la previsione precedente contro i numeri realmente estratti,
     evidenziando gli azzeccati;
  3) genera e salva la nuova previsione per la prossima estrazione.

Uso:
    python aggiorna_e_prevedi.py [archivio] [previsioni.json]

È lo script da lanciare a mano o da programmare ogni settimana (vedi
aggiorna_settimanale.bat per Windows).
"""

from __future__ import annotations
import sys
import datetime as _dt

from smartlotto import ingestion, aggiornamento, predittore, tracker
from smartlotto.tracker import prossimo_giorno_estrazione

VERDE = "\033[92m"; GRASSETTO = "\033[1m"; FINE = "\033[0m"


def main():
    archivio = sys.argv[1] if len(sys.argv) > 1 else "data/archivio_lotto.txt"
    prev = sys.argv[2] if len(sys.argv) > 2 else "previsioni.json"

    print(GRASSETTO + "SmartLotto — ciclo settimanale" + FINE,
          _dt.datetime.now().strftime("%d/%m/%Y %H:%M"))

    # 1) Aggiornamento archivio
    try:
        n = aggiornamento.aggiorna_archivio(archivio)
        print(f"[1] Archivio: {n} nuove estrazioni scaricate."
              if n else "[1] Archivio già aggiornato.")
    except Exception as e:
        print(f"[1] Download non riuscito: {e}")

    df = ingestion.carica_archivio(archivio)

    # 2) Verifica della previsione precedente
    voci = tracker.verifica_previsioni(prev, df)
    ultima = tracker.ultima_verificata(prev)
    if ultima:
        print(f"\n[2] Verifica previsione del {ultima['data_estrazione']}:")
        tot = 0
        for ruota, e in ultima["esiti"].items():
            azz = set(e["azzeccati"])
            numeri = "  ".join(
                (f"{VERDE}{GRASSETTO}{x:02d}{FINE}" if x in azz else f"{x:02d}")
                for x in e["previsti"])
            print(f"    {ruota:11s} previsti: {numeri}   → azzeccati: {e['n_azzeccati']}/5")
            tot += e["n_azzeccati"]
        print(f"    Totale numeri azzeccati: {tot}")
    else:
        print("\n[2] Nessuna previsione precedente da verificare.")

    # 3) Nuova previsione per la prossima estrazione
    ultima_data = df["data"].max().date()
    data_prossima = prossimo_giorno_estrazione(ultima_data)
    res = predittore.previsione_tutte_le_ruote(df, data_prossima)
    tracker.registra_previsione(
        prev, data_prossima,
        {r: nums for r, nums in res["ruote"].items()})
    print(f"\n[3] Nuova previsione per l'estrazione del {data_prossima.strftime('%d/%m/%Y')}:")
    for ruota, nums in res["ruote"].items():
        print(f"    {ruota:11s}: " + "  ".join(f"{x:02d}" for x in nums))
    print("\nSalvata in", prev, "— sarà verificata al prossimo aggiornamento.")


if __name__ == "__main__":
    main()
