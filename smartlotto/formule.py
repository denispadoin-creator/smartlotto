"""
Motore di ricerca euristico di formule/regole + validazione out-of-sample.

Questo modulo cerca, fra migliaia di regole candidate, quelle che nel PASSATO
(periodo "in-sample") sembrano battere il caso. Poi le rimette alla prova sul
FUTURO mai visto (periodo "out-of-sample"). È il modulo che rende visibile in
modo quantitativo il fenomeno dell'overfitting: cercando abbastanza regole se ne
trova sempre qualcuna "vincente" nel passato, ma quasi mai regge sul futuro.

Per questo la ricerca applica:
  - una separazione temporale netta in-sample / out-of-sample (niente look-ahead)
  - una correzione per confronti multipli (Bonferroni) e il conteggio dei falsi
    positivi ATTESI per puro caso, da confrontare con quelli osservati.
"""

from __future__ import annotations
import numpy as np
from dataclasses import dataclass
from scipy import stats

from . import numerologia
from .backtest import P_ESTRATTO


@dataclass
class Regola:
    """Regola deterministica: gioca `numero` quando `condizione(data)` è vera.
    Se `condizione` è None la regola gioca sempre quel numero."""
    numero: int
    etichetta: str
    condizione: object = None  # callable(date)->bool oppure None

    def attiva(self, data) -> bool:
        return True if self.condizione is None else bool(self.condizione(data))


def _riga(etichetta: str, numero: int, gi: int, si: int, go: int, so: int) -> dict:
    """Costruisce la riga-risultato di una regola con il p-value binomiale in-sample."""
    p_in = float(stats.binom.sf(si - 1, gi, P_ESTRATTO)) if gi else 1.0
    return {
        "regola": etichetta, "numero": numero,
        "giocate_in": gi, "successi_in": si,
        "tasso_in": si / gi if gi else float("nan"), "p_in": p_in,
        "giocate_out": go, "successi_out": so,
        "tasso_out": so / go if go else float("nan"),
    }


def genera_regole() -> list[Regola]:
    """Genera una famiglia ampia di regole candidate:
      - 90 regole "gioca sempre il numero N"
      - 90 numeri x 9 cicli pitagorici = 810 regole condizionate
    Totale 900 regole. Nessuna ha fondamento causale: è proprio il punto.
    """
    regole = []
    for n in range(1, 91):
        regole.append(Regola(n, f"sempre {n}", None))
    for ciclo in range(1, 10):
        cond = (lambda c: (lambda d: numerologia.ciclo_pitagorico(d) == c))(ciclo)
        for n in range(1, 91):
            regole.append(Regola(n, f"{n} se ciclo pitagorico={ciclo}", cond))
    return regole


def _conteggi(df) -> tuple[np.ndarray, np.ndarray, int]:
    """Precalcolo vettorizzato per la ricerca formule.

    Ritorna:
      occ[c, n]  = quante estrazioni con ciclo pitagorico c (1..9) hanno il numero n
      occ_all[n] = quante estrazioni in totale hanno il numero n
      tot_ciclo[c] = quante estrazioni (righe data,ruota) hanno ciclo pitagorico c
    Una sola passata sui dati; ogni data ha il suo ciclo calcolato una volta.
    """
    from . import numerologia
    occ = np.zeros((10, 91), dtype=np.int64)   # riga 0 non usata
    tot_ciclo = np.zeros(10, dtype=np.int64)
    cache_ciclo: dict = {}
    for data, insieme in zip(df["data"].values, df["numeri"].values):
        d = np.datetime64(data, "D").astype("datetime64[D]").astype(object)
        c = cache_ciclo.get(d)
        if c is None:
            c = numerologia.ciclo_pitagorico(d)
            cache_ciclo[d] = c
        tot_ciclo[c] += 1
        for n in insieme:
            occ[c, n] += 1
    occ_all = occ.sum(axis=0)
    return occ, occ_all, int(tot_ciclo.sum()), tot_ciclo


def cerca_formule(df, data_taglio: str, top: int = 15) -> dict:
    """Cerca le migliori regole in-sample e le valida out-of-sample.

    data_taglio: 'AAAA-MM-GG'. In-sample = estrazioni < taglio; out = >= taglio.
    Ritorna un dizionario con classifiche e metriche di overfitting.
    """
    taglio = np.datetime64(data_taglio)
    df_in = df[df["data"] < taglio]
    df_out = df[df["data"] >= taglio]

    occ_in, occ_all_in, tot_in, totc_in = _conteggi(df_in)
    occ_out, occ_all_out, tot_out, totc_out = _conteggi(df_out)

    righe = []
    # Regole "sempre N"
    for n in range(1, 91):
        gi, si = tot_in, int(occ_all_in[n])
        go, so = tot_out, int(occ_all_out[n])
        righe.append(_riga(f"sempre {n}", n, gi, si, go, so))
    # Regole condizionate "N se ciclo=c"
    for c in range(1, 10):
        for n in range(1, 91):
            gi, si = int(totc_in[c]), int(occ_in[c, n])
            go, so = int(totc_out[c]), int(occ_out[c, n])
            if gi == 0:
                continue
            righe.append(_riga(f"{n} se ciclo pitagorico={c}", n, gi, si, go, so))

    righe = [r for r in righe if r["giocate_in"] > 0]
    n_regole = len(righe)
    soglia_bonferroni = 0.05 / n_regole if n_regole else 0.05
    significative_in = [r for r in righe if r["p_in"] < 0.05]
    significative_bonf = [r for r in righe if r["p_in"] < soglia_bonferroni]
    attesi_falsi_positivi = 0.05 * n_regole  # a p<0.05, per puro caso

    # Migliori in-sample per p-value, con il loro esito out-of-sample
    migliori = sorted(righe, key=lambda r: r["p_in"])[:top]

    # Le migliori in-sample battono il caso anche out? Confronto dei tassi.
    out_sopra_media = [r for r in migliori
                       if not np.isnan(r["tasso_out"]) and r["tasso_out"] > P_ESTRATTO]

    # Correlazione fra rendimento in-sample e out-of-sample su TUTTE le regole
    tin = np.array([r["tasso_in"] for r in righe])
    tout = np.array([r["tasso_out"] for r in righe])
    mask = ~np.isnan(tout)
    if mask.sum() > 2:
        corr = float(np.corrcoef(tin[mask], tout[mask])[0, 1])
    else:
        corr = float("nan")

    return {
        "n_regole": n_regole,
        "soglia_bonferroni": soglia_bonferroni,
        "significative_p05": len(significative_in),
        "attesi_falsi_positivi_p05": attesi_falsi_positivi,
        "significative_bonferroni": len(significative_bonf),
        "migliori_in_sample": migliori,
        "migliori_che_reggono_out": len(out_sopra_media),
        "correlazione_in_out": corr,
        "p_estratto_teorico": P_ESTRATTO,
    }
