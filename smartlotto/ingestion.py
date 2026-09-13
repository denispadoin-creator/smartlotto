"""
Ingestione, pulizia e strutturazione dell'archivio storico.

Formato del file di archivio (una riga per data di estrazione):
    AAAAMMGG RUOTA:n.n.n.n.n RUOTA:n.n.n.n.n ...
Esempio:
    20260911 BA:53.20.40.62.16 CA:33.22.88.17.06 ... VE:26.87.14.71.07

Restituisce due strutture pandas:
  - draws_long: un record per (data, ruota) con le 5 posizioni n1..n5
  - draws_wide: comodo per lookup rapido, indicizzato per (data, ruota)
"""

from __future__ import annotations
import datetime as _dt
import pandas as pd
import numpy as np

from . import RUOTE


def carica_archivio(percorso: str) -> pd.DataFrame:
    """Legge il file di testo e produce un DataFrame 'lungo'.

    Colonne: data (datetime64), ruota (str), n1..n5 (int8), numeri (frozenset).
    Ogni riga = un'estrazione su una ruota in una certa data.
    """
    records = []
    with open(percorso, "r", encoding="utf-8") as fh:
        for riga in fh:
            riga = riga.strip()
            if not riga:
                continue
            campi = riga.split(" ")
            data_str = campi[0]
            # Parsing rigido della data AAAAMMGG.
            data = _dt.date(int(data_str[0:4]), int(data_str[4:6]), int(data_str[6:8]))
            for blocco in campi[1:]:
                ruota, numeri_str = blocco.split(":")
                nums = [int(x) for x in numeri_str.split(".")]
                if len(nums) != 5:
                    raise ValueError(f"Estrazione non valida (attesi 5 numeri): {blocco} @ {data_str}")
                if any(not (1 <= x <= 90) for x in nums):
                    raise ValueError(f"Numero fuori range 1..90: {blocco} @ {data_str}")
                records.append((data, ruota, *nums))

    df = pd.DataFrame(records, columns=["data", "ruota", "n1", "n2", "n3", "n4", "n5"])
    df["data"] = pd.to_datetime(df["data"])
    for c in ["n1", "n2", "n3", "n4", "n5"]:
        df[c] = df[c].astype("int16")
    df = df.sort_values(["data", "ruota"], kind="stable").reset_index(drop=True)
    # Insieme dei 5 numeri, utile per i controlli di appartenenza.
    df["numeri"] = df[["n1", "n2", "n3", "n4", "n5"]].apply(lambda r: frozenset(int(x) for x in r), axis=1)
    return df


def matrice_ruota(df: pd.DataFrame, ruota: str) -> pd.DataFrame:
    """Sottoinsieme ordinato per una singola ruota (indice = data)."""
    sub = df[df["ruota"] == ruota].sort_values("data", kind="stable").reset_index(drop=True)
    return sub


def serie_date(df: pd.DataFrame) -> list[pd.Timestamp]:
    """Elenco ordinato e unico delle date di estrazione presenti nell'archivio."""
    return sorted(df["data"].unique())


def riepilogo(df: pd.DataFrame) -> dict:
    """Statistiche descrittive di base sull'archivio, per verifica integrità."""
    per_ruota = df.groupby("ruota")["data"].agg(["count", "min", "max"])
    return {
        "estrazioni_totali_righe": int(len(df)),
        "date_distinte": int(df["data"].nunique()),
        "prima_data": df["data"].min().date().isoformat(),
        "ultima_data": df["data"].max().date().isoformat(),
        "ruote": sorted(df["ruota"].unique().tolist()),
        "conteggio_per_ruota": {r: int(per_ruota.loc[r, "count"]) for r in per_ruota.index},
    }


def matrice_presenza(sub_ruota: pd.DataFrame) -> np.ndarray:
    """Matrice booleana (n_estrazioni x 91): presenza[i, k] = numero k estratto
    all'estrazione i su questa ruota. La colonna 0 non è usata (numeri 1..90).

    È la struttura base per calcoli vettorizzati di frequenze e ritardi.
    """
    n = len(sub_ruota)
    P = np.zeros((n, 91), dtype=bool)
    for i, (_, row) in enumerate(sub_ruota.iterrows()):
        for k in (row.n1, row.n2, row.n3, row.n4, row.n5):
            P[i, k] = True
    return P
