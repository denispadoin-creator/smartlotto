"""
Tracciamento delle previsioni: salva ogni pronostico PRIMA dell'estrazione e,
dopo che l'estrazione è avvenuta ed è entrata in archivio, lo confronta con i
numeri realmente usciti, evidenziando quelli azzeccati.

Le previsioni sono salvate in un file JSON (`previsioni.json` di default), una
voce per data di estrazione e ruota.
"""

from __future__ import annotations
import datetime as _dt
import json
import os

from . import ingestion

# Giorni di estrazione del Lotto (feriali: 0=lun ... 6=dom).
# Calendario attuale: martedì, giovedì, venerdì, sabato.
GIORNI_ESTRAZIONE = {1, 3, 4, 5}


def prossimo_giorno_estrazione(da: _dt.date) -> _dt.date:
    """Prima data di estrazione strettamente successiva a `da`."""
    d = da + _dt.timedelta(days=1)
    for _ in range(14):
        if d.weekday() in GIORNI_ESTRAZIONE:
            return d
        d += _dt.timedelta(days=1)
    return da + _dt.timedelta(days=1)


def carica_previsioni(percorso: str) -> list[dict]:
    if not os.path.exists(percorso):
        return []
    with open(percorso, "r", encoding="utf-8") as fh:
        return json.load(fh)


def salva_previsioni(percorso: str, voci: list[dict]) -> None:
    with open(percorso, "w", encoding="utf-8") as fh:
        json.dump(voci, fh, ensure_ascii=False, indent=2)


def registra_previsione(percorso_prev: str, data_estrazione: _dt.date,
                        ruote: dict[str, list[int]]) -> dict:
    """Aggiunge (o sostituisce) la previsione per una certa data di estrazione."""
    voci = carica_previsioni(percorso_prev)
    voci = [v for v in voci if v["data_estrazione"] != data_estrazione.isoformat()]
    voce = {
        "data_estrazione": data_estrazione.isoformat(),
        "generata_il": _dt.datetime.now().isoformat(timespec="seconds"),
        "ruote": {r: sorted(n) for r, n in ruote.items()},
        "verificata": False,
        "esiti": {},
    }
    voci.append(voce)
    voci.sort(key=lambda v: v["data_estrazione"])
    salva_previsioni(percorso_prev, voci)
    return voce


def verifica_previsioni(percorso_prev: str, df) -> list[dict]:
    """Confronta le previsioni non ancora verificate con l'archivio aggiornato.

    Per ogni ruota segna: numeri previsti, numeri realmente estratti, e quali
    numeri previsti sono usciti ('azzeccati'). Ritorna le voci aggiornate.
    """
    voci = carica_previsioni(percorso_prev)
    # Indice rapido: (data_iso, ruota_nome) -> set numeri estratti
    from . import RUOTE_NOMI
    estratti_idx: dict[tuple[str, str], frozenset] = {}
    for _, row in df.iterrows():
        estratti_idx[(row["data"].date().isoformat(), RUOTE_NOMI[row["ruota"]])] = row["numeri"]

    for voce in voci:
        if voce.get("verificata"):
            continue
        data_iso = voce["data_estrazione"]
        esiti = {}
        trovata_almeno_una = False
        for ruota, previsti in voce["ruote"].items():
            estr = estratti_idx.get((data_iso, ruota))
            if estr is None:
                continue  # estrazione non ancora in archivio
            trovata_almeno_una = True
            azzeccati = sorted(n for n in previsti if n in estr)
            esiti[ruota] = {
                "previsti": sorted(previsti),
                "estratti": sorted(int(x) for x in estr),
                "azzeccati": azzeccati,
                "n_azzeccati": len(azzeccati),
            }
        if trovata_almeno_una:
            voce["esiti"] = esiti
            voce["verificata"] = True
    salva_previsioni(percorso_prev, voci)
    return voci


def ultima_verificata(percorso_prev: str) -> dict | None:
    voci = [v for v in carica_previsioni(percorso_prev) if v.get("verificata")]
    return voci[-1] if voci else None


def ultima_in_attesa(percorso_prev: str) -> dict | None:
    voci = [v for v in carica_previsioni(percorso_prev) if not v.get("verificata")]
    return voci[-1] if voci else None
