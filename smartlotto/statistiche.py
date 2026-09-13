"""
Analisi matematica classica: frequenze, ritardi, cicli.

Tutte le funzioni sono "causali": calcolano lo stato usando SOLO le estrazioni
fino a un certo indice `fino_a` (escluso), così da poter essere usate nel
backtesting walk-forward senza guardare il futuro.
"""

from __future__ import annotations
import numpy as np


def frequenze(P: np.ndarray, fino_a: int | None = None) -> np.ndarray:
    """Conteggio di uscite per ogni numero 1..90 fino all'estrazione `fino_a`.

    Ritorna un array di 91 interi (indice 0 ignorato).
    """
    if fino_a is None:
        fino_a = P.shape[0]
    return P[:fino_a].sum(axis=0).astype(int)


def ritardi(P: np.ndarray, fino_a: int | None = None) -> np.ndarray:
    """Ritardo attuale di ogni numero: quante estrazioni sono passate dall'ultima
    volta che è uscito, calcolato allo stato appena PRIMA dell'indice `fino_a`.

    Se un numero non è mai uscito, il ritardo è pari al numero di estrazioni viste.
    Ritorna un array di 91 interi (indice 0 ignorato).
    """
    if fino_a is None:
        fino_a = P.shape[0]
    finestra = P[:fino_a]
    n = finestra.shape[0]
    rit = np.full(91, n, dtype=int)  # default: mai uscito -> ritardo = n
    if n == 0:
        return rit
    for k in range(1, 91):
        col = finestra[:, k]
        # indice dell'ultima uscita
        idx = np.nonzero(col)[0]
        if idx.size:
            rit[k] = n - 1 - idx[-1]
    return rit


def ritardo_massimo_storico(P: np.ndarray, fino_a: int | None = None) -> np.ndarray:
    """Per ogni numero, il ritardo massimo mai osservato fino a `fino_a`.
    Serve come riferimento per capire quanto un ritardo attuale sia "anomalo".
    """
    if fino_a is None:
        fino_a = P.shape[0]
    finestra = P[:fino_a]
    n = finestra.shape[0]
    out = np.zeros(91, dtype=int)
    for k in range(1, 91):
        idx = np.nonzero(finestra[:, k])[0]
        if idx.size == 0:
            out[k] = n
            continue
        gaps = np.diff(np.concatenate(([-1], idx)))  # distanze fra uscite
        coda = n - 1 - idx[-1]  # ritardo attuale finale
        out[k] = int(max(gaps.max() - 1 if gaps.size else 0, coda))
    return out


def indice_ciclicita(P: np.ndarray, numero: int, fino_a: int | None = None) -> dict:
    """Analisi del 'ciclo' di un numero: media e deviazione degli intervalli fra
    uscite consecutive. Un gioco davvero casuale ha intervalli distribuiti in modo
    geometrico (media teorica ~ 90/5 = 18 estrazioni per l'estratto su una ruota).
    """
    if fino_a is None:
        fino_a = P.shape[0]
    idx = np.nonzero(P[:fino_a, numero])[0]
    if idx.size < 2:
        return {"numero": numero, "uscite": int(idx.size), "intervallo_medio": None,
                "intervallo_std": None}
    gaps = np.diff(idx)
    return {
        "numero": numero,
        "uscite": int(idx.size),
        "intervallo_medio": float(gaps.mean()),
        "intervallo_std": float(gaps.std(ddof=1)) if gaps.size > 1 else 0.0,
    }


def classifica_ritardatari(P: np.ndarray, fino_a: int | None = None, top: int = 10) -> list[tuple[int, int]]:
    """I `top` numeri con ritardo attuale più alto (numero, ritardo)."""
    rit = ritardi(P, fino_a)
    ordine = sorted(range(1, 91), key=lambda k: rit[k], reverse=True)
    return [(k, int(rit[k])) for k in ordine[:top]]


def classifica_frequenti(P: np.ndarray, fino_a: int | None = None, top: int = 10, caldi: bool = True) -> list[tuple[int, int]]:
    """I `top` numeri più frequenti (caldi=True) o meno frequenti (caldi=False)."""
    fr = frequenze(P, fino_a)
    ordine = sorted(range(1, 91), key=lambda k: fr[k], reverse=caldi)
    return [(k, int(fr[k])) for k in ordine[:top]]
