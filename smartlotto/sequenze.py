"""
Sequenze matematiche mappate sull'intervallo 1..90.

Fibonacci, progressioni geometriche e aritmetiche. Queste "sequenze" non hanno
alcun legame causale con le estrazioni: il modulo le fornisce come insiemi di
numeri da testare nel backtesting come qualunque altra strategia.
"""

from __future__ import annotations


def fibonacci_fino_a(limite: int = 90) -> list[int]:
    """Numeri di Fibonacci compresi in 1..limite (senza ripetizioni)."""
    seq, a, b = [], 1, 2
    seq.append(1)
    while b <= limite:
        seq.append(b)
        a, b = b, a + b
    return sorted(set(x for x in seq if 1 <= x <= limite))


def progressione_geometrica(base: int, ragione: int, limite: int = 90) -> list[int]:
    """Progressione geometrica base, base*r, base*r^2 ... entro 1..limite."""
    out, v = [], base
    while v <= limite:
        if v >= 1:
            out.append(v)
        v *= ragione
        if ragione <= 1:
            break
    return sorted(set(out))


def progressione_aritmetica(inizio: int, passo: int, limite: int = 90) -> list[int]:
    """Progressione aritmetica inizio, inizio+passo, ... entro 1..limite."""
    out, v = [], inizio
    while v <= limite:
        if v >= 1:
            out.append(v)
        v += passo
        if passo <= 0:
            break
    return sorted(set(out))


def quadrati(limite: int = 90) -> list[int]:
    """Quadrati perfetti in 1..limite."""
    out, k = [], 1
    while k * k <= limite:
        out.append(k * k)
        k += 1
    return out


def triangolari(limite: int = 90) -> list[int]:
    """Numeri triangolari (1,3,6,10,...) in 1..limite."""
    out, k, t = [], 1, 1
    while t <= limite:
        out.append(t)
        k += 1
        t += k
    return out


def numeri_primi(limite: int = 90) -> list[int]:
    """Numeri primi in 1..limite (crivello semplice)."""
    if limite < 2:
        return []
    setaccio = [True] * (limite + 1)
    setaccio[0] = setaccio[1] = False
    for i in range(2, int(limite ** 0.5) + 1):
        if setaccio[i]:
            for j in range(i * i, limite + 1, i):
                setaccio[j] = False
    return [i for i in range(2, limite + 1) if setaccio[i]]
