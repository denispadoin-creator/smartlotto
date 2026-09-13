"""
Definizione delle strategie di gioco.

Ogni strategia, data una data di estrazione e lo stato storico "fino ad ora"
(numeri già usciti su una ruota), propone un insieme di numeri da giocare.
Le strategie sono funzioni pure e causali: NON vedono mai il futuro.

Lo `StatoIncrementale` viene aggiornato passo dopo passo dal motore di
backtesting, così i ritardi/frequenze/regole si formano anno dopo anno (finestra
espandente) senza ricalcolare tutto da capo ad ogni estrazione.
"""

from __future__ import annotations
import datetime as _dt
import numpy as np

from . import numerologia, sequenze, astro


class StatoIncrementale:
    """Stato storico di UNA ruota, aggiornato in modo incrementale.

    - conteggio[k]  = quante volte il numero k è uscito finora
    - ultima[k]     = indice dell'ultima estrazione in cui k è uscito (-1 = mai)
    - n             = numero di estrazioni osservate finora
    """

    def __init__(self):
        self.conteggio = np.zeros(91, dtype=np.int64)
        self.ultima = np.full(91, -1, dtype=np.int64)
        self.n = 0
        # Strutture aggiuntive per le regole avanzate:
        self.cooc = np.zeros((91, 91), dtype=np.int64)   # co-uscite di coppie
        self.trans = np.zeros((91, 91), dtype=np.int64)  # a(estr. precedente)->b(successiva)
        self.somma_gap = np.zeros(91, dtype=np.float64)  # somma degli intervalli fra uscite
        self.n_gap = np.zeros(91, dtype=np.int64)        # numero di intervalli osservati
        self.ultimo_draw = None                          # numeri dell'ultima estrazione

    def aggiorna(self, numeri) -> None:
        numeri = list(numeri)
        # intervalli fra uscite (per l'analisi dei cicli)
        for k in numeri:
            if self.ultima[k] >= 0:
                self.somma_gap[k] += (self.n - self.ultima[k])
                self.n_gap[k] += 1
        # co-occorrenze (coppie nella stessa estrazione)
        for i in range(len(numeri)):
            for j in range(i + 1, len(numeri)):
                a, b = numeri[i], numeri[j]
                self.cooc[a, b] += 1
                self.cooc[b, a] += 1
        # transizioni dall'estrazione precedente a questa
        if self.ultimo_draw is not None:
            for a in self.ultimo_draw:
                for b in numeri:
                    self.trans[a, b] += 1
        for k in numeri:
            self.conteggio[k] += 1
            self.ultima[k] = self.n
        self.ultimo_draw = numeri
        self.n += 1

    def ritardi(self) -> np.ndarray:
        """Ritardo attuale di ogni numero (1..90). Mai uscito -> ritardo = n."""
        rit = np.where(self.ultima >= 0, self.n - 1 - self.ultima, self.n)
        rit[0] = -1
        return rit

    def gap_medio(self) -> np.ndarray:
        """Intervallo medio storico fra due uscite, per numero (0 se sconosciuto)."""
        with np.errstate(divide="ignore", invalid="ignore"):
            g = np.where(self.n_gap > 0, self.somma_gap / np.maximum(self.n_gap, 1), 0.0)
        return g


# ---------------------------------------------------------------------------
# Ogni strategia è un oggetto con .nome e .proponi(stato, data) -> list[int].
# ---------------------------------------------------------------------------

class Strategia:
    nome = "base"

    def proponi(self, stato: StatoIncrementale, data: _dt.date) -> list[int]:
        raise NotImplementedError


class Ritardatario(Strategia):
    """Gioca l'ambata più ritardataria (il numero uscito più tempo fa)."""
    nome = "ritardatario"

    def __init__(self, quanti: int = 1):
        self.quanti = quanti

    def proponi(self, stato, data):
        rit = stato.ritardi()
        ordine = np.argsort(rit[1:])[::-1] + 1  # dal più ritardatario
        return [int(x) for x in ordine[:self.quanti]]


class Frequente(Strategia):
    """Gioca l'ambata più frequente (il numero 'caldo')."""
    nome = "frequente_caldo"

    def __init__(self, quanti: int = 1):
        self.quanti = quanti

    def proponi(self, stato, data):
        c = stato.conteggio.copy(); c[0] = -1
        ordine = np.argsort(c[1:])[::-1] + 1
        return [int(x) for x in ordine[:self.quanti]]


class Raro(Strategia):
    """Gioca l'ambata meno frequente (il numero 'freddo')."""
    nome = "raro_freddo"

    def __init__(self, quanti: int = 1):
        self.quanti = quanti

    def proponi(self, stato, data):
        c = stato.conteggio.copy().astype(float); c[0] = np.inf
        ordine = np.argsort(c[1:]) + 1
        return [int(x) for x in ordine[:self.quanti]]


class Fibonacci(Strategia):
    """Gioca i numeri di Fibonacci entro 1..90 (insieme fisso)."""
    nome = "fibonacci"

    def __init__(self):
        self._nums = sequenze.fibonacci_fino_a(90)

    def proponi(self, stato, data):
        return list(self._nums)


class Numerologica(Strategia):
    """Gioca i numeri derivati numerologicamente dalla data (deterministica)."""
    nome = "numerologia_data"

    def proponi(self, stato, data):
        return numerologia.numeri_da_data(data if isinstance(data, _dt.date) else data.date())


class Lunare(Strategia):
    """Gioca i numeri derivati dai fenomeni celesti della data (Luna/pianeti)."""
    nome = "astro_luna_pianeti"

    def __init__(self):
        self._cache: dict = {}

    def proponi(self, stato, data):
        d = data if isinstance(data, _dt.date) else data.date()
        if d not in self._cache:
            self._cache[d] = astro.numeri_da_cielo(d)
        return self._cache[d]


class NumeroFisso(Strategia):
    """Gioca sempre lo stesso numero (utile come controllo)."""
    def __init__(self, numero: int):
        self.numero = numero
        self.nome = f"fisso_{numero}"

    def proponi(self, stato, data):
        return [self.numero]


def strategie_standard() -> list[Strategia]:
    """Insieme di strategie da confrontare nel backtesting."""
    return [
        Ritardatario(1),
        Ritardatario(3),
        Frequente(1),
        Raro(1),
        Fibonacci(),
        Numerologica(),
        Lunare(),
        NumeroFisso(90),  # controllo: il celebre "90 ritardatario"
    ]
