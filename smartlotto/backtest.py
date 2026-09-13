"""
Sistema di backtesting rigoroso — finestra ESPANDENTE (walk-forward).

Principio (richiesto esplicitamente): si parte dalle prime estrazioni e si
avanza nel tempo. Ad ogni estrazione i, ogni strategia propone i numeri usando
SOLO le estrazioni precedenti (0..i-1); poi si controlla l'esito contro
l'estrazione i. Le regole (ritardi, frequenze, ...) si formano e si aggiornano
anno dopo anno. Nessuna informazione dal futuro entra mai nella previsione.

Metrica di riferimento: il tasso di successo "per numero giocato". Su una ruota,
un numero qualsiasi ha probabilità teorica 5/90 = 5.556% di essere fra i 5
estratti. Una strategia ha valore SOLO se batte questo 5.556% in modo
statisticamente significativo. Il test è binomiale esatto: sotto l'ipotesi nulla
(estrazioni indipendenti e uniformi) il numero di successi segue una
Binomiale(numero_giocate, 5/90).

Viene inoltre calcolato un baseline empirico "giocata a caso" via Monte Carlo,
che rispetta l'estrazione senza reimmissione, come termine di paragone concreto.
"""

from __future__ import annotations
import numpy as np
from dataclasses import dataclass, field
from scipy import stats

from .strategie import StatoIncrementale, Strategia
from . import RUOTE

P_ESTRATTO = 5.0 / 90.0  # probabilità teorica che un numero fissato sia estratto


@dataclass
class RisultatoStrategia:
    nome: str
    giocate: int = 0        # numeri totali giocati (trials)
    successi: int = 0       # numeri giocati che sono usciti (hits)
    estrazioni: int = 0     # numero di estrazioni testate
    estrazioni_vincenti: int = 0  # estrazioni con almeno un successo (ambata sortita)

    @property
    def tasso_per_numero(self) -> float:
        return self.successi / self.giocate if self.giocate else 0.0

    @property
    def tasso_vincenti(self) -> float:
        return self.estrazioni_vincenti / self.estrazioni if self.estrazioni else 0.0

    def statistica(self) -> dict:
        """Confronto col caso: attesi, differenza, z-score e p-value binomiale
        (unilaterale, 'la strategia fa MEGLIO del caso')."""
        T, H = self.giocate, self.successi
        attesi = T * P_ESTRATTO
        if T == 0:
            return {"attesi": 0, "z": 0.0, "p_value": 1.0}
        sd = (T * P_ESTRATTO * (1 - P_ESTRATTO)) ** 0.5
        z = (H - attesi) / sd if sd > 0 else 0.0
        # p-value esatto: P(Binom(T,p) >= H)
        p_value = float(stats.binom.sf(H - 1, T, P_ESTRATTO))
        return {"attesi": attesi, "z": float(z), "p_value": p_value}


def backtest_ruota(sub_ruota, strategie: list[Strategia], warmup: int = 200) -> dict[str, RisultatoStrategia]:
    """Esegue il walk-forward espandente su una singola ruota.

    sub_ruota: DataFrame ordinato per data della ruota (colonne data, numeri).
    Ritorna {nome_strategia: RisultatoStrategia}.
    """
    stato = StatoIncrementale()
    risultati = {s.nome: RisultatoStrategia(nome=s.nome) for s in strategie}

    date = list(sub_ruota["data"])
    insiemi = list(sub_ruota["numeri"])

    for i in range(len(insiemi)):
        estratti = insiemi[i]
        if i >= warmup:
            data_i = date[i].date()
            for s in strategie:
                picks = s.proponi(stato, data_i)
                if not picks:
                    continue
                r = risultati[s.nome]
                colpiti = sum(1 for x in picks if x in estratti)
                r.giocate += len(picks)
                r.successi += colpiti
                r.estrazioni += 1
                if colpiti > 0:
                    r.estrazioni_vincenti += 1
        stato.aggiorna(estratti)

    return risultati


def baseline_casuale(sub_ruota, dimensioni_per_estrazione: list[int], warmup: int,
                     ripetizioni: int = 200, seed: int = 12345) -> dict:
    """Baseline Monte Carlo: per ogni estrazione testata si giocano numeri
    CASUALI in pari quantità a quelli giocati dalla strategia, e si contano i
    successi. Ripetuto `ripetizioni` volte per stimare media e deviazione.

    `dimensioni_per_estrazione`: numero di numeri giocati ad ogni estrazione
    testata (stessa lunghezza delle estrazioni testate).
    """
    rng = np.random.default_rng(seed)
    insiemi = list(sub_ruota["numeri"])[warmup:]
    assert len(insiemi) == len(dimensioni_per_estrazione)
    tot_giocate = sum(dimensioni_per_estrazione)
    successi_rip = np.zeros(ripetizioni)
    for r in range(ripetizioni):
        s = 0
        for estratti, k in zip(insiemi, dimensioni_per_estrazione):
            if k <= 0:
                continue
            scelti = rng.choice(np.arange(1, 91), size=k, replace=False)
            s += int(sum(1 for x in scelti if x in estratti))
        successi_rip[r] = s
    return {
        "giocate_totali": tot_giocate,
        "successi_medi": float(successi_rip.mean()),
        "successi_std": float(successi_rip.std(ddof=1)) if ripetizioni > 1 else 0.0,
        "tasso_medio": float(successi_rip.mean() / tot_giocate) if tot_giocate else 0.0,
    }


def aggrega(risultati_per_ruota: list[dict[str, RisultatoStrategia]]) -> dict[str, RisultatoStrategia]:
    """Somma i risultati di una strategia su più ruote in un unico risultato."""
    nomi = risultati_per_ruota[0].keys()
    agg = {n: RisultatoStrategia(nome=n) for n in nomi}
    for per_ruota in risultati_per_ruota:
        for n, r in per_ruota.items():
            a = agg[n]
            a.giocate += r.giocate
            a.successi += r.successi
            a.estrazioni += r.estrazioni
            a.estrazioni_vincenti += r.estrazioni_vincenti
    return agg
