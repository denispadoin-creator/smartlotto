"""
Modulo predittivo: genera una cinquina (5 numeri) per la prossima estrazione,
su una ruota o su tutte.

IMPORTANTE (onestà scientifica): questi 5 numeri hanno esattamente la stessa
probabilità di qualunque altra cinquina. Il modulo esiste perché l'app deve
"dare 5 numeri", ma il backtesting mostra che nessun criterio batte il caso.
Serve al gioco/curiosità, non come previsione affidabile.

Il criterio combina, in modo trasparente, i segnali richiesti dal progetto:
ritardo, (in)frequenza, numerologia della data e numeri del cielo. Ogni numero
1..90 riceve un punteggio; si scelgono i 5 col punteggio più alto.
"""

from __future__ import annotations
import datetime as _dt
import numpy as np

from . import ingestion, numerologia, astro, RUOTE, RUOTE_NOMI
from .strategie import StatoIncrementale


def _stato_finale(sub_ruota) -> StatoIncrementale:
    st = StatoIncrementale()
    for numeri in sub_ruota["numeri"]:
        st.aggiorna(numeri)
    return st


def punteggi(sub_ruota, data: _dt.date) -> np.ndarray:
    """Punteggio 1..90 per una ruota alla vigilia di `data`.

    Combina (normalizzati 0..1):
      + ritardo attuale        (i ritardatari pesano di più)
      - frequenza storica       (i numeri 'freddi' pesano di più)
      + bonus numerologia data  (se il numero è fra quelli del giorno)
      + bonus cielo del giorno  (se il numero deriva dai fenomeni celesti)
    I pesi sono espliciti e modificabili.
    """
    st = _stato_finale(sub_ruota)
    rit = st.ritardi().astype(float)
    freq = st.conteggio.astype(float)

    def norm(x):
        x = x[1:].copy()
        rng = x.max() - x.min()
        return (x - x.min()) / rng if rng > 0 else np.zeros_like(x)

    s_rit = norm(rit)              # 0..1, alto = molto ritardatario
    s_fred = 1 - norm(freq)       # 0..1, alto = poco frequente (freddo)

    bonus = np.zeros(90)
    for n in numerologia.numeri_da_data(data):
        bonus[n - 1] += 1.0
    for n in astro.numeri_da_cielo(data):
        bonus[n - 1] += 1.0
    bonus = bonus / 2.0           # 0..1

    # Pesi (trasparenti): ritardo 0.4, freddo 0.3, numerologia+cielo 0.3
    score = 0.4 * s_rit + 0.3 * s_fred + 0.3 * bonus
    out = np.zeros(91)
    out[1:] = score
    return out


def cinquina(sub_ruota, data: _dt.date, quanti: int = 5) -> list[int]:
    """I `quanti` numeri col punteggio più alto per la prossima estrazione."""
    sc = punteggi(sub_ruota, data)
    ordine = np.argsort(sc[1:])[::-1] + 1
    return sorted(int(x) for x in ordine[:quanti])


def prossima_data(df) -> _dt.date:
    """Stima la data della prossima estrazione (giorno dopo l'ultima presente)."""
    ultima = df["data"].max().date()
    return ultima + _dt.timedelta(days=1)


def previsione_tutte_le_ruote(df, data: _dt.date | None = None) -> dict:
    """Genera una cinquina per ogni ruota disponibile alla data indicata."""
    if data is None:
        data = prossima_data(df)
    out = {"data": data.isoformat(), "ruote": {}}
    for r in RUOTE:
        sub = ingestion.matrice_ruota(df, r)
        if len(sub) < 50:
            continue
        out["ruote"][RUOTE_NOMI[r]] = cinquina(sub, data)
    return out


def cinquina_da_regole(sub_ruota, data: _dt.date, chiavi: list[str] | None = None,
                       quanti: int = 5) -> list[int]:
    """Genera una cinquina applicando SOLO le regole selezionate (per chiave).

    Ogni regola selezionata propone dei numeri; si contano i "voti" ricevuti da
    ciascun numero e si scelgono i più votati. È il criterio trasparente usato
    quando l'utente sceglie quali regole attivare.
    """
    from . import regole as _reg
    from .strategie import StatoIncrementale
    strategie = _reg.costruisci_strategie(chiavi)
    stato = StatoIncrementale()
    for numeri in sub_ruota["numeri"]:
        stato.aggiorna(numeri)
    voti = np.zeros(91)
    for s in strategie:
        for n in s.proponi(stato, data):
            if 1 <= n <= 90:
                voti[n] += 1
    if voti.sum() == 0:
        return cinquina(sub_ruota, data, quanti)
    ordine = np.argsort(voti[1:])[::-1] + 1
    return sorted(int(x) for x in ordine[:quanti])
