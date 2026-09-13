"""
Modulo di analisi numerologica tradizionale e cabalistica.

Implementa i calcoli richiesti: riduzione teosofica, numeri maestri, somma
cabalistica sulla data, cicli pitagorici. Sono tutti deterministici a partire
dalla data (o dai numeri estratti), quindi generano insiemi di numeri da
sottoporre al backtesting come qualunque strategia.

Nota: la numerologia non ha fondamento fisico rispetto all'esito delle
estrazioni. Qui è trattata come una famiglia di regole deterministiche da
validare, non come una fonte di verità.
"""

from __future__ import annotations
import datetime as _dt


def riduzione_teosofica(n: int) -> int:
    """Riduzione teosofica: somma ricorsiva delle cifre finché resta una cifra
    singola (1..9). Lo zero resta zero.
    Esempio: 2026 -> 2+0+2+6 = 10 -> 1+0 = 1.
    """
    n = abs(int(n))
    while n > 9:
        n = sum(int(c) for c in str(n))
    return n


def riduzione_con_maestri(n: int) -> int:
    """Come la riduzione teosofica ma preserva i 'numeri maestri' 11, 22, 33
    (non li riduce ulteriormente), secondo la tradizione numerologica.
    """
    n = abs(int(n))
    while n > 9:
        if n in (11, 22, 33):
            return n
        n = sum(int(c) for c in str(n))
    return n


def somma_cabalistica_data(data: _dt.date) -> int:
    """Somma di tutte le cifre di giorno, mese e anno.
    Esempio: 11/09/2026 -> 1+1+0+9+2+0+2+6 = 21.
    """
    s = f"{data.day:02d}{data.month:02d}{data.year:04d}"
    return sum(int(c) for c in s)


def numeri_da_data(data: _dt.date) -> list[int]:
    """Genera un piccolo insieme di numeri 'numerologici' dalla data, mappati
    nell'intervallo giocabile 1..90. Combinazione di:
      - somma cabalistica delle cifre
      - riduzione teosofica di quella somma
      - giorno + mese
      - anno ridotto
    I valori sono riportati in 1..90 con modulo (mantenendo 90 al posto di 0).
    """
    def in_range(x: int) -> int:
        x = x % 90
        return 90 if x == 0 else x

    somma = somma_cabalistica_data(data)
    valori = {
        in_range(somma),
        in_range(riduzione_teosofica(somma)),
        in_range(data.day + data.month),
        in_range(riduzione_teosofica(data.year)),
        in_range(data.day * data.month),
    }
    return sorted(valori)


def ciclo_pitagorico(data: _dt.date) -> int:
    """'Numero del giorno' pitagorico: riduzione teosofica della somma
    cabalistica della data (1..9). Usato per raggruppare i giorni in 9 cicli.
    """
    return riduzione_teosofica(somma_cabalistica_data(data))


def vibrazione_numero(n: int) -> int:
    """Riduzione teosofica di un singolo numero estratto (1..9): la sua
    'vibrazione' numerologica.
    """
    return riduzione_teosofica(n)
