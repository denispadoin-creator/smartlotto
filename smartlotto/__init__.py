"""
SmartLotto — Analisi matematica, numerologica e olistica delle estrazioni
del Gioco del Lotto italiano.

ATTENZIONE / PREMESSA SCIENTIFICA
---------------------------------
Le estrazioni del Lotto sono, per costruzione fisica, eventi indipendenti:
il risultato di un'estrazione non dipende in alcun modo dalle precedenti.
Questo pacchetto NON promette di prevedere i numeri vincenti — nessun metodo
può farlo. Serve a fare l'opposto in modo onesto: misurare con rigore se una
qualsiasi strategia (ritardatari, frequenze, sequenze, numerologia, luna,
pianeti) batte davvero una giocata a caso, usando un backtesting
walk-forward (finestra espandente) che simula il tempo reale.

Il verdetto atteso, in base alla matematica del gioco, è "nessuna strategia
batte il caso in modo statisticamente significativo". Il valore del progetto
è dimostrarlo sui dati veri, non aggirarlo.

Moduli:
    ingestion   -> caricamento e strutturazione dell'archivio storico
    statistiche -> ritardi, frequenze, cicli
    sequenze    -> Fibonacci, progressioni geometriche/aritmetiche
    numerologia -> riduzione teosofica, numeri maestri, somma cabalistica
    astro       -> fasi lunari e posizioni planetarie (libreria ephem)
    strategie   -> ogni strategia come funzione pura (storico, data, ruota)->set
    backtest    -> motore walk-forward espandente + baseline casuale + metriche
    formule     -> motore euristico di ricerca formule con correzione multi-test
"""

__version__ = "1.0.0"

# Le 11 ruote del Lotto con il codice a due lettere usato nell'archivio.
RUOTE = ["BA", "CA", "FI", "GE", "MI", "NA", "PA", "RM", "RN", "TO", "VE"]

RUOTE_NOMI = {
    "BA": "Bari", "CA": "Cagliari", "FI": "Firenze", "GE": "Genova",
    "MI": "Milano", "NA": "Napoli", "PA": "Palermo", "RM": "Roma",
    "RN": "Nazionale", "TO": "Torino", "VE": "Venezia",
}
