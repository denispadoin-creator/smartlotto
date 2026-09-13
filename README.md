# SmartLotto — Analisi predittiva delle estrazioni del Lotto

Applicazione Python per l'analisi matematica, sequenziale, numerologica e
olistica (Luna e pianeti) delle estrazioni del Gioco del Lotto italiano, con un
sistema di **backtesting walk-forward a finestra espandente** che confronta ogni
strategia con la giocata casuale.

## Premessa onesta (importante)

Le estrazioni del Lotto sono eventi indipendenti: ogni numero ha probabilità
**5/90 = 5,556%** di uscire come estratto su una ruota, ad **ogni** estrazione,
indipendentemente dal passato. Questo software **non prevede** i numeri vincenti
— nessun metodo può. Serve a **verificare in modo rigoroso** se ritardi,
frequenze, sequenze, numerologia o fenomeni celesti battono davvero il caso.

Sui dati reali (1871–2026, ~10.900 estrazioni) il risultato è quello atteso
dalla matematica: **nessuna strategia batte il caso** in modo statisticamente
solido, e le "formule" che sembrano funzionare nel passato **non reggono** nel
futuro (overfitting). Il valore del progetto è dimostrarlo con numeri veri.

## Installazione

```bash
pip install -r requirements.txt
```

Dipendenze: `pandas`, `numpy`, `scipy`, `ephem` (astronomia, senza download di
effemeridi), `streamlit` (interfaccia).

## Dati

L'archivio storico è in `data/archivio_lotto.txt`, una riga per data:

```
AAAAMMGG RUOTA:n.n.n.n.n RUOTA:n.n.n.n.n ...
20260911 BA:53.20.40.62.16 CA:33.22.88.17.06 ... VE:26.87.14.71.07
```

Ruote: BA Bari, CA Cagliari, FI Firenze, GE Genova, MI Milano, NA Napoli,
PA Palermo, RM Roma, RN Nazionale, TO Torino, VE Venezia. Le ruote disponibili
cambiano nel tempo (7 nel 1871, 10 dal 1939-40, Nazionale dal 2005).

## Esecuzione

Analisi completa da riga di comando (produce `report_backtest.json/.txt`):

```bash
python run_backtest.py data/archivio_lotto.txt --warmup 300 --taglio 2010-01-01
```

Interfaccia interattiva:

```bash
streamlit run app.py
```

## Previsione, aggiornamento e verifica

- **Previsione**: la scheda "Previsione" (o `predittore.py`) genera 5 numeri per
  ogni ruota per la prossima estrazione e li salva in `previsioni.json`.
- **Aggiornamento**: il tasto "Scarica estrazioni mancanti" (o
  `python -m smartlotto.aggiornamento`) aggiunge in archivio le estrazioni nuove.
- **Verifica**: dopo l'estrazione, all'aggiornamento i numeri realmente usciti
  che erano stati previsti vengono **evidenziati in verde**.

Ciclo completo con un solo comando (scarica + verifica + nuova previsione):

```bash
python aggiorna_e_prevedi.py data/archivio_lotto.txt previsioni.json
```

**Ogni settimana in automatico (Windows)**: programma `aggiorna_settimanale.bat`
in Utilità di pianificazione (Task Scheduler) con cadenza settimanale. Il file
`log_settimanale.txt` registra ogni esecuzione.

## Struttura

```
smartlotto/
  ingestion.py    caricamento e strutturazione dell'archivio
  statistiche.py  ritardi, frequenze, cicli (funzioni causali)
  sequenze.py     Fibonacci, geometriche, primi, quadrati, triangolari
  numerologia.py  riduzione teosofica, numeri maestri, somma cabalistica, cicli
  astro.py        fasi lunari e posizioni planetarie (ephem)
  strategie.py    ogni strategia come funzione pura (stato, data) -> numeri
  backtest.py     motore walk-forward espandente + test binomiale + baseline
  formule.py      motore euristico + validazione out-of-sample (anti-overfitting)
run_backtest.py   esecuzione completa da riga di comando
app.py            interfaccia Streamlit
```

## Metodo di backtesting

Finestra **espandente**: si parte dalle prime estrazioni e si avanza nel tempo.
Ad ogni estrazione *i*, ogni strategia propone i numeri usando **solo** le
estrazioni 0..*i*-1; poi si verifica l'esito contro l'estrazione *i*. Le regole
(ritardi, frequenze, ecc.) si formano e si aggiornano anno dopo anno. Nessuna
informazione futura entra mai nella previsione.

La metrica è il **tasso di successo per numero giocato**, confrontato con la
soglia teorica 5/90 tramite un test **binomiale esatto**. Il motore di formule
applica inoltre la **correzione per confronti multipli** (Bonferroni) e valida
le regole su un periodo futuro mai visto.
