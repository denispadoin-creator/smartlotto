"""
Aggiornamento automatico dell'archivio: scarica le estrazioni mancanti e le
aggiunge in fondo a `archivio_lotto.txt`.

Sorgente: estrazionilottooggi.it (report per intervallo di date). Il download
gira sulla macchina dove parte l'app (che ha accesso a internet); richiede solo
la libreria standard `urllib` (nessuna dipendenza extra).

Uso da codice:
    from smartlotto import aggiornamento
    n = aggiornamento.aggiorna_archivio("data/archivio_lotto.txt")
    print(f"{n} nuove estrazioni aggiunte")

Uso da riga di comando:
    python -m smartlotto.aggiornamento data/archivio_lotto.txt
"""

from __future__ import annotations
import datetime as _dt
import re
import sys
import urllib.request
import urllib.parse

BASE = "http://www.estrazionilottooggi.it/report-estrazioni"

# Nome ruota sul sito -> codice a due lettere usato nell'archivio.
CODICE_RUOTA = {
    "Bari": "BA", "Cagliari": "CA", "Firenze": "FI", "Genova": "GE",
    "Milano": "MI", "Napoli": "NA", "Palermo": "PA", "Roma": "RM",
    "Nazionale": "RN", "Torino": "TO", "Venezia": "VE",
}

# Una riga della tabella: data, ruota, e i 5 numeri.
_RE_RIGA = re.compile(
    r'<td>(\d{2}/\d{2}/\d{4})</td>\s*'
    r'<td class="std">\d+</td>\s*'
    r'<td class="firsttd">([^<]+)</td>'
    r'((?:\s*<td class="num">\d+</td>){5})',
    re.S,
)
_RE_NUM = re.compile(r'<td class="num">(\d+)</td>')


def ultima_data_archivio(percorso: str) -> _dt.date | None:
    """Ultima data presente nell'archivio (None se il file è vuoto/assente)."""
    ultima = None
    try:
        with open(percorso, "r", encoding="utf-8") as fh:
            for riga in fh:
                riga = riga.strip()
                if not riga:
                    continue
                d = riga.split(" ", 1)[0]
                data = _dt.date(int(d[0:4]), int(d[4:6]), int(d[6:8]))
                if ultima is None or data > ultima:
                    ultima = data
    except FileNotFoundError:
        return None
    return ultima


def _scarica_html(dal: _dt.date, al: _dt.date, timeout: int = 30) -> str:
    """Scarica la pagina report per l'intervallo [dal, al]."""
    q = urllib.parse.urlencode({
        "dataDa": dal.strftime("%d/%m/%Y"),
        "dataA": al.strftime("%d/%m/%Y"),
        "ReportButton": "Ricerca",
    })
    req = urllib.request.Request(f"{BASE}?{q}", headers={"User-Agent": "Mozilla/5.0 SmartLotto"})
    with urllib.request.urlopen(req, timeout=timeout) as resp:
        return resp.read().decode("utf-8", errors="replace")


def analizza_html(html: str) -> dict[_dt.date, dict[str, list[int]]]:
    """Estrae dall'HTML un dizionario {data: {codice_ruota: [5 numeri]}}."""
    per_data: dict[_dt.date, dict[str, list[int]]] = {}
    for m in _RE_RIGA.finditer(html):
        gg, mm, aaaa = m.group(1).split("/")
        data = _dt.date(int(aaaa), int(mm), int(gg))
        nome_ruota = m.group(2).strip()
        codice = CODICE_RUOTA.get(nome_ruota)
        if not codice:
            continue
        nums = [int(x) for x in _RE_NUM.findall(m.group(3))]
        if len(nums) != 5 or any(not (1 <= n <= 90) for n in nums):
            continue
        per_data.setdefault(data, {})[codice] = nums
    return per_data


def _riga_archivio(data: _dt.date, ruote: dict[str, list[int]]) -> str:
    """Formatta una data nel formato dell'archivio: AAAAMMGG RUOTA:n.n.n.n.n ..."""
    key = f"{data.year:04d}{data.month:02d}{data.day:02d}"
    blocchi = []
    for codice in sorted(ruote):
        nums = ".".join(f"{n:02d}" for n in ruote[codice])
        blocchi.append(f"{codice}:{nums}")
    return key + " " + " ".join(blocchi)


def scarica_nuove(ultima: _dt.date | None, fino_a: _dt.date | None = None,
                  timeout: int = 15) -> list[str]:
    """Scarica le estrazioni successive a `ultima` e le ritorna come righe di
    archivio (senza scrivere su file). Usata online: le righe verranno salvate
    nel browser dell'utente così sopravvivono ai riavvii di Streamlit.
    """
    if fino_a is None:
        fino_a = _dt.date.today()
    dal = (ultima + _dt.timedelta(days=1)) if ultima else _dt.date(1871, 1, 1)
    if dal > fino_a:
        return []
    html = _scarica_html(dal, fino_a, timeout=timeout)
    per_data = analizza_html(html)
    nuove = sorted(d for d in per_data if (ultima is None or d > ultima))
    return [_riga_archivio(d, per_data[d]) for d in nuove]


def aggiorna_archivio(percorso: str, fino_a: _dt.date | None = None) -> int:
    """Scarica le estrazioni successive all'ultima presente e le aggiunge.

    Ritorna il numero di NUOVE date di estrazione aggiunte. Non modifica nulla
    se non c'è niente di nuovo. Ogni riga aggiunta è nel formato dell'archivio.
    """
    if fino_a is None:
        fino_a = _dt.date.today()
    ultima = ultima_data_archivio(percorso)
    dal = (ultima + _dt.timedelta(days=1)) if ultima else _dt.date(1871, 1, 1)
    if dal > fino_a:
        return 0

    html = _scarica_html(dal, fino_a)
    per_data = analizza_html(html)
    # Tiene solo le date davvero nuove, in ordine cronologico.
    nuove = sorted(d for d in per_data if (ultima is None or d > ultima))
    if not nuove:
        return 0

    righe = [_riga_archivio(d, per_data[d]) for d in nuove]
    with open(percorso, "a", encoding="utf-8") as fh:
        for r in righe:
            fh.write(r + "\n")
    return len(nuove)


if __name__ == "__main__":
    percorso = sys.argv[1] if len(sys.argv) > 1 else "data/archivio_lotto.txt"
    n = aggiorna_archivio(percorso)
    print(f"{n} nuove estrazioni aggiunte a {percorso}"
          if n else "Archivio già aggiornato: nessuna nuova estrazione.")
