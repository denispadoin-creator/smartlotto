"""
Modulo olistico e planetario.

Usa la libreria `ephem` (autosufficiente, nessun download di effemeridi) per
calcolare, data una certa data:
  - la fase lunare (illuminazione 0..1) e l'età della Luna in giorni
  - il segno zodiacale del Sole
  - la posizione (longitudine eclittica e segno) dei pianeti principali

Da questi fenomeni si derivano numeri 1..90 deterministici, da testare nel
backtesting come qualunque altra strategia. Anche qui: nessun legame causale
noto con le estrazioni; si tratta di validare regole, non di assumerle vere.
"""

from __future__ import annotations
import datetime as _dt
import math
import ephem

SEGNI = ["Ariete", "Toro", "Gemelli", "Cancro", "Leone", "Vergine",
         "Bilancia", "Scorpione", "Sagittario", "Capricorno", "Acquario", "Pesci"]

PIANETI = {
    "Sole": ephem.Sun, "Luna": ephem.Moon, "Mercurio": ephem.Mercury,
    "Venere": ephem.Venus, "Marte": ephem.Mars, "Giove": ephem.Jupiter,
    "Saturno": ephem.Saturn,
}


def _mezzogiorno(data: _dt.date) -> str:
    """Le effemeridi vengono valutate a mezzogiorno UTC della data data."""
    return f"{data.year}/{data.month}/{data.day} 12:00:00"


def fase_lunare(data: _dt.date) -> dict:
    """Illuminazione della Luna (0..1), età in giorni dal novilunio e nome fase."""
    d = ephem.Date(_mezzogiorno(data))
    luna = ephem.Moon(d)
    illum = float(luna.phase) / 100.0  # ephem.phase è la % illuminata
    prec_nov = ephem.previous_new_moon(d)
    eta = float(d - prec_nov)  # giorni dal novilunio
    # Nome fase in base all'età (ciclo ~29.53 giorni)
    frazione = (eta % 29.53) / 29.53
    if frazione < 0.03 or frazione > 0.97:
        nome = "Novilunio"
    elif frazione < 0.22:
        nome = "Luna crescente"
    elif frazione < 0.28:
        nome = "Primo quarto"
    elif frazione < 0.47:
        nome = "Gibbosa crescente"
    elif frazione < 0.53:
        nome = "Plenilunio"
    elif frazione < 0.72:
        nome = "Gibbosa calante"
    elif frazione < 0.78:
        nome = "Ultimo quarto"
    else:
        nome = "Luna calante"
    return {"illuminazione": illum, "eta_giorni": eta, "fase": nome}


def _segno_da_longitudine(lon_rad: float) -> tuple[str, float]:
    """Converte una longitudine eclittica (radianti) in (segno, gradi_nel_segno)."""
    gradi = math.degrees(lon_rad) % 360.0
    idx = int(gradi // 30)
    return SEGNI[idx], gradi - idx * 30


def posizioni_pianeti(data: _dt.date) -> dict:
    """Longitudine eclittica (gradi 0..360) e segno per ogni pianeta principale."""
    d = ephem.Date(_mezzogiorno(data))
    out = {}
    for nome, cls in PIANETI.items():
        corpo = cls(d)
        ecl = ephem.Ecliptic(corpo)
        gradi = math.degrees(float(ecl.lon)) % 360.0
        segno, g_segno = _segno_da_longitudine(float(ecl.lon))
        out[nome] = {"longitudine": gradi, "segno": segno, "gradi_nel_segno": g_segno}
    return out


def segno_solare(data: _dt.date) -> str:
    """Segno zodiacale del Sole (segno 'del giorno')."""
    return posizioni_pianeti(data)["Sole"]["segno"]


def numeri_da_cielo(data: _dt.date) -> list[int]:
    """Deriva un insieme di numeri 1..90 dai fenomeni celesti della data:
      - età della Luna (arrotondata) -> giorno lunare 1..30
      - longitudine del Sole in gradi -> 1..90 via modulo
      - somma delle longitudini di Luna e Marte -> 1..90
    Deterministico; usato come strategia nel backtesting.
    """
    def in_range(x) -> int:
        x = int(round(x)) % 90
        return 90 if x == 0 else x

    fase = fase_lunare(data)
    pos = posizioni_pianeti(data)
    valori = {
        in_range(fase["eta_giorni"]),
        in_range(pos["Sole"]["longitudine"]),
        in_range(pos["Luna"]["longitudine"] + pos["Marte"]["longitudine"]),
        in_range(pos["Giove"]["longitudine"]),
    }
    return sorted(valori)
