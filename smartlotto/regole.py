"""
Registro completo delle regole di gioco, ognuna come strategia testabile.

Comprende sia le regole già validate sia quelle nuove del catalogo. Ogni regola
è una `Strategia` (funzione pura e causale: usa solo il passato) e appartiene a
una categoria. Il registro `REGISTRO` permette all'app e al backtesting di
scegliere QUALI regole applicare.

Alcune voci sono "utilità" (richiedono un input, es. la smorfia o il nome) o
"analisi" (producono statistiche, non una giocata): sono elencate a parte e non
entrano nel backtesting come strategia.
"""

from __future__ import annotations
import datetime as _dt
import numpy as np

from .strategie import Strategia, StatoIncrementale
from . import sequenze, numerologia, astro


def _range(x: int) -> int:
    x = int(x) % 90
    return 90 if x == 0 else x


def _to_date(d):
    return d if isinstance(d, _dt.date) else d.date()


# ============================ MATEMATICHE / STATISTICHE ============================

class Ritardatario(Strategia):
    nome = "ritardatario"
    def __init__(self, quanti=1): self.quanti = quanti
    def proponi(self, s, data):
        ordine = np.argsort(s.ritardi()[1:])[::-1] + 1
        return [int(x) for x in ordine[:self.quanti]]


class Frequente(Strategia):
    nome = "frequente_caldo"
    def __init__(self, quanti=1): self.quanti = quanti
    def proponi(self, s, data):
        c = s.conteggio.copy(); c[0] = -1
        return [int(x) for x in (np.argsort(c[1:])[::-1] + 1)[:self.quanti]]


class Raro(Strategia):
    nome = "raro_freddo"
    def __init__(self, quanti=1): self.quanti = quanti
    def proponi(self, s, data):
        c = s.conteggio.astype(float).copy(); c[0] = np.inf
        return [int(x) for x in (np.argsort(c[1:]) + 1)[:self.quanti]]


class CicloRitardo(Strategia):
    """Gioca il numero il cui ritardo attuale supera di più il suo intervallo medio."""
    nome = "ciclo_intervalli"
    def proponi(self, s, data):
        if s.n < 50: return []
        rit = s.ritardi().astype(float); gm = s.gap_medio()
        score = np.where(gm[1:] > 0, rit[1:] - gm[1:], -1e9)
        return [int(np.argmax(score) + 1)]


class InsiemeFisso(Strategia):
    def __init__(self, nome, numeri): self.nome = nome; self._n = sorted(set(numeri))
    def proponi(self, s, data): return list(self._n)


class TerminaleRitardataria(Strategia):
    """Gioca il numero più ritardatario fra quelli con la terminale (cifra finale)
    complessivamente più in ritardo."""
    nome = "terminale_cadenza"
    def proponi(self, s, data):
        rit = s.ritardi()
        # ritardo minimo per terminale = quanto è "assente" quella cifra finale
        best_term, best_val = 0, -1
        for t in range(10):
            nums = [k for k in range(1, 91) if k % 10 == t]
            v = min(rit[k] for k in nums)
            if v > best_val: best_val, best_term = v, t
        nums = [k for k in range(1, 91) if k % 10 == best_term]
        return [max(nums, key=lambda k: rit[k])]


class DecinaFredda(Strategia):
    """Gioca il numero più ritardatario della decina meno uscita di recente."""
    nome = "fasce_decine"
    def proponi(self, s, data):
        rit = s.ritardi()
        decine = [list(range(base, base + 10)) for base in range(1, 91, 10)]  # 1-10,...,81-90
        best = max(decine, key=lambda dr: min(rit[k] for k in dr))
        return [max(best, key=lambda k: rit[k])]


class ParitaRitardo(Strategia):
    """Gioca il pari (o dispari) più ritardatario, a seconda di quale gruppo è più assente."""
    nome = "pari_dispari"
    def proponi(self, s, data):
        rit = s.ritardi()
        pari = [k for k in range(1, 91) if k % 2 == 0]
        disp = [k for k in range(1, 91) if k % 2 == 1]
        gruppo = pari if min(rit[k] for k in pari) >= min(rit[k] for k in disp) else disp
        return [max(gruppo, key=lambda k: rit[k])]


class AltiBassi(Strategia):
    """Gioca il numero più ritardatario della metà (bassa 1-45 / alta 46-90) più assente."""
    nome = "alti_bassi"
    def proponi(self, s, data):
        rit = s.ritardi()
        bassi = list(range(1, 46)); alti = list(range(46, 91))
        gruppo = bassi if min(rit[k] for k in bassi) >= min(rit[k] for k in alti) else alti
        return [max(gruppo, key=lambda k: rit[k])]


class SommaTendenza(Strategia):
    """Gioca il numero che riavvicina la somma dell'ultima estrazione alla media storica (~227)."""
    nome = "somma_estratti"
    def proponi(self, s, data):
        if not s.ultimo_draw: return []
        somma = sum(s.ultimo_draw)
        return [_range(abs(227 - somma))]


class Gemelli(Strategia):
    nome = "consecutivi_gemelli"
    def proponi(self, s, data): return [11, 22, 33, 44, 55, 66, 77, 88]


class NumeroSpia(Strategia):
    """Numero 'spia': dal numero più frequente dell'ultima estrazione, gioca il suo
    successore storicamente più probabile (matrice delle transizioni)."""
    nome = "numeri_spia"
    def proponi(self, s, data):
        if not s.ultimo_draw or s.n < 50: return []
        spia = max(s.ultimo_draw, key=lambda k: s.conteggio[k])
        succ = s.trans[spia].copy(); succ[0] = 0
        if succ.sum() == 0: return []
        return [int(np.argmax(succ[1:]) + 1)]


class Markov(Strategia):
    """Modello di Markov: dai numeri dell'ultima estrazione, gioca il numero con la
    più alta probabilità di transizione aggregata."""
    nome = "markov"
    def proponi(self, s, data):
        if not s.ultimo_draw or s.n < 50: return []
        agg = s.trans[list(s.ultimo_draw)].sum(axis=0); agg[0] = 0
        if agg.sum() == 0: return []
        return [int(np.argmax(agg[1:]) + 1)]


class CoppiaFrequente(Strategia):
    """Gioca i due numeri della coppia (ambo) storicamente più uscita insieme."""
    nome = "ambi_coppie"
    def proponi(self, s, data):
        if s.n < 50: return []
        idx = np.argmax(s.cooc)
        a, b = idx // 91, idx % 91
        return sorted({int(a), int(b)} - {0}) or []


class Vertibile(Strategia):
    """Gioca i 'vertibili' dei numeri dell'ultima estrazione (12->21, 4->40)."""
    nome = "vertibili"
    def proponi(self, s, data):
        if not s.ultimo_draw: return []
        out = set()
        for n in s.ultimo_draw:
            if n < 10:
                v = n * 10
            else:
                v = (n % 10) * 10 + (n // 10)
            if 1 <= v <= 90 and v != n: out.add(v)
        return sorted(out)


class Simmetrico(Strategia):
    """Gioca i 'simmetrici' (91 - n) dei numeri dell'ultima estrazione."""
    nome = "simmetrici"
    def proponi(self, s, data):
        if not s.ultimo_draw: return []
        return sorted({91 - n for n in s.ultimo_draw if 1 <= 91 - n <= 90})


# ============================ NUMEROLOGICHE ============================

class NumUnoData(Strategia):
    def __init__(self, nome, fn): self.nome = nome; self.fn = fn
    def proponi(self, s, data): return [_range(self.fn(_to_date(data)))]


class VibrazioneGiorno(Strategia):
    """Gioca i numeri la cui radice numerica (1-9) coincide con il ciclo del giorno."""
    nome = "vibrazione_numero"
    def proponi(self, s, data):
        c = numerologia.ciclo_pitagorico(_to_date(data))
        return [k for k in range(1, 91) if numerologia.riduzione_teosofica(k) == c]


class SommaCabalisticaEstratti(Strategia):
    """Gioca il numero dato dalla somma cabalistica dei numeri dell'ultima estrazione."""
    nome = "somma_cabalistica_estratti"
    def proponi(self, s, data):
        if not s.ultimo_draw: return []
        cifre = sum(int(c) for n in s.ultimo_draw for c in str(n))
        return [_range(cifre)]


class RadiceNumericaEstratti(Strategia):
    """Gioca i numeri con radice numerica pari a quella più frequente nell'ultima estrazione."""
    nome = "radice_numerica_estratti"
    def proponi(self, s, data):
        if not s.ultimo_draw: return []
        radici = [numerologia.riduzione_teosofica(n) for n in s.ultimo_draw]
        top = max(set(radici), key=radici.count)
        return [k for k in range(1, 91) if numerologia.riduzione_teosofica(k) == top]


# ============================ ASTROLOGICHE / OLISTICHE ============================

class AstroDaData(Strategia):
    """Strategia astronomica basata su una funzione data->numeri, con cache per data."""
    def __init__(self, nome, fn):
        self.nome = nome; self.fn = fn; self._cache = {}
    def proponi(self, s, data):
        d = _to_date(data)
        if d not in self._cache: self._cache[d] = self.fn(d)
        return self._cache[d]


def _num_fase_lunare(d):
    return [_range(round(astro.fase_lunare(d)["eta_giorni"]))]

def _num_giorno_lunare(d):
    return [_range(int(astro.fase_lunare(d)["eta_giorni"]) + 1)]

def _num_dal_cielo(d):
    return astro.numeri_da_cielo(d)

# Corrispondenze tradizionali segno zodiacale -> numeri (astrologia + smorfia, indicative)
SEGNO_NUMERI = {
    "Ariete": [1, 9, 19], "Toro": [2, 6, 24], "Gemelli": [5, 14, 23], "Cancro": [2, 7, 20],
    "Leone": [1, 8, 19], "Vergine": [5, 6, 23], "Bilancia": [6, 15, 24], "Scorpione": [9, 18, 27],
    "Sagittario": [3, 12, 21], "Capricorno": [8, 17, 26], "Acquario": [4, 13, 22], "Pesci": [7, 16, 25],
}
def _num_segno(d):
    return SEGNO_NUMERI.get(astro.segno_solare(d), [])

# Pianeta che governa il giorno della settimana -> numero (0=Lun..6=Dom)
PIANETA_GIORNO = {0: 2, 1: 3, 2: 4, 3: 5, 4: 6, 5: 7, 6: 1}  # Luna,Marte,Merc,Giove,Venere,Sat,Sole
def _num_pianeta_giorno(d):
    return [PIANETA_GIORNO[d.weekday()]]

def _num_stagione(d):
    # numero indicativo per stagione: primavera/estate/autunno/inverno
    m = d.month
    stag = 1 if m in (3, 4, 5) else 2 if m in (6, 7, 8) else 3 if m in (9, 10, 11) else 4
    return [_range(stag * 22)]

def _num_transiti(d):
    # minima separazione angolare fra due pianeti -> numero (aspetto stretto)
    pos = astro.posizioni_pianeti(d)
    lon = [v["longitudine"] for v in pos.values()]
    best = 999
    for i in range(len(lon)):
        for j in range(i + 1, len(lon)):
            diff = abs(lon[i] - lon[j]) % 360
            diff = min(diff, 360 - diff)
            best = min(best, diff)
    return [_range(round(best))]

def _num_eclissi(d):
    # distanza in giorni dalla prossima/precedente eclissi solare -> numero
    import ephem
    dd = ephem.Date(f"{d.year}/{d.month}/{d.day} 12:00:00")
    # usa i noviluni/pleniluni come proxy dei nodi (approssimazione olistica)
    prossimo = ephem.next_new_moon(dd)
    giorni = float(prossimo - dd)
    return [_range(round(giorni * 3))]


# ============================ REGISTRO ============================
# tipo: "giocata" = testabile nel backtest; "utilita"/"analisi" = solo funzione.

REGISTRO = [
    # --- Matematiche e statistiche ---
    ("ritardatario", "Ritardatari", "Matematiche", "giocata", lambda: Ritardatario(1)),
    ("frequente_caldo", "Frequenze (caldi)", "Matematiche", "giocata", lambda: Frequente(1)),
    ("raro_freddo", "Frequenze (freddi)", "Matematiche", "giocata", lambda: Raro(1)),
    ("fibonacci", "Fibonacci", "Matematiche", "giocata", lambda: InsiemeFisso("fibonacci", sequenze.fibonacci_fino_a(90))),
    ("ciclo_intervalli", "Cicli / intervalli", "Matematiche", "giocata", lambda: CicloRitardo()),
    ("progr_geometrica", "Progressione geometrica", "Matematiche", "giocata", lambda: InsiemeFisso("progr_geometrica", sequenze.progressione_geometrica(1, 2, 90))),
    ("progr_aritmetica", "Progressione aritmetica", "Matematiche", "giocata", lambda: InsiemeFisso("progr_aritmetica", sequenze.progressione_aritmetica(9, 9, 90))),
    ("quadrati", "Quadrati perfetti", "Matematiche", "giocata", lambda: InsiemeFisso("quadrati", sequenze.quadrati(90))),
    ("triangolari", "Triangolari", "Matematiche", "giocata", lambda: InsiemeFisso("triangolari", sequenze.triangolari(90))),
    ("primi", "Numeri primi", "Matematiche", "giocata", lambda: InsiemeFisso("primi", sequenze.numeri_primi(90))),
    ("fasce_decine", "Analisi per fasce/decine", "Matematiche", "giocata", lambda: DecinaFredda()),
    ("pari_dispari", "Pari / dispari", "Matematiche", "giocata", lambda: ParitaRitardo()),
    ("alti_bassi", "Alti / bassi", "Matematiche", "giocata", lambda: AltiBassi()),
    ("somma_estratti", "Somma dei 5 estratti", "Matematiche", "giocata", lambda: SommaTendenza()),
    ("consecutivi_gemelli", "Consecutivi e gemelli", "Matematiche", "giocata", lambda: Gemelli()),
    ("terminale_cadenza", "Terminali e cadenze", "Matematiche", "giocata", lambda: TerminaleRitardataria()),
    ("numeri_spia", "Numeri spia (dopo-numero)", "Matematiche", "giocata", lambda: NumeroSpia()),
    ("ambi_coppie", "Ambi/coppie frequenti", "Matematiche", "giocata", lambda: CoppiaFrequente()),
    ("vertibili", "Vertibili", "Matematiche", "giocata", lambda: Vertibile()),
    ("simmetrici", "Simmetrici (91-n)", "Matematiche", "giocata", lambda: Simmetrico()),
    ("markov", "Modello di Markov", "Matematiche", "giocata", lambda: Markov()),
    # --- Numerologiche ---
    ("num_somma_cabalistica", "Somma cabalistica data", "Numerologiche", "giocata",
     lambda: NumUnoData("num_somma_cabalistica", numerologia.somma_cabalistica_data)),
    ("num_teosofica", "Riduzione teosofica", "Numerologiche", "giocata",
     lambda: NumUnoData("num_teosofica", lambda d: numerologia.riduzione_teosofica(numerologia.somma_cabalistica_data(d)))),
    ("num_ciclo_pitagorico", "Ciclo pitagorico", "Numerologiche", "giocata",
     lambda: NumUnoData("num_ciclo_pitagorico", numerologia.ciclo_pitagorico)),
    ("num_giorno_mese", "Giorno+mese", "Numerologiche", "giocata",
     lambda: NumUnoData("num_giorno_mese", lambda d: d.day + d.month)),
    ("vibrazione_numero", "Vibrazione del numero", "Numerologiche", "giocata", lambda: VibrazioneGiorno()),
    ("somma_cabalistica_estratti", "Somma cabalistica estratti", "Numerologiche", "giocata", lambda: SommaCabalisticaEstratti()),
    ("radice_numerica_estratti", "Radice numerica estratti", "Numerologiche", "giocata", lambda: RadiceNumericaEstratti()),
    # --- Astrologiche / olistiche ---
    ("astro_fase_lunare", "Fase lunare", "Astrologiche", "giocata", lambda: AstroDaData("astro_fase_lunare", _num_fase_lunare)),
    ("astro_giorno_lunare", "Giorno lunare (1-30)", "Astrologiche", "giocata", lambda: AstroDaData("astro_giorno_lunare", _num_giorno_lunare)),
    ("astro_numeri_cielo", "Numeri dal cielo", "Astrologiche", "giocata", lambda: AstroDaData("astro_numeri_cielo", _num_dal_cielo)),
    ("astro_segno", "Segno → numeri", "Astrologiche", "giocata", lambda: AstroDaData("astro_segno", _num_segno)),
    ("astro_pianeta_giorno", "Pianeta → numero", "Astrologiche", "giocata", lambda: AstroDaData("astro_pianeta_giorno", _num_pianeta_giorno)),
    ("astro_transiti", "Transiti e aspetti", "Astrologiche", "giocata", lambda: AstroDaData("astro_transiti", _num_transiti)),
    ("astro_eclissi", "Eclissi (nodi lunari)", "Astrologiche", "giocata", lambda: AstroDaData("astro_eclissi", _num_eclissi)),
    ("astro_stagioni", "Stagioni / solstizi", "Astrologiche", "giocata", lambda: AstroDaData("astro_stagioni", _num_stagione)),
]

# Indice per chiave e per categoria
PER_CHIAVE = {r[0]: r for r in REGISTRO}
CATEGORIE = ["Matematiche", "Numerologiche", "Astrologiche"]


def regole_per_categoria() -> dict:
    """{categoria: [(chiave, nome), ...]} per costruire i menu di selezione."""
    out = {c: [] for c in CATEGORIE}
    for chiave, nome, cat, tipo, _ in REGISTRO:
        out[cat].append((chiave, nome))
    return out


def costruisci_strategie(chiavi=None) -> list:
    """Istanzia le strategie richieste (per chiave). Se chiavi=None: tutte."""
    if chiavi is None:
        chiavi = [r[0] for r in REGISTRO]
    strat = []
    for k in chiavi:
        if k in PER_CHIAVE:
            strat.append(PER_CHIAVE[k][4]())
    return strat


# ============================ UTILITÀ (richiedono input, non backtestabili) ============================

# Smorfia napoletana (estratto rappresentativo dei 90 significati tradizionali)
SMORFIA = {
    1: "l'Italia", 2: "la bambina", 3: "la gatta", 4: "il maialino", 5: "la mano",
    6: "quella che guarda giù", 7: "il vaso", 8: "la Madonna", 9: "la figliolanza", 10: "i fagioli",
    13: "Sant'Antonio", 17: "la disgrazia", 18: "il sangue", 22: "il pazzo", 25: "Natale",
    33: "gli anni di Cristo", 47: "il morto", 48: "il morto che parla", 77: "le gambe delle donne",
    88: "i caciocavalli", 90: "la paura",
}

def numerologia_nome(nome: str) -> int:
    """Valore numerologico di un nome/parola (pitagorico A=1..I=9, poi ridotto), in 1-90."""
    tot = 0
    for ch in nome.upper():
        if ch.isalpha():
            tot += (ord(ch) - ord("A")) % 9 + 1
    return _range(tot)
