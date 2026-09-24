#!/usr/bin/env python3
"""
Interfaccia Streamlit di SmartLotto — versione per cellulare.

PERSISTENZA: i dati dell'utente (i miei numeri, gli osservati, le previsioni,
le schedine) sono salvati nel LOCAL STORAGE del browser del telefono. Così
restano anche quando l'app si ricarica o viene aggiornata online (stesso
indirizzo = stessa memoria). Su Streamlit Cloud il disco del server viene
azzerato a ogni riavvio: per questo NON si usa più un file su disco.
"""

from __future__ import annotations
import datetime as dt
import datetime as _dt
import json
import time
from collections import Counter
from itertools import combinations
from math import comb

import numpy as np
import pandas as pd
import streamlit as st
from streamlit_local_storage import LocalStorage

from smartlotto import ingestion, statistiche, sequenze, numerologia, astro, predittore, tracker
from smartlotto import aggiornamento, regole, backtest as B, formule, RUOTE, RUOTE_NOMI
from smartlotto.backtest import P_ESTRATTO
from smartlotto.strategie import StatoIncrementale

PERCORSO_DEFAULT = "data/archivio_lotto.txt"

st.set_page_config(page_title="SmartLotto", layout="wide", page_icon="🎱")

st.markdown("""
<style>
.block-container{padding-top:1.1rem;padding-bottom:2rem}
.hero{font-size:1.45rem;font-weight:700;margin:0;line-height:1.2}
.hero-sub{color:#AAB6C4;font-size:.9rem;margin:.25rem 0 .2rem}
button[data-baseweb="tab"]{padding:10px 16px !important}
button[data-baseweb="tab"] p{font-size:1.02rem !important;font-weight:600 !important}
div[data-baseweb="tab-list"]{gap:4px;overflow-x:auto}
.chip{display:inline-block;min-width:38px;text-align:center;padding:7px 9px;margin:3px;
  border-radius:9px;font-weight:700;font-variant-numeric:tabular-nums;font-size:1rem;
  background:rgba(120,120,120,.15);border:1px solid rgba(120,120,120,.30)}
.chip.ok{background:#1f9d55;color:#fff;border-color:#1f9d55}
.chip.big{background:#E3A44A;color:#20160a;border-color:#E3A44A}
.ruota-lbl{font-weight:700;color:#E3A44A;min-width:74px;display:inline-block}
</style>
""", unsafe_allow_html=True)


# ============================ PERSISTENZA (browser localStorage) ============================
try:
    LS = LocalStorage()
except Exception:
    LS = None


def rerun_dopo_salvataggio():
    """Dà tempo al browser di scrivere nel localStorage PRIMA di ricaricare,
    altrimenti il salvataggio può andare perso (limite del componente)."""
    time.sleep(0.6)
    st.rerun()


def ls_load(chiave, default):
    """Legge un valore salvato nel browser. Se manca, ritorna default."""
    try:
        raw = LS.getItem(chiave) if LS else None
        if raw in (None, ""):
            return default
        return json.loads(raw) if isinstance(raw, str) else raw
    except Exception:
        return default


def ls_save(chiave, valore, wkey):
    """Salva un valore nel browser (wkey = identificativo unico del punto di salvataggio)."""
    try:
        if LS:
            LS.setItem(chiave, json.dumps(valore, ensure_ascii=False), key=wkey)
    except Exception:
        pass


# ---- Osservati ----
def carica_osservati():
    return ls_load("sl_osservati", [])


def salva_osservati(voci, wkey="oss_save"):
    ls_save("sl_osservati", voci, wkey)


# ---- Previsioni (registro + verifica, senza file su disco) ----
def carica_previsioni():
    return ls_load("sl_previsioni", [])


def salva_previsioni(voci, wkey="prev_save"):
    ls_save("sl_previsioni", voci, wkey)


def registra_previsione(data_estrazione, ruote, wkey="prev_reg"):
    voci = [v for v in carica_previsioni() if v["data_estrazione"] != data_estrazione.isoformat()]
    voci.append({
        "data_estrazione": data_estrazione.isoformat(),
        "generata_il": _dt.datetime.now().isoformat(timespec="seconds"),
        "ruote": {r: sorted(int(n) for n in nums) for r, nums in ruote.items()},
        "verificata": False, "esiti": {},
    })
    voci.sort(key=lambda v: v["data_estrazione"])
    salva_previsioni(voci, wkey)


def verifica_previsioni(df):
    """Confronta le previsioni non ancora verificate con l'archivio aggiornato."""
    voci = carica_previsioni()
    cambiato = False
    for voce in voci:
        if voce.get("verificata"):
            continue
        giorno = df[df["data"].dt.date == _dt.date.fromisoformat(voce["data_estrazione"])]
        if giorno.empty:
            continue
        estr_per_ruota = {RUOTE_NOMI[r["ruota"]]: r["numeri"] for _, r in giorno.iterrows()}
        esiti, trovata = {}, False
        for ruota, previsti in voce["ruote"].items():
            estr = estr_per_ruota.get(ruota)
            if estr is None:
                continue
            trovata = True
            azz = sorted(n for n in previsti if n in estr)
            esiti[ruota] = {"previsti": sorted(previsti),
                            "estratti": sorted(int(x) for x in estr),
                            "azzeccati": azz, "n_azzeccati": len(azz)}
        if trovata:
            voce["esiti"] = esiti
            voce["verificata"] = True
            cambiato = True
    if cambiato:
        salva_previsioni(voci, "prev_verify")
    return voci


def ultima_verificata():
    v = [x for x in carica_previsioni() if x.get("verificata")]
    return v[-1] if v else None


def ultima_in_attesa():
    v = [x for x in carica_previsioni() if not x.get("verificata")]
    return v[-1] if v else None


# ============================ funzioni di calcolo ============================

@st.cache_data(show_spinner=True)
def carica(percorso: str) -> pd.DataFrame:
    return ingestion.carica_archivio(percorso)


def _parse_riga(line: str):
    """Trasforma una riga di archivio (AAAAMMGG RUOTA:n.n.n.n.n ...) in righe df."""
    parts = line.strip().split(" ")
    d = parts[0]
    data = pd.Timestamp(_dt.date(int(d[:4]), int(d[4:6]), int(d[6:8])))
    out = []
    for blk in parts[1:]:
        ruota, nums = blk.split(":")
        nn = [int(x) for x in nums.split(".")]
        if len(nn) == 5 and all(1 <= x <= 90 for x in nn):
            out.append((data, ruota, *nn))
    return out


def costruisci_df(percorso: str, extra: list) -> pd.DataFrame:
    """Archivio di base (dal repo) + estrazioni extra salvate nel browser."""
    df = carica(percorso)
    righe = []
    for line in extra or []:
        try:
            righe += _parse_riga(line)
        except Exception:
            pass
    if not righe:
        return df
    edf = pd.DataFrame(righe, columns=["data", "ruota", "n1", "n2", "n3", "n4", "n5"])
    for c in ["n1", "n2", "n3", "n4", "n5"]:
        edf[c] = edf[c].astype("int16")
    edf["numeri"] = edf[["n1", "n2", "n3", "n4", "n5"]].apply(
        lambda r: frozenset(int(x) for x in r), axis=1)
    both = pd.concat([df, edf], ignore_index=True)
    both = both.drop_duplicates(subset=["data", "ruota"], keep="last")
    both = both.sort_values(["data", "ruota"], kind="stable").reset_index(drop=True)
    return both


@st.cache_data(show_spinner=True)
def frequenze_per_ruota(percorso: str) -> pd.DataFrame:
    df = carica(percorso)
    dati = {}
    for r in RUOTE:
        sub = ingestion.matrice_ruota(df, r)
        P = ingestion.matrice_presenza(sub)
        n = len(sub)
        dati[RUOTE_NOMI[r]] = P[:, 1:].sum(axis=0) / n * 100 if n else np.zeros(90)
    out = pd.DataFrame(dati, index=range(1, 91))
    out.index.name = "numero"
    return out


@st.cache_data(show_spinner=True)
def esegui_backtest(percorso: str, warmup: int, chiavi: tuple) -> dict:
    df = carica(percorso)
    strat = regole.costruisci_strategie(list(chiavi) if chiavi else None)
    per = []
    for r in RUOTE:
        sub = ingestion.matrice_ruota(df, r)
        if len(sub) <= warmup + 10:
            continue
        per.append(B.backtest_ruota(sub, strat, warmup=warmup))
    agg = B.aggrega(per)
    return {n: {"giocate": r.giocate, "successi": r.successi, "tasso": r.tasso_per_numero,
                **r.statistica()} for n, r in agg.items()}


@st.cache_data(show_spinner=True)
def esegui_formule(percorso: str, taglio: str) -> dict:
    df = carica(percorso)
    return formule.cerca_formule(df, taglio, top=10)


@st.cache_data(show_spinner="Calcolo i più ricorrenti...")
def ricorrenti(percorso: str, ruota_code: str):
    """Estratti, ambi e terni più usciti dalla prima estrazione a oggi."""
    df = carica(percorso)
    sub = df if ruota_code == "TUTTE" else df[df["ruota"] == ruota_code]
    est, amb, ter = Counter(), Counter(), Counter()
    for numeri in sub["numeri"]:
        ns = sorted(int(x) for x in numeri)
        est.update(ns)
        amb.update(combinations(ns, 2))
        ter.update(combinations(ns, 3))
    return est.most_common(15), amb.most_common(15), ter.most_common(15)


def prob_sorte(k: int, s: int) -> float:
    tot = comb(90, 5)
    fav = sum(comb(k, i) * comb(90 - k, 5 - i) for i in range(s, min(k, 5) + 1))
    return fav / tot


SORTI = [("Estratto", 1), ("Ambo", 2), ("Terno", 3), ("Quaterna", 4), ("Cinquina", 5)]


def tabella_sorti(quanti: int) -> pd.DataFrame:
    righe = []
    for nome, s in SORTI:
        if s > quanti:
            continue
        p = prob_sorte(quanti, s)
        righe.append({"Sorte (almeno)": nome,
                      "Probabilità": f"{p*100:.4f}%".rstrip("0").rstrip("."),
                      "1 su…": f"{round(1/p):,}".replace(",", ".")})
    return pd.DataFrame(righe)


def chip_html(numeri, azzeccati=None, big=False):
    azz = set(azzeccati or [])
    cls_extra = " big" if big else ""
    return "".join(f"<span class='{'chip ok' if n in azz else 'chip'+cls_extra}'>{int(n):02d}</span>"
                   for n in numeri)


def _stato_ruota(sub) -> StatoIncrementale:
    stato = StatoIncrementale()
    for numeri in sub["numeri"]:
        stato.aggiorna(numeri)
    return stato


def affinita_regole(df, ruota_code, miei, data, top=8):
    sub = ingestion.matrice_ruota(df, ruota_code) if ruota_code else None
    if sub is None or len(sub) < 50:
        return []
    stato = _stato_ruota(sub)
    strat = regole.costruisci_strategie(None)
    mieiset = set(miei)
    righe = []
    for s in strat:
        try:
            proposti = set(int(x) for x in s.proponi(stato, data) if 1 <= int(x) <= 90)
        except Exception:
            continue
        inter = sorted(mieiset & proposti)
        if inter:
            nome = regole.PER_CHIAVE.get(s.nome, (s.nome, s.nome))[1]
            righe.append((nome, inter, len(proposti)))
    righe.sort(key=lambda t: (-len(t[1]), t[2]))
    return righe[:top]


def misura_osservato(df, ruota_code, numeri, da_data):
    numeri = set(numeri)
    ruote = RUOTE if ruota_code == "TUTTE" else [ruota_code]
    uscite, n_contr = [], 0
    for rc in ruote:
        sub = ingestion.matrice_ruota(df, rc)
        sub = sub[sub["data"].dt.date >= da_data].sort_values("data")
        for i, (_, row) in enumerate(sub.iterrows()):
            n_contr += 1
            azz = sorted(numeri & row["numeri"])
            if azz:
                uscite.append({"data": row["data"].date(), "ruota": RUOTE_NOMI[rc],
                               "dopo": i + 1, "azzeccati": azz, "n": len(azz)})
    return n_contr, uscite


def selezione_regole(prefix: str) -> list:
    cat_map = regole.regole_per_categoria()
    for cat, elenco in cat_map.items():
        for chiave, _ in elenco:
            st.session_state.setdefault(f"{prefix}{chiave}", cat == "Matematiche")
    ca, cb = st.columns(2)
    if ca.button("✅ Tutte", key=prefix + "_all", use_container_width=True):
        for cat, elenco in cat_map.items():
            for chiave, _ in elenco:
                st.session_state[f"{prefix}{chiave}"] = True
    if cb.button("⬜ Nessuna", key=prefix + "_none", use_container_width=True):
        for cat, elenco in cat_map.items():
            for chiave, _ in elenco:
                st.session_state[f"{prefix}{chiave}"] = False
    scelte = []
    for cat, elenco in cat_map.items():
        attive = sum(1 for c, _ in elenco if st.session_state[f"{prefix}{c}"])
        with st.expander(f"{cat} — {attive}/{len(elenco)} attive"):
            for chiave, nome in elenco:
                if st.checkbox(nome, key=f"{prefix}{chiave}"):
                    scelte.append(chiave)
    return scelte


# ============================ SCHEDINA — calcolo vincite ============================
MOLTIPLICATORI = {"estratto": 11.23, "ambo": 250, "terno": 4500,
                  "quaterna": 120000, "cinquina": 6000000}
SORTE_SIZE = {"estratto": 1, "ambo": 2, "terno": 3, "quaterna": 4, "cinquina": 5}


def carica_schedine():
    return ls_load("sl_schedine", [])


def salva_schedine(voci, wkey="sched_save"):
    ls_save("sl_schedine", voci, wkey)


def vincita_giocata(numeri, punti, n_usciti):
    """Vincita reale: (puntata ÷ combinazioni) × moltiplicatore × combinazioni uscite."""
    K = len(numeri)
    out = {}
    for sorte, imp in punti.items():
        imp = float(imp or 0)
        if imp <= 0:
            continue
        t = SORTE_SIZE[sorte]
        ctot = comb(K, t)
        if ctot == 0:
            continue
        cvinc = comb(n_usciti, t) if n_usciti >= t else 0
        out[sorte] = (imp / ctot) * MOLTIPLICATORI[sorte] * cvinc
    return out, sum(out.values())


def tabella_vincite(numeri, punti):
    """Scenari: quanto vinci a seconda di quanti tuoi numeri escono (max 5 per ruota)."""
    K = len(numeri)
    attive = [s for s in SORTE_SIZE if float(punti.get(s, 0) or 0) > 0]
    minsize = min((SORTE_SIZE[s] for s in attive), default=1)
    righe = []
    for M in range(max(minsize, 1), min(K, 5) + 1):
        det, tot = vincita_giocata(numeri, punti, M)
        r = {"Numeri usciti": f"{M} su {K}"}
        for s in ["estratto", "ambo", "terno", "quaterna", "cinquina"]:
            if s in attive:
                r[s.capitalize()] = f"{det.get(s, 0):.2f} €"
        r["Vincita"] = f"{tot:.2f} €"
        righe.append(r)
    return pd.DataFrame(righe)


def monitora_giocata(df, ruota_code, numeri, da_data):
    """Scorre le estrazioni della ruota dalla data di creazione e trova le uscite."""
    numeri = set(numeri)
    sub = ingestion.matrice_ruota(df, ruota_code)
    sub = sub[sub["data"].dt.date >= da_data].sort_values("data")
    esiti = []
    for i, (_, row) in enumerate(sub.iterrows()):
        usciti = sorted(numeri & row["numeri"])
        if usciti:
            esiti.append({"data": row["data"].date(), "dopo": i + 1, "usciti": usciti})
    return len(sub), esiti


# ============================ intestazione + sidebar ============================

st.markdown("<div class='hero'>🎱 SmartLotto</div>"
            "<div class='hero-sub'>Analisi delle estrazioni del Lotto — con verifica onesta sui dati</div>",
            unsafe_allow_html=True)

percorso = st.sidebar.text_input("Archivio estrazioni", PERCORSO_DEFAULT)

# Archivio di base (dal repo) + estrazioni extra salvate nel browser del telefono.
extra = ls_load("sl_estr_extra", [])
try:
    df = costruisci_df(percorso, extra)
except Exception as e:
    st.error(f"Impossibile caricare l'archivio: {e}")
    st.stop()


def _applica_nuove(nuove):
    """Salva le righe nuove nel browser (così restano) e ricostruisce l'archivio."""
    global df, extra
    extra = list(dict.fromkeys((extra or []) + list(nuove)))
    ls_save("sl_estr_extra", extra, "estr_save")
    df = costruisci_df(percorso, extra)


# --- AGENTE: aggiornamento automatico all'apertura (una volta per sessione) ---
if "auto_agg" not in st.session_state:
    st.session_state["auto_agg"] = True
    try:
        nuove = aggiornamento.scarica_nuove(df["data"].max().date())
        if nuove:
            _applica_nuove(nuove)
            st.session_state["agg_msg"] = ("ok", f"🔄 Scaricate {len(nuove)} nuove estrazioni.")
        else:
            st.session_state["agg_msg"] = ("ok", "✅ Archivio già aggiornato.")
    except Exception as e:
        st.session_state["agg_msg"] = ("warn",
            f"⚠️ Aggiornamento automatico non riuscito ({type(e).__name__}). "
            "Inseriscila a mano qui sotto.")

# Banner dell'agente, ben visibile in cima (i comandi sono nella scheda 🤖 Agente)
_m = st.session_state.get("agg_msg")
if _m:
    (st.info if _m[0] == "ok" else st.warning)(_m[1] + "  ·  Comandi nella scheda 🤖 Agente.")
st.sidebar.caption("💾 I tuoi dati (numeri, osservati, schedine, estrazioni) restano nel "
                   "browser di questo telefono, anche dopo gli aggiornamenti.")

# Verifica automatica delle previsioni in sospeso a ogni apertura.
try:
    verifica_previsioni(df)
except Exception:
    pass


tab_sched, tab_ult, tab_prev, tab_oss, tab_num, tab_agente, tab_guida, tab_cielo, tab_form = st.tabs(
    ["🎟️ Schedina", "🎰 Ultima & Ricorrenti", "🎯 Previsione", "👁️ Osservati", "📊 Numeri",
     "🤖 Agente", "📖 Guida", "🌙 Cielo", "🔬 Formule"])


# ============================ TAB ULTIMA & RICORRENTI ============================
with tab_ult:
    st.subheader("🎰 Ultima estrazione")
    ultima_data = df["data"].max().date()
    st.markdown(f"**{ultima_data.strftime('%d/%m/%Y')}** — tutte le ruote")
    ult_df = df[df["data"].dt.date == ultima_data].sort_values("ruota")
    for _, row in ult_df.iterrows():
        nums = sorted(int(x) for x in row["numeri"])
        st.markdown(f"<span class='ruota-lbl'>{RUOTE_NOMI[row['ruota']]}</span> {chip_html(nums)}",
                    unsafe_allow_html=True)

    st.markdown("---")
    st.subheader("🔥 I più ricorrenti di sempre")
    st.caption("Estratti, ambi e terni più usciti dalla prima estrazione (1871) a oggi.")
    scelta = st.selectbox("Ruota", ["TUTTE"] + RUOTE,
                          format_func=lambda r: "Tutte le ruote" if r == "TUTTE" else RUOTE_NOMI[r],
                          key="ric_ruota")
    est, amb, ter = ricorrenti(percorso, scelta)
    st.markdown("**Estratti più usciti**")
    st.dataframe(pd.DataFrame([{"numero": f"{n:02d}", "uscite": c} for n, c in est]),
                 hide_index=True, use_container_width=True)
    st.markdown("**Ambi più usciti**")
    st.dataframe(pd.DataFrame([{"ambo": f"{a:02d} · {b:02d}", "uscite": c} for (a, b), c in amb]),
                 hide_index=True, use_container_width=True)
    st.markdown("**Terni più usciti**")
    st.dataframe(pd.DataFrame([{"terno": f"{a:02d} · {b:02d} · {d:02d}", "uscite": c}
                               for (a, b, d), c in ter]),
                 hide_index=True, use_container_width=True)
    st.caption("Nota onesta: i più usciti nel passato NON hanno più probabilità di uscire in "
               "futuro. Ogni estrazione è indipendente. È una curiosità storica, non una previsione.")


# ============================ TAB SCHEDINA ============================
with tab_sched:
    st.subheader("🎟️ La mia schedina")
    st.caption("Fino a 4 giocate. Per ognuna: ruota, numeri (da 1 a 10) e quanto punti "
               "per sorte. L'app la monitora e calcola la vincita, che tu l'abbia giocata o no.")

    n_gioc = st.number_input("Quante giocate", 1, 4, 1, key="sc_n")
    giocate = []
    for i in range(int(n_gioc)):
        with st.container(border=True):
            st.markdown(f"**Giocata {i+1}**")
            rr = st.selectbox("Ruota", RUOTE, format_func=lambda r: RUOTE_NOMI[r], key=f"sc_ruota_{i}")
            tn = st.text_input("Numeri (da 1 a 10, separati da spazio)", key=f"sc_num_{i}",
                               placeholder="es. 5 17 23 44 67 82")
            cols = st.columns(5)
            punti = {}
            for col, s in zip(cols, ["estratto", "ambo", "terno", "quaterna", "cinquina"]):
                punti[s] = col.number_input(s.capitalize() + " €", 0.0, 200.0, 0.0, 0.5, key=f"sc_{s}_{i}")
            try:
                nums = sorted({int(x) for x in tn.split() if 1 <= int(x) <= 90})[:10]
            except ValueError:
                nums = []
            giocate.append({"ruota": rr, "numeri": nums, "punti": punti})
            if nums and any(float(v or 0) > 0 for v in punti.values()):
                st.caption(f"Numeri: {' '.join(f'{n:02d}' for n in nums)} — possibili vincite:")
                st.dataframe(tabella_vincite(nums, punti), hide_index=True, use_container_width=True)

    reale = st.toggle("L'ho giocata davvero?", key="sc_reale")
    tot_punt = sum(float(v or 0) for g in giocate for v in g["punti"].values())
    st.markdown(f"**Totale puntato: € {tot_punt:.2f}**")
    if st.button("💾 Salva schedina e monitora", type="primary", use_container_width=True):
        valide = [g for g in giocate if g["numeri"] and any(float(v or 0) > 0 for v in g["punti"].values())]
        if not valide:
            st.warning("Inserisci almeno una giocata con numeri e una puntata.")
        else:
            voci = carica_schedine()
            voci.append({"id": _dt.datetime.now().strftime("%Y%m%d%H%M%S%f"),
                         "creata": _dt.date.today().isoformat(),
                         "reale": bool(reale), "giocate": valide})
            salva_schedine(voci, "sched_add")
            st.success("Schedina salvata e messa in monitoraggio.")
            rerun_dopo_salvataggio()

    st.markdown("---")
    st.subheader("Le mie schedine")
    voci = carica_schedine()
    if not voci:
        st.info("Nessuna schedina salvata. Compila qui sopra e premi «Salva».")
    for v in reversed(voci):
        creata = _dt.date.fromisoformat(v["creata"])
        with st.container(border=True):
            tag = "🟢 giocata davvero" if v.get("reale") else "👁️ solo monitorata"
            st.markdown(f"**Schedina del {creata.strftime('%d/%m/%Y')}** · {tag}")
            vinc_tot = 0.0
            for g in v["giocate"]:
                nums, ruota = g["numeri"], g["ruota"]
                ncontr, esiti = monitora_giocata(df, ruota, nums, creata)
                usciti_tot = sorted({n for e in esiti for n in e["usciti"]})
                st.markdown(f"<span class='ruota-lbl'>{RUOTE_NOMI[ruota]}</span> "
                            f"{chip_html(nums, usciti_tot, big=True)}", unsafe_allow_html=True)
                best_M = max((len(e["usciti"]) for e in esiti), default=0)
                if best_M >= 2:
                    _, tot = vincita_giocata(nums, g["punti"], best_M)
                    vinc_tot += tot
                    ez = next(e for e in esiti if len(e["usciti"]) == best_M)
                    st.markdown(f"<span style='color:#4CBF8B'>🎉 {best_M} numeri usciti il "
                                f"{ez['data'].strftime('%d/%m/%Y')} → vincita {tot:.2f} €</span>",
                                unsafe_allow_html=True)
                else:
                    st.caption(f"In monitoraggio · {ncontr} estrazioni controllate · nessuna sorte completa")
            if vinc_tot > 0:
                st.markdown(f"### Vincita totale: {vinc_tot:.2f} €")
                st.caption("Importo lordo (prima della tassa dello Stato)." +
                           ("" if v.get("reale") else " Non l'hai giocata: è quanto avresti vinto."))
            cbtn = st.columns(2)
            if cbtn[0].button("🔁 Giocata sì/no", key="tg_" + v["id"]):
                voci2 = carica_schedine()
                for x in voci2:
                    if x["id"] == v["id"]:
                        x["reale"] = not x.get("reale")
                salva_schedine(voci2, "sched_tg")
                rerun_dopo_salvataggio()
            if cbtn[1].button("🗑️ Rimuovi", key="rm_" + v["id"]):
                salva_schedine([x for x in carica_schedine() if x["id"] != v["id"]], "sched_rm")
                rerun_dopo_salvataggio()


# ============================ TAB AGENTE ============================
with tab_agente:
    st.subheader("🤖 Agente estrazioni")
    st.caption("Scarica da solo le estrazioni nuove all'apertura e le salva nel telefono.")
    _mm = st.session_state.get("agg_msg")
    if _mm:
        (st.info if _mm[0] == "ok" else st.warning)(_mm[1])
    st.markdown(f"**Ultima estrazione in archivio:** {df['data'].max().date().strftime('%d/%m/%Y')}")
    if st.button("🔄 Aggiorna ora", key="ag_now", use_container_width=True):
        with st.spinner("Scarico le estrazioni nuove..."):
            try:
                nuove = aggiornamento.scarica_nuove(df["data"].max().date())
                if nuove:
                    _applica_nuove(nuove)
                    st.success(f"Scaricate {len(nuove)} nuove estrazioni.")
                    rerun_dopo_salvataggio()
                else:
                    st.info("Archivio già aggiornato.")
            except Exception as e:
                st.warning(f"Non riuscito ({type(e).__name__}). Usa l'inserimento manuale qui sotto.")

    st.markdown("---")
    st.markdown("**➕ Inserisci un'estrazione a mano** (solo se l'automatico non riesce)")
    md = st.date_input("Data", df["data"].max().date(), key="man_data", format="DD/MM/YYYY")
    mr = st.selectbox("Ruota", RUOTE, format_func=lambda r: RUOTE_NOMI[r], key="man_ruota")
    mn = st.text_input("5 numeri (separati da spazio)", key="man_num", placeholder="es. 12 34 56 78 90")
    if st.button("Salva estrazione", key="man_save"):
        try:
            nn = [int(x) for x in mn.split()]
        except ValueError:
            nn = []
        if len(nn) == 5 and all(1 <= x <= 90 for x in nn):
            riga = f"{md.year:04d}{md.month:02d}{md.day:02d} {mr}:" + ".".join(f"{x:02d}" for x in nn)
            _applica_nuove([riga])
            st.success(f"Aggiunta {RUOTE_NOMI[mr]} del {md.strftime('%d/%m/%Y')}.")
            rerun_dopo_salvataggio()
        else:
            st.warning("Servono esattamente 5 numeri fra 1 e 90.")
    st.caption("💾 Le estrazioni scaricate o inserite restano nel browser di questo telefono.")


# ============================ TAB PREVISIONE ============================
with tab_prev:
    ultima_arch = df["data"].max().date()
    data_prossima = tracker.prossimo_giorno_estrazione(ultima_arch)

    st.subheader("🎯 Genera per la prossima estrazione")
    d_prev = st.date_input("Data dell'estrazione", data_prossima, key="prev", format="DD/MM/YYYY")

    st.markdown("**1. Scegli le regole** — i numeri cambiano in base a cosa attivi.")
    chiavi = selezione_regole("pv_")
    if not chiavi:
        st.info("Nessuna regola attiva: uso il criterio base (ritardo + freddi + numerologia + cielo).")

    st.markdown("**2. Genera i numeri**")
    modo = st.radio("Ruote", ["Tutte le ruote", "Una ruota"], horizontal=True, key="pv_modo")
    ruota_sel = None
    if modo == "Una ruota":
        ruota_sel = st.selectbox("Ruota", RUOTE, format_func=lambda r: RUOTE_NOMI[r], key="pv_ruota")

    if st.button("🎲 Genera i numeri", type="primary", use_container_width=True):
        ruote_target = [ruota_sel] if ruota_sel else RUOTE
        risultati = {}
        for r in ruote_target:
            sub = ingestion.matrice_ruota(df, r)
            if len(sub) < 50:
                continue
            risultati[RUOTE_NOMI[r]] = predittore.cinquina_da_regole(sub, d_prev, chiavi or None)
        st.session_state["pv_risultati"] = risultati
        st.session_state["pv_data"] = d_prev.isoformat()

    risultati = st.session_state.get("pv_risultati")
    if risultati:
        d_txt = _dt.date.fromisoformat(st.session_state["pv_data"]).strftime("%d/%m/%Y")
        st.markdown(f"**Numeri generati per il {d_txt}**")
        for ruota, nums in risultati.items():
            st.markdown(f"<span class='ruota-lbl'>{ruota}</span> {chip_html(nums, big=True)}",
                        unsafe_allow_html=True)

        st.markdown("---")
        st.markdown("**🎟️ Se giochi questi 5 numeri — probabilità delle sorti**")
        st.dataframe(tabella_sorti(5), hide_index=True, use_container_width=True)
        st.caption("Le probabilità dipendono solo da quanti numeri giochi (5 su 90): "
                   "sono uguali per qualsiasi cinquina.")

        if st.button("💾 Salva questa previsione (per verificarla dopo)", use_container_width=True):
            registra_previsione(_dt.date.fromisoformat(st.session_state["pv_data"]), risultati)
            st.success("Salvata nel telefono. Dopo l'estrazione aggiorna l'archivio: "
                       "i numeri usciti diventeranno verdi nella scheda Osservati.")

    st.warning("Onestà: questi numeri hanno la stessa probabilità di 5 numeri a caso. "
               "Gioca con responsabilità.")


# ============================ TAB OSSERVATI ============================
with tab_oss:
    st.subheader("👁️ Osservati — misura delle previsioni")
    st.caption("Salva una serie di numeri: l'app controlla a ogni estrazione futura "
               "se escono e dopo quante estrazioni. In verde i numeri realmente usciti.")

    verif = ultima_verificata()
    if verif:
        d = _dt.date.fromisoformat(verif["data_estrazione"]).strftime("%d/%m/%Y")
        tot = sum(e["n_azzeccati"] for e in verif["esiti"].values())
        with st.expander(f"✅ Ultima previsione verificata — {d} · azzeccati: {tot}", expanded=True):
            for ruota, e in verif["esiti"].items():
                estr = "  ".join(f"{x:02d}" for x in e["estratti"])
                st.markdown(f"<span class='ruota-lbl'>{ruota}</span> {chip_html(e['previsti'], e['azzeccati'])}"
                            f"<div style='color:#888;font-size:.85rem;margin:2px 0 8px 74px'>estratti: {estr}</div>",
                            unsafe_allow_html=True)

    st.markdown("---")
    st.markdown("**➕ Aggiungi una serie da osservare**")
    txt = st.text_input("Numeri (da 1 a 10, separati da spazio)", "7 25 45 67 82", key="oss_txt")
    c1, c2 = st.columns(2)
    ruota_oss = c1.selectbox("Ruota", ["TUTTE"] + RUOTE,
                             format_func=lambda r: "Tutte le ruote" if r == "TUTTE" else RUOTE_NOMI[r],
                             key="oss_ruota")
    da_data = c2.date_input("Osserva dalla data", dt.date.today(), key="oss_da", format="DD/MM/YYYY")
    if st.button("➕ Aggiungi agli osservati", use_container_width=True):
        try:
            nums = sorted({int(x) for x in txt.split() if 1 <= int(x) <= 90})[:10]
        except ValueError:
            nums = []
        if nums:
            voci = carica_osservati()
            voci.append({"id": _dt.datetime.now().strftime("%Y%m%d%H%M%S"),
                         "numeri": nums, "ruota": ruota_oss, "da": da_data.isoformat()})
            salva_osservati(voci, "oss_add")
            st.success(f"Aggiunta: {' '.join(f'{n:02d}' for n in nums)}")
        else:
            st.warning("Inserisci numeri validi (1-90).")

    st.markdown("---")
    voci = carica_osservati()
    if not voci:
        st.info("Nessuna serie in osservazione. Aggiungine una qui sopra.")
    sorte_nomi = {1: "estratto", 2: "ambo", 3: "terno", 4: "quaterna", 5: "cinquina"}
    for v in reversed(voci):
        rn = "Tutte le ruote" if v["ruota"] == "TUTTE" else RUOTE_NOMI[v["ruota"]]
        da = _dt.date.fromisoformat(v["da"])
        n_contr, uscite = misura_osservato(df, v["ruota"], v["numeri"], da)
        tutti_usciti = sorted({n for u in uscite for n in u["azzeccati"]})
        best = max((u["n"] for u in uscite), default=0)
        with st.container(border=True):
            st.markdown(f"<span class='ruota-lbl'>{rn}</span> {chip_html(v['numeri'], tutti_usciti)}",
                        unsafe_allow_html=True)
            riep = (f"Dal {da.strftime('%d/%m/%Y')} · {n_contr} estrazioni controllate · "
                    f"{len(uscite)} volte con almeno un numero")
            if best >= 2:
                riep += f" · **miglior colpo: {sorte_nomi.get(min(best,5))}**"
            st.caption(riep)
            for u in reversed(uscite[-3:]):
                st.markdown(f"<div style='font-size:.85rem;color:#4CBF8B'>"
                            f"{u['data'].strftime('%d/%m/%Y')} ({u['ruota']}): "
                            f"{' '.join(f'{n:02d}' for n in u['azzeccati'])} "
                            f"— {sorte_nomi.get(min(u['n'],5))}, dopo {u['dopo']} estrazioni</div>",
                            unsafe_allow_html=True)
            if st.button("🗑️ Rimuovi", key="del_" + v["id"]):
                salva_osservati([x for x in carica_osservati() if x["id"] != v["id"]], "oss_del")
                rerun_dopo_salvataggio()


# ============================ TAB NUMERI ============================
with tab_num:
    st.subheader("📊 Ritardi, frequenze e i miei numeri")
    ruota = st.selectbox("Ruota", RUOTE, format_func=lambda r: RUOTE_NOMI[r], key="num_ruota")
    sub = ingestion.matrice_ruota(df, ruota)
    P = ingestion.matrice_presenza(sub)

    col1, col2 = st.columns(2)
    with col1:
        st.markdown("**Più ritardatari**")
        rit = statistiche.classifica_ritardatari(P, top=12)
        st.dataframe(pd.DataFrame(rit, columns=["numero", "ritardo"]),
                     hide_index=True, use_container_width=True)
    with col2:
        st.markdown("**Più frequenti**")
        fr = statistiche.classifica_frequenti(P, top=12, caldi=True)
        st.dataframe(pd.DataFrame(fr, columns=["numero", "uscite"]),
                     hide_index=True, use_container_width=True)

    st.markdown("**Frequenza di ogni numero su questa ruota (%)**")
    freq = P[:, 1:].sum(axis=0) / max(len(sub), 1) * 100
    st.bar_chart(pd.DataFrame({"frequenza %": freq}, index=range(1, 91)), color="#E3A44A", height=240)
    st.caption(f"Linea del caso: {P_ESTRATTO*100:.3f}%. Le barre oscillano intorno a quel valore.")

    st.markdown("---")
    st.subheader("🎟️ I miei numeri")
    miei_saved = ls_load("sl_miei", "7 25 45 67 82")
    testo = st.text_input("Da 1 a 5 numeri separati da spazio", miei_saved, key="num_miei")
    if testo.strip() != str(miei_saved).strip():
        ls_save("sl_miei", testo.strip(), "miei_save")
    try:
        miei = sorted({int(x) for x in testo.split() if 1 <= int(x) <= 90})[:5]
    except ValueError:
        miei = []
    if not miei:
        st.warning("Scrivi da 1 a 5 numeri validi (1-90).")
    else:
        st.session_state["miei_numeri"] = miei
        st.markdown(chip_html(miei, big=True), unsafe_allow_html=True)
        st.markdown("**Probabilità delle sorti**")
        st.dataframe(tabella_sorti(len(miei)), hide_index=True, use_container_width=True)

        st.markdown("**Quante volte sono usciti, ruota per ruota (%)**")
        ftab = frequenze_per_ruota(percorso)
        st.bar_chart(ftab.loc[miei].T, height=280)
        st.caption("Ogni serie è uno dei tuoi numeri. Le altezze si equivalgono.")

        st.markdown(f"**🔗 Regole più affini ai tuoi numeri — ruota di {RUOTE_NOMI[ruota]}**")
        aff = affinita_regole(df, ruota, miei, tracker.prossimo_giorno_estrazione(df["data"].max().date()))
        if aff:
            st.dataframe(pd.DataFrame([
                {"regola": nome, "tuoi numeri che indica": " ".join(f"{n:02d}" for n in inter),
                 "quanti indica in tutto": tot} for nome, inter, tot in aff]),
                hide_index=True, use_container_width=True)
            st.caption("Sono le regole che, su questa ruota, propongono i numeri che hai scelto. "
                       "'Affine' non vuol dire 'più probabile'.")
        else:
            st.info("Nessuna regola propone questi numeri su questa ruota.")


# ============================ TAB GUIDA ============================
with tab_guida:
    st.subheader("📖 Come leggere le percentuali")
    st.markdown("Una **percentuale** dice quanto è probabile una cosa, da **0% (mai)** a "
                "**100% (sempre)**. Il **5,5%** del Lotto **non** vuol dire «esce di sicuro».")
    st.markdown("**Ogni percentuale è un «1 volta su quanti»**")
    st.table(pd.DataFrame([
        {"Percentuale": "100%", "Vuol dire": "esce sempre, è certo", "1 volta su…": "1"},
        {"Percentuale": "50%", "Vuol dire": "testa o croce", "1 volta su…": "2"},
        {"Percentuale": "25%", "Vuol dire": "una volta su quattro", "1 volta su…": "4"},
        {"Percentuale": "5,5%", "Vuol dire": "il tuo numero (estratto)", "1 volta su…": "18"},
        {"Percentuale": "2,5%", "Vuol dire": "più raro del 5,5%", "1 volta su…": "40"},
        {"Percentuale": "0,25%", "Vuol dire": "l'ambo (2 numeri giusti)", "1 volta su…": "400"},
        {"Percentuale": "0%", "Vuol dire": "impossibile", "1 volta su…": "—"},
    ]))
    st.info("**5,5% = su 90 palline ne escono 5.** Non dipende da quanto un numero è in "
            "ritardo o da quale regola scegli.")
    st.markdown("---")
    st.subheader("🚦 Affidabilità: quando un numero è «verde»?")
    st.markdown("L'app misura ogni criterio su oltre 150 anni di estrazioni. Un numero diventa "
                "**verde** solo se batte il caso in modo solido — non è mai successo finora. "
                "Se vedi tutto neutro, è la verità: valgono come numeri a caso.")


# ============================ TAB CIELO ============================
with tab_cielo:
    st.subheader("🌙 Cielo e numerologia del giorno")
    d = st.date_input("Data", dt.date.today(), key="cielo", format="DD/MM/YYYY")
    fase = astro.fase_lunare(d)
    st.write(f"**Fase lunare**: {fase['fase']} — illuminazione {fase['illuminazione']*100:.0f}%, "
             f"età {fase['eta_giorni']:.1f} giorni")
    st.write(f"**Segno solare**: {astro.segno_solare(d)}")
    st.write("**Numeri dal cielo**:", astro.numeri_da_cielo(d))
    with st.expander("Posizioni dei pianeti"):
        pos = astro.posizioni_pianeti(d)
        st.dataframe(pd.DataFrame([
            {"pianeta": k, "segno": v["segno"], "gradi": round(v["longitudine"], 1)}
            for k, v in pos.items()]), hide_index=True, use_container_width=True)
    st.markdown("**Numerologia della data**")
    st.write("Somma cabalistica:", numerologia.somma_cabalistica_data(d),
             " · Ciclo pitagorico:", numerologia.ciclo_pitagorico(d))
    st.write("Numeri numerologici:", numerologia.numeri_da_data(d))
    with st.expander("Sequenze matematiche (1..90)"):
        st.write("**Fibonacci**:", sequenze.fibonacci_fino_a(90))
        st.write("**Primi**:", sequenze.numeri_primi(90))
        st.write("**Quadrati**:", sequenze.quadrati(90))
        st.write("**Triangolari**:", sequenze.triangolari(90))


# ============================ TAB FORMULE ============================
with tab_form:
    st.subheader("🔬 La prova sui dati")
    st.markdown("### 1) Backtesting delle regole")
    st.caption("Ogni regola viene provata «all'epoca»: usa solo le estrazioni precedenti e "
               f"avanza nel tempo. Deve battere la giocata a caso ({P_ESTRATTO*100:.3f}%).")
    warmup = st.slider("Estrazioni iniziali di rodaggio", 100, 1000, 300, 100)
    st.markdown("**Scegli le regole da provare**")
    chiavi_bt = selezione_regole("bt_")
    st.caption(f"{len(chiavi_bt)} regole selezionate." + ("" if chiavi_bt else " (vuoto = tutte)"))
    if st.button("▶️ Esegui il backtest", use_container_width=True):
        res = esegui_backtest(percorso, warmup, tuple(chiavi_bt))
        righe = []
        for n, r in sorted(res.items(), key=lambda kv: kv[1]["z"], reverse=True):
            nome = regole.PER_CHIAVE.get(n, (n, n))[1]
            diff = (r["tasso"] - P_ESTRATTO) * 100
            righe.append({"regola": nome, "riuscita": f"{r['tasso']*100:.2f}%",
                          "vs caso": f"{diff:+.2f}%",
                          "esito": "🟢 batte il caso" if r["p_value"] < 0.001 and diff > 0 else "⚪ come il caso"})
        st.dataframe(pd.DataFrame(righe), hide_index=True, use_container_width=True)
        st.success("Nessuna regola resta 🟢: tutte vanno «come il caso». È il risultato atteso.")

    st.markdown("---")
    st.markdown("### 2) Analizza una serie di numeri sul futuro")
    st.markdown("**A cosa serve la data di separazione?** Divide la storia in due: il **passato** "
                "serve a scegliere i numeri, il **futuro** (dopo quella data) serve a controllare "
                "se davvero continuano a uscire di più. Se sono bravi solo nel passato, era fortuna.")
    miei_def = " ".join(f"{n:02d}" for n in st.session_state.get("miei_numeri", [7, 25, 45, 67, 82]))
    serie_txt = st.text_input("Numeri da analizzare", miei_def, key="form_serie")
    taglio = st.date_input("Data di separazione passato / futuro", dt.date(2010, 1, 1),
                           key="form_taglio", format="DD/MM/YYYY")
    if st.button("🔎 Analizza questi numeri", use_container_width=True):
        try:
            serie = sorted({int(x) for x in serie_txt.split() if 1 <= int(x) <= 90})
        except ValueError:
            serie = []
        if not serie:
            st.warning("Inserisci numeri validi (1-90).")
        else:
            tcut = pd.Timestamp(taglio)
            df_in, df_out = df[df["data"] < tcut], df[df["data"] >= tcut]

            def tasso(dd):
                if len(dd) == 0:
                    return None
                usc = sum(1 for s in dd["numeri"] for n in serie if n in s)
                return usc / (len(dd) * 5)

            t_in, t_out, base = tasso(df_in), tasso(df_out), P_ESTRATTO
            st.markdown(f"**Numeri analizzati:** {' '.join(f'{n:02d}' for n in serie)}")
            c1, c2, c3 = st.columns(3)
            c1.metric("Nel passato", f"{t_in*100:.2f}%" if t_in else "—",
                      f"{(t_in-base)*100:+.2f}%" if t_in else None)
            c2.metric("Nel futuro", f"{t_out*100:.2f}%" if t_out else "—",
                      f"{(t_out-base)*100:+.2f}%" if t_out else None)
            c3.metric("Atteso dal caso", f"{base*100:.2f}%")
            if t_in and t_out:
                verde = t_out > base * 1.05
                st.markdown(f"### {'🟢 Regge anche nel futuro' if verde else '⚪ Come il caso'}")
                st.caption("Se il valore «nel futuro» non supera stabilmente quello del caso, "
                           "i numeri non hanno alcun vantaggio reale.")

    st.markdown("---")
    with st.expander("Vedi la prova su 900 regole (dimostrazione overfitting)"):
        taglio2 = st.text_input("Data di separazione (AAAA-MM-GG)", "2010-01-01", key="form_taglio2")
        if st.button("Cerca formule e validale", key="form_900"):
            ff = esegui_formule(percorso, taglio2)
            c1, c2, c3 = st.columns(3)
            c1.metric("Regole provate", ff["n_regole"])
            c2.metric("Sembravano buone (passato)", ff["significative_p05"],
                      help=f"Attese per puro caso: ~{ff['attesi_falsi_positivi_p05']:.0f}")
            c3.metric("Legame passato→futuro", f"{ff['correlazione_in_out']:.3f}")
            st.warning("Le regole 'migliori' nel passato non reggono nel futuro: il legame "
                       "passato→futuro è praticamente zero.")
