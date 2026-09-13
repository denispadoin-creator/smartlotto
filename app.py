#!/usr/bin/env python3
"""
Interfaccia Streamlit di SmartLotto — versione per cellulare.

Avvio:
    streamlit run app.py

Schermate: Previsione (con scelta delle regole e probabilità delle sorti),
Osservati (misura delle previsioni nel tempo), Numeri (ritardi/frequenze +
i miei numeri + regole affini), Guida (come leggere le percentuali), Cielo,
Formule (backtesting e validazione sul futuro, in parole semplici).
"""

from __future__ import annotations
import datetime as dt
import datetime as _dt
import json
import os
from math import comb

import numpy as np
import pandas as pd
import streamlit as st

from smartlotto import ingestion, statistiche, sequenze, numerologia, astro, predittore, tracker
from smartlotto import aggiornamento, regole, backtest as B, formule, RUOTE, RUOTE_NOMI
from smartlotto.backtest import P_ESTRATTO
from smartlotto.strategie import StatoIncrementale

PERCORSO_PREV = "previsioni.json"
PERCORSO_OSS = "osservati.json"
PERCORSO_DEFAULT = "data/archivio_lotto.txt"

st.set_page_config(page_title="SmartLotto", layout="wide", page_icon="🎱")

# --- Stile: header compatto, tab grandi e leggibili sul telefono ---
st.markdown("""
<style>
.block-container{padding-top:1.1rem;padding-bottom:2rem}
.hero{font-size:1.45rem;font-weight:700;margin:0;line-height:1.2}
.hero-sub{color:#AAB6C4;font-size:.9rem;margin:.25rem 0 .2rem}
/* Tab piu grandi, con etichette ben visibili */
button[data-baseweb="tab"]{padding:10px 16px !important}
button[data-baseweb="tab"] p{font-size:1.02rem !important;font-weight:600 !important}
div[data-baseweb="tab-list"]{gap:4px;overflow-x:auto}
/* Chip numeri */
.chip{display:inline-block;min-width:38px;text-align:center;padding:7px 9px;margin:3px;
  border-radius:9px;font-weight:700;font-variant-numeric:tabular-nums;font-size:1rem;
  background:rgba(120,120,120,.15);border:1px solid rgba(120,120,120,.30)}
.chip.ok{background:#1f9d55;color:#fff;border-color:#1f9d55}
.chip.big{background:#E3A44A;color:#20160a;border-color:#E3A44A}
.ruota-lbl{font-weight:700;color:#E3A44A;min-width:74px;display:inline-block}
</style>
""", unsafe_allow_html=True)


# ============================ funzioni di supporto ============================

@st.cache_data(show_spinner=True)
def carica(percorso: str) -> pd.DataFrame:
    return ingestion.carica_archivio(percorso)


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


def prob_sorte(k: int, s: int) -> float:
    """Probabilità di fare la sorte s (almeno) giocando k numeri su una ruota."""
    tot = comb(90, 5)
    fav = sum(comb(k, i) * comb(90 - k, 5 - i) for i in range(s, min(k, 5) + 1))
    return fav / tot


SORTI = [("Estratto", 1), ("Ambo", 2), ("Terno", 3), ("Quaterna", 4), ("Cinquina", 5)]


def tabella_sorti(quanti: int) -> pd.DataFrame:
    """Tabella probabilità estratto/ambo/.../cinquina giocando `quanti` numeri."""
    righe = []
    for nome, s in SORTI:
        if s > quanti:
            continue
        p = prob_sorte(quanti, s)
        righe.append({"Sorte (almeno)": nome, "Probabilità": f"{p*100:.4f}%".rstrip("0").rstrip("."),
                      "1 su…": f"{round(1/p):,}".replace(",", ".")})
    return pd.DataFrame(righe)


def chip_html(numeri, azzeccati=None, big=False):
    azz = set(azzeccati or [])
    cls_extra = " big" if big else ""
    out = []
    for n in numeri:
        cls = "chip ok" if n in azz else "chip" + cls_extra
        out.append(f"<span class='{cls}'>{int(n):02d}</span>")
    return "".join(out)


def _stato_ruota(sub) -> StatoIncrementale:
    stato = StatoIncrementale()
    for numeri in sub["numeri"]:
        stato.aggiorna(numeri)
    return stato


def affinita_regole(df, ruota_code, miei, data, top=8):
    """Quali regole 'propongono' i miei numeri, su una ruota. Ordina per quanti
    dei miei numeri la regola indica (anche uno solo conta)."""
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


def carica_osservati():
    if not os.path.exists(PERCORSO_OSS):
        return []
    try:
        with open(PERCORSO_OSS, "r", encoding="utf-8") as fh:
            return json.load(fh)
    except Exception:
        return []


def salva_osservati(voci):
    with open(PERCORSO_OSS, "w", encoding="utf-8") as fh:
        json.dump(voci, fh, ensure_ascii=False, indent=2)


def misura_osservato(df, ruota_code, numeri, da_data):
    """Scorre le estrazioni dalla data di inserimento in poi e registra ogni
    volta che escono numeri della serie. Ritorna (n_controllate, lista_uscite)."""
    numeri = set(numeri)
    if ruota_code == "TUTTE":
        ruote = RUOTE
    else:
        ruote = [ruota_code]
    uscite = []
    n_contr = 0
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


# ============================ intestazione + sidebar ============================

st.markdown("<div class='hero'>🎱 SmartLotto</div>"
            "<div class='hero-sub'>Analisi delle estrazioni del Lotto — con verifica onesta sui dati</div>",
            unsafe_allow_html=True)

percorso = st.sidebar.text_input("Archivio estrazioni", PERCORSO_DEFAULT)
st.sidebar.subheader("Aggiornamento")
if st.sidebar.button("⬇️ Scarica estrazioni mancanti"):
    with st.spinner("Scarico le estrazioni nuove..."):
        try:
            n = aggiornamento.aggiorna_archivio(percorso)
            carica.clear()
            try:
                tracker.verifica_previsioni(PERCORSO_PREV, carica(percorso))
            except Exception:
                pass
            st.sidebar.success(f"{n} nuove estrazioni aggiunte." if n else "Archivio già aggiornato.")
        except Exception as e:
            st.sidebar.error(f"Aggiornamento non riuscito: {e}")
ult = aggiornamento.ultima_data_archivio(percorso)
if ult:
    st.sidebar.caption(f"Ultima estrazione: {ult.strftime('%d/%m/%Y')}")

if "auto_agg" not in st.session_state:
    try:
        aggiornamento.aggiorna_archivio(percorso)
    except Exception:
        pass
    st.session_state["auto_agg"] = True

try:
    df = carica(percorso)
except Exception as e:
    st.error(f"Impossibile caricare l'archivio: {e}")
    st.stop()


def selezione_regole(prefix: str) -> list:
    """Scelta delle regole con checkbox e pulsanti Tutte/Nessuna. Ritorna le chiavi scelte."""
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


tab_prev, tab_oss, tab_num, tab_guida, tab_cielo, tab_form = st.tabs(
    ["🎯 Previsione", "👁️ Osservati", "📊 Numeri", "📖 Guida", "🌙 Cielo", "🔬 Formule"])


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
            nums = predittore.cinquina_da_regole(sub, d_prev, chiavi or None)
            risultati[RUOTE_NOMI[r]] = nums
        st.session_state["pv_risultati"] = risultati
        st.session_state["pv_data"] = d_prev.isoformat()

    risultati = st.session_state.get("pv_risultati")
    if risultati:
        st.markdown(f"**Numeri generati per il {_dt.date.fromisoformat(st.session_state['pv_data']).strftime('%d/%m/%Y')}**")
        for ruota, nums in risultati.items():
            st.markdown(f"<span class='ruota-lbl'>{ruota}</span> {chip_html(nums, big=True)}",
                        unsafe_allow_html=True)

        st.markdown("---")
        st.markdown("**🎟️ Se giochi questi 5 numeri — probabilità delle sorti**")
        st.dataframe(tabella_sorti(5), hide_index=True, use_container_width=True)
        st.caption("Le probabilità dipendono solo da quanti numeri giochi (5 su 90): "
                   "sono uguali per qualsiasi cinquina. L'estratto (almeno un numero) "
                   "capita in media 1 volta ogni ~4 estrazioni; il terno 1 su ~1.200.")

        if st.button("💾 Salva questa previsione (per verificarla dopo)", use_container_width=True):
            d_obj = _dt.date.fromisoformat(st.session_state["pv_data"])
            tracker.registra_previsione(PERCORSO_PREV, d_obj, risultati)
            st.success("Salvata. Dopo l'estrazione aggiorna l'archivio: i numeri usciti "
                       "diventeranno verdi nella scheda Osservati.")

    st.warning("Onestà: questi numeri hanno la stessa probabilità di 5 numeri a caso. "
               "Il criterio è trasparente, ma il backtesting (scheda Formule) mostra che "
               "non batte il caso. Gioca con responsabilità.")


# ============================ TAB OSSERVATI ============================
with tab_oss:
    st.subheader("👁️ Osservati — misura delle previsioni")
    st.caption("Salva una serie di numeri: l'app controlla a ogni estrazione futura "
               "se escono e dopo quante estrazioni. In verde i numeri realmente usciti.")

    # --- verifica dell'ultima previsione salvata in Previsione ---
    verif = tracker.ultima_verificata(PERCORSO_PREV)
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
            salva_osservati(voci)
            st.success(f"Aggiunta: {' '.join(f'{n:02d}' for n in nums)}")
        else:
            st.warning("Inserisci numeri validi (1-90).")

    st.markdown("---")
    voci = carica_osservati()
    if not voci:
        st.info("Nessuna serie in osservazione. Aggiungine una qui sopra.")
    for v in reversed(voci):
        rn = "Tutte le ruote" if v["ruota"] == "TUTTE" else RUOTE_NOMI[v["ruota"]]
        da = _dt.date.fromisoformat(v["da"])
        n_contr, uscite = misura_osservato(df, v["ruota"], v["numeri"], da)
        tutti_usciti = sorted({n for u in uscite for n in u["azzeccati"]})
        best = max((u["n"] for u in uscite), default=0)
        sorte_nomi = {1: "estratto", 2: "ambo", 3: "terno", 4: "quaterna", 5: "cinquina"}
        with st.container(border=True):
            st.markdown(f"<span class='ruota-lbl'>{rn}</span> {chip_html(v['numeri'], tutti_usciti)}",
                        unsafe_allow_html=True)
            riepilogo = (f"Dal {da.strftime('%d/%m/%Y')} · {n_contr} estrazioni controllate · "
                         f"{len(uscite)} volte con almeno un numero")
            if best >= 2:
                riepilogo += f" · **miglior colpo: {sorte_nomi.get(min(best,5))}**"
            st.caption(riepilogo)
            if uscite:
                ult3 = uscite[-3:]
                for u in reversed(ult3):
                    st.markdown(f"<div style='font-size:.85rem;color:#4CBF8B'>"
                                f"{u['data'].strftime('%d/%m/%Y')} ({u['ruota']}): "
                                f"{' '.join(f'{n:02d}' for n in u['azzeccati'])} "
                                f"— {sorte_nomi.get(min(u['n'],5))}, dopo {u['dopo']} estrazioni</div>",
                                unsafe_allow_html=True)
            if st.button("🗑️ Rimuovi", key="del_" + v["id"]):
                salva_osservati([x for x in carica_osservati() if x["id"] != v["id"]])
                st.rerun()


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
    testo = st.text_input("Da 1 a 5 numeri separati da spazio", "7 25 45 67 82", key="num_miei")
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
        st.caption("Ogni serie è uno dei tuoi numeri. Le altezze si equivalgono: "
                   "nessuna ruota è più generosa.")

        st.markdown(f"**🔗 Regole più affini ai tuoi numeri — ruota di {RUOTE_NOMI[ruota]}**")
        aff = affinita_regole(df, ruota, miei, tracker.prossimo_giorno_estrazione(df["data"].max().date()))
        if aff:
            st.dataframe(pd.DataFrame([
                {"regola": nome, "tuoi numeri che indica": " ".join(f"{n:02d}" for n in inter),
                 "quanti indica in tutto": tot} for nome, inter, tot in aff]),
                hide_index=True, use_container_width=True)
            st.caption("Sono le regole che, applicate a questa ruota, propongono i numeri "
                       "che hai scelto. 'Affine' non vuol dire 'più probabile': è solo il "
                       "criterio che ti ha portato a quei numeri.")
        else:
            st.info("Nessuna regola propone questi numeri su questa ruota.")


# ============================ TAB GUIDA ============================
with tab_guida:
    st.subheader("📖 Come leggere le percentuali")
    st.markdown("""
Una **percentuale** dice quanto è probabile una cosa, su una scala da **0% (mai)**
a **100% (sempre, certo)**. Il famoso **5,5%** del Lotto **non** vuol dire "esce di sicuro".
""")
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
    st.info("**5,5% = su 90 palline ne escono 5.** Il tuo numero è una casella qualsiasi: "
            "ha 5 possibilità su 90. Non dipende da quanto è in ritardo o da quale regola scegli.")
    st.markdown("**Più piccola è la percentuale, più raro è l'evento** — e più devi aspettare. "
                "Quindi sì: 2,5% è meno probabile di 5,5%.")

    st.markdown("---")
    st.subheader("🚦 Affidabilità: quando un numero è «verde»?")
    st.markdown("""
L'app misura ogni criterio su **oltre 150 anni** di estrazioni. Un numero o una regola
diventa **verde** solo se batte il caso in modo solido e ripetuto — non è mai successo
finora. Se vedi tutto **neutro/grigio**, è la verità: quei numeri valgono come numeri a caso.
Meglio saperlo che illudersi.
""")


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
    st.markdown("""
Qui l'app **si mette alla prova con onestà**. Due strumenti:
""")

    st.markdown("### 1) Backtesting delle regole")
    st.caption("Ogni regola viene provata «all'epoca»: usa solo le estrazioni precedenti "
               "e avanza nel tempo, dalle prime estrazioni a oggi. Deve battere la giocata "
               f"a caso ({P_ESTRATTO*100:.3f}%).")
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
            righe.append({
                "regola": nome,
                "riuscita": f"{r['tasso']*100:.2f}%",
                "vs caso": f"{diff:+.2f}%",
                "esito": "🟢 batte il caso" if r["p_value"] < 0.001 and diff > 0 else "⚪ come il caso",
            })
        st.dataframe(pd.DataFrame(righe), hide_index=True, use_container_width=True)
        st.success("Nessuna regola resta 🟢: tutte vanno «come il caso». "
                   "È il risultato atteso — le estrazioni sono indipendenti.")

    st.markdown("---")
    st.markdown("### 2) Analizza una serie di numeri sul futuro")
    st.markdown("""
**A cosa serve la data di separazione?** Divide la storia in due:
il **passato** serve a scegliere/valutare i numeri, il **futuro** (dopo quella data) serve
a controllare se davvero continuano a uscire di più. Se un numero è «bravo» solo nel
passato ma non nel futuro, era solo fortuna (si chiama *overfitting*).
""")
    miei_def = " ".join(f"{n:02d}" for n in st.session_state.get("miei_numeri", [7, 25, 45, 67, 82]))
    serie_txt = st.text_input("Numeri da analizzare (li trovi anche nella scheda Numeri)",
                              miei_def, key="form_serie")
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
            df_in = df[df["data"] < tcut]
            df_out = df[df["data"] >= tcut]

            def tasso(dd):
                if len(dd) == 0:
                    return None
                usc = sum(1 for s in dd["numeri"] for n in serie if n in s)
                return usc / (len(dd) * 5)  # per posizione estratta

            t_in, t_out = tasso(df_in), tasso(df_out)
            base = P_ESTRATTO
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
                           "i numeri non hanno alcun vantaggio reale — qualunque cosa dica il passato.")

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
            st.warning("Le regole 'migliori' nel passato non reggono nel futuro. "
                       "Cercando tante regole se ne trova sempre qualcuna buona per caso, "
                       "ma il caso non si ripete: il legame passato→futuro è praticamente zero.")
