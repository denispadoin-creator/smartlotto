#!/usr/bin/env python3
"""
Interfaccia Streamlit di SmartLotto.

Avvio:
    streamlit run app.py

Mostra: riepilogo archivio, ritardi/frequenze per ruota, sequenze e numerologia,
fenomeni celesti del giorno, e — soprattutto — il backtesting walk-forward che
confronta ogni strategia con la giocata casuale, con verdetto statistico.
"""

from __future__ import annotations
import datetime as dt
import numpy as np
import pandas as pd
import streamlit as st

import datetime as _dt
from smartlotto import ingestion, statistiche, sequenze, numerologia, astro, predittore, tracker
from smartlotto import aggiornamento, regole, strategie as S, backtest as B, formule, RUOTE, RUOTE_NOMI

PERCORSO_PREV = "previsioni.json"


def chip(numeri, azzeccati=None):
    """Rende una fila di numeri come 'chip'; se azzeccato, in verde."""
    azz = set(azzeccati or [])
    out = []
    for n in numeri:
        verde = n in azz
        bg = "#1f9d55" if verde else "rgba(120,120,120,.15)"
        col = "#fff" if verde else "inherit"
        bordo = "#1f9d55" if verde else "rgba(120,120,120,.3)"
        out.append(f"<span style='display:inline-block;min-width:34px;text-align:center;"
                   f"padding:5px 8px;margin:3px;border-radius:8px;font-weight:600;"
                   f"font-variant-numeric:tabular-nums;background:{bg};color:{col};"
                   f"border:1px solid {bordo}'>{n:02d}</span>")
    return "".join(out)
from smartlotto.backtest import P_ESTRATTO

st.set_page_config(page_title="SmartLotto — Analisi Estrazioni", layout="wide")

PERCORSO_DEFAULT = "data/archivio_lotto.txt"


@st.cache_data(show_spinner=True)
def carica(percorso: str) -> pd.DataFrame:
    return ingestion.carica_archivio(percorso)


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
    return {n: {"giocate": r.giocate, "successi": r.successi,
                "tasso": r.tasso_per_numero, "tasso_vinc": r.tasso_vincenti,
                **r.statistica()} for n, r in agg.items()}


@st.cache_data(show_spinner=True)
def frequenze_per_ruota(percorso: str) -> pd.DataFrame:
    """Tabella numero (1-90) x ruota con la frequenza % di uscita come estratto."""
    df = carica(percorso)
    dati = {}
    for r in RUOTE:
        sub = ingestion.matrice_ruota(df, r)
        P = ingestion.matrice_presenza(sub)
        n = len(sub)
        fr = P[:, 1:].sum(axis=0) / n * 100 if n else np.zeros(90)
        dati[RUOTE_NOMI[r]] = fr
    out = pd.DataFrame(dati, index=range(1, 91))
    out.index.name = "numero"
    return out


@st.cache_data(show_spinner=True)
def esegui_formule(percorso: str, taglio: str) -> dict:
    df = carica(percorso)
    return formule.cerca_formule(df, taglio, top=12)


# ---------------------------------------------------------------------------
st.title("🎱 SmartLotto — Analisi delle estrazioni del Lotto")
st.caption("Analisi matematica, numerologica e olistica con backtesting rigoroso. "
           "Le estrazioni sono eventi indipendenti: questo strumento serve a "
           "verificare onestamente se una strategia batte il caso, non a prevedere i numeri.")

percorso = st.sidebar.text_input("Archivio estrazioni", PERCORSO_DEFAULT)
st.sidebar.divider()
st.sidebar.subheader("Aggiornamento")
if st.sidebar.button("⬇️ Scarica estrazioni mancanti"):
    with st.spinner("Scarico le estrazioni nuove..."):
        try:
            n = aggiornamento.aggiorna_archivio(percorso)
            carica.clear()  # invalida la cache così l'archivio viene riletto
            # verifica le previsioni in attesa contro i nuovi dati
            try:
                tracker.verifica_previsioni(PERCORSO_PREV, carica(percorso))
            except Exception:
                pass
            if n:
                st.sidebar.success(f"{n} nuove estrazioni aggiunte.")
            else:
                st.sidebar.info("Archivio già aggiornato.")
        except Exception as e:
            st.sidebar.error(f"Aggiornamento non riuscito: {e}")
ult = aggiornamento.ultima_data_archivio(percorso)
if ult:
    st.sidebar.caption(f"Ultima estrazione in archivio: {ult.strftime('%d/%m/%Y')}")

# Auto-aggiornamento all'apertura (una volta per sessione): online scarica da solo
# le estrazioni nuove, così l'app è sempre aggiornata senza toccare nulla.
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

rip = ingestion.riepilogo(df)
c1, c2, c3, c4 = st.columns(4)
c1.metric("Estrazioni (data,ruota)", f"{rip['estrazioni_totali_righe']:,}")
c2.metric("Date distinte", f"{rip['date_distinte']:,}")
c3.metric("Dal", rip["prima_data"])
c4.metric("Al", rip["ultima_data"])

tab0, tab1, tab5, tab2, tab3, tab4 = st.tabs(
    ["🎯 Previsione", "📊 Ritardi & Frequenze", "📈 Grafici & Probabilità",
     "🔢 Sequenze & Numerologia", "🌙 Cielo del giorno", "🧪 Backtesting & Formule"])

# --- Tab 0: previsione ---
with tab0:
    ultima_arch = df["data"].max().date()
    data_prossima = tracker.prossimo_giorno_estrazione(ultima_arch)

    st.subheader("Cinquine per la prossima estrazione")
    d_prev = st.date_input("Data dell'estrazione", data_prossima, key="prev")
    if st.button("Genera e salva le cinquine"):
        res = predittore.previsione_tutte_le_ruote(df, d_prev)
        tracker.registra_previsione(PERCORSO_PREV, d_prev, res["ruote"])
        st.success(f"Previsione per il {d_prev.strftime('%d/%m/%Y')} salvata. "
                   "Dopo l'estrazione, aggiorna l'archivio: i numeri usciti diventeranno verdi.")

    # Mostra la previsione in attesa (non ancora verificata)
    attesa = tracker.ultima_in_attesa(PERCORSO_PREV)
    if attesa:
        st.markdown(f"**Previsione in attesa — estrazione del "
                    f"{_dt.date.fromisoformat(attesa['data_estrazione']).strftime('%d/%m/%Y')}**")
        for ruota, nums in attesa["ruote"].items():
            st.markdown(f"<b>{ruota}</b> {chip(nums)}", unsafe_allow_html=True)

    st.divider()
    st.subheader("✅ Verifica dell'ultima previsione")
    verif = tracker.ultima_verificata(PERCORSO_PREV)
    if verif:
        d = _dt.date.fromisoformat(verif["data_estrazione"]).strftime("%d/%m/%Y")
        tot = sum(e["n_azzeccati"] for e in verif["esiti"].values())
        st.markdown(f"Estrazione del **{d}** — numeri azzeccati (in verde): **{tot}**")
        for ruota, e in verif["esiti"].items():
            st.markdown(
                f"<b>{ruota}</b> {chip(e['previsti'], e['azzeccati'])}"
                f"<span style='color:#888;margin-left:8px'>· estratti: "
                f"{'  '.join(f'{x:02d}' for x in e['estratti'])}</span>",
                unsafe_allow_html=True)
    else:
        st.info("Ancora nessuna previsione verificata. Genera una cinquina, poi — "
                "dopo l'estrazione — aggiorna l'archivio dalla barra laterale.")

    st.warning("Questi 5 numeri hanno la stessa probabilità di 5 numeri a caso: "
               "il criterio (ritardo + numeri freddi + numerologia + cielo) è trasparente "
               "ma il backtesting mostra che non batte il caso. Gioca con responsabilità.")

# --- Tab 1: ritardi & frequenze ---
with tab1:
    ruota = st.selectbox("Ruota", RUOTE, format_func=lambda r: RUOTE_NOMI[r])
    sub = ingestion.matrice_ruota(df, ruota)
    P = ingestion.matrice_presenza(sub)
    col1, col2 = st.columns(2)
    with col1:
        st.subheader("Numeri più ritardatari")
        rit = statistiche.classifica_ritardatari(P, top=15)
        st.dataframe(pd.DataFrame(rit, columns=["numero", "ritardo (estrazioni)"]),
                     hide_index=True, use_container_width=True)
    with col2:
        st.subheader("Numeri più frequenti")
        fr = statistiche.classifica_frequenti(P, top=15, caldi=True)
        st.dataframe(pd.DataFrame(fr, columns=["numero", "uscite totali"]),
                     hide_index=True, use_container_width=True)
    st.subheader("Frequenza di ogni numero su questa ruota")
    freq = P[:, 1:].sum(axis=0) / max(len(sub), 1) * 100
    st.bar_chart(pd.DataFrame({"frequenza %": freq}, index=range(1, 91)),
                 color="#E3A44A", height=260)
    st.caption(f"Linea di riferimento del caso: {P_ESTRATTO*100:.3f}%. "
               "Le barre oscillano intorno a quel valore: nessun numero è davvero privilegiato.")
    st.info("Nota: un numero molto ritardatario non ha maggiore probabilità di "
            "uscire alla prossima estrazione. La probabilità resta 5/90 ogni volta.")

# --- Tab Grafici & Probabilità ---
with tab5:
    st.subheader("Probabilità delle sorti")
    from math import comb
    def prob_sorte(k, s):
        tot = comb(90, 5)
        fav = sum(comb(k, i) * comb(90 - k, 5 - i) for i in range(s, min(k, 5) + 1))
        return fav / tot
    base = [("Estratto", 1, 1), ("Ambo", 2, 2), ("Terno", 3, 3),
            ("Quaterna", 4, 4), ("Cinquina", 5, 5)]
    st.dataframe(pd.DataFrame([
        {"sorte": nm, "giochi": k, "probabilità %": round(prob_sorte(k, s) * 100, 6),
         "1 su…": round(1 / prob_sorte(k, s))} for nm, k, s in base]),
        hide_index=True, use_container_width=True)

    st.divider()
    st.subheader("I miei numeri — probabilità e confronto fra ruote")
    testo = st.text_input("Inserisci da 1 a 5 numeri separati da spazio", "7 25 45 67 82")
    try:
        miei = sorted({int(x) for x in testo.split() if 1 <= int(x) <= 90})[:5]
    except ValueError:
        miei = []
    if miei:
        st.write("Numeri:", "  ".join(f"{n:02d}" for n in miei))
        righe = []
        for nm, _, s in base:
            if s <= len(miei):
                p = prob_sorte(len(miei), s)
                righe.append({"sorte (almeno)": nm, "probabilità %": round(p * 100, 5),
                              "1 su…": round(1 / p)})
        st.dataframe(pd.DataFrame(righe), hide_index=True, use_container_width=True)

        st.markdown("**Quante volte i tuoi numeri sono usciti, ruota per ruota (%)**")
        ftab = frequenze_per_ruota(percorso)
        comp = ftab.loc[miei]                      # righe = numeri, colonne = ruote
        st.bar_chart(comp.T, height=300)           # asse x = ruote, serie = numeri
        st.caption("Ogni serie è uno dei tuoi numeri; ogni gruppo una ruota. "
                   "Le altezze si equivalgono: nessuna ruota è più generosa di un'altra.")
    else:
        st.warning("Scrivi da 1 a 5 numeri validi (1-90).")

# --- Tab 2: sequenze & numerologia ---
with tab2:
    col1, col2 = st.columns(2)
    with col1:
        st.subheader("Sequenze matematiche (1..90)")
        st.write("**Fibonacci**:", sequenze.fibonacci_fino_a(90))
        st.write("**Primi**:", sequenze.numeri_primi(90))
        st.write("**Quadrati**:", sequenze.quadrati(90))
        st.write("**Triangolari**:", sequenze.triangolari(90))
    with col2:
        st.subheader("Numerologia di una data")
        d = st.date_input("Data", dt.date.today())
        st.write("**Somma cabalistica**:", numerologia.somma_cabalistica_data(d))
        st.write("**Ciclo pitagorico (1..9)**:", numerologia.ciclo_pitagorico(d))
        st.write("**Numeri numerologici del giorno**:", numerologia.numeri_da_data(d))

# --- Tab 3: cielo ---
with tab3:
    d = st.date_input("Data del cielo", dt.date.today(), key="cielo")
    fase = astro.fase_lunare(d)
    st.write(f"**Fase lunare**: {fase['fase']} — illuminazione {fase['illuminazione']*100:.0f}%, "
             f"età {fase['eta_giorni']:.1f} giorni")
    st.write(f"**Segno solare**: {astro.segno_solare(d)}")
    pos = astro.posizioni_pianeti(d)
    st.dataframe(pd.DataFrame([
        {"pianeta": k, "segno": v["segno"], "gradi": round(v["longitudine"], 1)}
        for k, v in pos.items()]), hide_index=True, use_container_width=True)
    st.write("**Numeri derivati dal cielo**:", astro.numeri_da_cielo(d))

# --- Tab 4: backtest ---
with tab4:
    st.subheader("Backtesting walk-forward (finestra espandente)")
    st.write("Ogni previsione usa solo le estrazioni precedenti; si avanza nel "
             "tempo dalle prime estrazioni ad oggi. La soglia da battere è "
             f"**{P_ESTRATTO*100:.3f}%** (giocata a caso).")
    warmup = st.slider("Warmup (estrazioni iniziali di rodaggio)", 100, 1000, 300, 100)

    st.markdown("**Scegli quali regole testare** (per categoria)")
    cat_map = regole.regole_per_categoria()
    scelte = []
    cols = st.columns(3)
    for col, (cat, elenco) in zip(cols, cat_map.items()):
        with col:
            nomi = {nome: chiave for chiave, nome in elenco}
            sel = st.multiselect(cat, list(nomi.keys()),
                                 default=list(nomi.keys()) if cat == "Matematiche" else [],
                                 key="ms_" + cat)
            scelte += [nomi[n] for n in sel]
    st.caption(f"{len(scelte)} regole selezionate."
               + ("" if scelte else " Vuoto = tutte le regole."))

    if st.button("Esegui backtesting su tutte le ruote"):
        res = esegui_backtest(percorso, warmup, tuple(scelte))
        righe = []
        for n, r in sorted(res.items(), key=lambda kv: kv[1]["z"], reverse=True):
            righe.append({
                "strategia": n, "giocate": r["giocate"], "successi": r["successi"],
                "tasso %": round(r["tasso"] * 100, 4),
                "atteso % (caso)": round(P_ESTRATTO * 100, 4),
                "z": round(r["z"], 2), "p-value": round(r["p_value"], 4),
                "verdetto": "come il caso" if r["p_value"] > 0.01 else "da verificare",
            })
        st.dataframe(pd.DataFrame(righe), hide_index=True, use_container_width=True)
        st.success("Nessun p-value regge alla correzione per confronti multipli "
                   "(Bonferroni) quando è vicino a 0.05: è il comportamento atteso "
                   "da dati casuali.")

    st.divider()
    st.subheader("Motore di ricerca formule + validazione sul futuro")
    taglio = st.text_input("Data di separazione passato/futuro", "2010-01-01")
    if st.button("Cerca formule e validale"):
        ff = esegui_formule(percorso, taglio)
        c1, c2, c3 = st.columns(3)
        c1.metric("Regole esaminate", ff["n_regole"])
        c2.metric("Signif. a p<0.05", ff["significative_p05"],
                  help=f"Attese per puro caso: ~{ff['attesi_falsi_positivi_p05']:.0f}")
        c3.metric("Corr. passato→futuro", f"{ff['correlazione_in_out']:.3f}")
        st.dataframe(pd.DataFrame(ff["migliori_in_sample"])[
            ["regola", "tasso_in", "tasso_out"]].assign(
            tasso_in=lambda x: (x.tasso_in * 100).round(3),
            tasso_out=lambda x: (x.tasso_out * 100).round(3)),
            hide_index=True, use_container_width=True)
        st.warning("Le regole 'migliori' nel passato non reggono nel futuro: è "
                   "l'overfitting. Cercando tante regole se ne trova sempre qualcuna "
                   "buona per caso, ma il caso non si ripete.")
