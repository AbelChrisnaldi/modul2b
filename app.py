import time

import pandas as pd
import streamlit as st
from groq import Groq

from classifier import NAMA_FITUR, classify, load_config

st.set_page_config(page_title="Status Akademik (LLM)", page_icon="🤖")

# ---------- 1. Muat konfigurasi hasil evaluasi & siapkan client ----------
cfg = load_config("config.json")
FITUR = list(NAMA_FITUR)

try:
    API_KEY = st.secrets["GROQ_API_KEY"]
except Exception:                       # file secrets belum ada atau key belum diisi
    st.error("GROQ_API_KEY belum diatur. Isi di .streamlit/secrets.toml (lokal) "
             "atau menu Secrets (Streamlit Community Cloud).")
    st.stop()


@st.cache_resource
def get_client():
    return Groq(api_key=API_KEY)


client = get_client()


@st.cache_data(show_spinner=False)
def prediksi(ipk, kehadiran, jam_belajar, tugas_terkumpul, nilai_uts, aktif_organisasi):
    # hasil disimpan di cache: data yang sama tidak memanggil API dua kali (hemat kuota)
    data = {"ipk": ipk, "kehadiran": kehadiran, "jam_belajar": jam_belajar,
            "tugas_terkumpul": tugas_terkumpul, "nilai_uts": nilai_uts,
            "aktif_organisasi": aktif_organisasi}
    return classify(client, cfg, data)


# ---------- 2. Sidebar: informasi konfigurasi yang dikunci ----------
ev = cfg["evaluasi"]
with st.sidebar:
    st.header("Tentang Model")
    st.write(f"LLM: `{cfg['model']}` (Groq)")
    st.write(f"Few-shot: {len(cfg['few_shot'])} contoh")
    st.metric("F1 macro LLM (data uji)", f"{ev['f1_macro']:.3f}")
    if "f1_dnn_pembanding" in ev:
        st.metric("F1 macro DNN Modul 1A (pembanding)", f"{ev['f1_dnn_pembanding']:.3f}")
    st.caption(f"Dievaluasi pada {ev['n']} data uji yang sama dengan Modul 1A, "
               f"tanggal {ev['tanggal']}")
    with st.expander("Lihat system prompt"):
        st.code(cfg["system_prompt"], language=None)

st.title("🤖 Prediksi Status Akademik dengan LLM")
st.write("Masukkan data mahasiswa untuk memprediksi status: **Aman**, **Waspada**, atau **Berisiko**.")

tab1, tab2 = st.tabs(["Prediksi Satu Mahasiswa", "Prediksi dari File CSV"])

# ---------- 3. Prediksi satu data ----------
with tab1:
    with st.form("form_prediksi"):
        col1, col2 = st.columns(2)
        with col1:
            ipk = st.number_input("IPK", min_value=0.0, max_value=4.0, value=3.0, step=0.01)
            kehadiran = st.slider("Kehadiran (%)", 0, 100, 80)
            jam_belajar = st.slider("Jam belajar mandiri per minggu", 0, 40, 8)
        with col2:
            tugas = st.slider("Tugas terkumpul (%)", 0, 100, 80)
            uts = st.slider("Nilai UTS", 0, 100, 70)
            organisasi = st.radio("Aktif organisasi?", ["Ya", "Tidak"], horizontal=True)
        kirim = st.form_submit_button("Prediksi")

    if kirim:
        with st.spinner("Menghubungi Groq..."):
            hasil = prediksi(ipk, kehadiran, jam_belajar, tugas, uts, organisasi)
        tampil = {"Aman": st.success, "Waspada": st.warning, "Berisiko": st.error}
        tampil.get(hasil, st.info)(f"Prediksi status: **{hasil}**")

# ---------- 4. Prediksi banyak data dari CSV ----------
with tab2:
    st.write("Unggah CSV dengan kolom: " + ", ".join(f"`{c}`" for c in FITUR)
             + " (maksimal 30 baris agar hemat kuota).")
    file = st.file_uploader("Pilih file CSV", type="csv")
    if file is not None:
        df = pd.read_csv(file)
        kurang = [c for c in FITUR if c not in df.columns]
        if kurang:
            st.error(f"Kolom berikut tidak ditemukan: {', '.join(kurang)}")
        else:
            df = df.head(30)
            if st.button("Prediksi semua"):
                bar = st.progress(0.0)
                hasil = []
                for i, (_, b) in enumerate(df.iterrows(), start=1):
                    hasil.append(prediksi(float(b["ipk"]), int(b["kehadiran"]), int(b["jam_belajar"]),
                                          int(b["tugas_terkumpul"]), int(b["nilai_uts"]),
                                          str(b["aktif_organisasi"])))
                    bar.progress(i / len(df))
                    time.sleep(3)               # jaga batas request per menit
                df["prediksi_status"] = hasil
                st.dataframe(df)
                st.write(df["prediksi_status"].value_counts())
                st.download_button("Unduh hasil prediksi", df.to_csv(index=False),
                                   file_name="hasil_prediksi_llm.csv", mime="text/csv")