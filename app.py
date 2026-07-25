# ============================================================
# APLIKASI WEB KLASIFIKASI FASHION-MNIST (Streamlit)
# ============================================================
# Menerima input gambar dari pengguna, melakukan preprocessing
# yang SAMA PERSIS dengan proses training (28x28, grayscale, /255),
# lalu menampilkan hasil prediksi model CNN beserta tingkat keyakinan.
#
# Menjalankan secara lokal:
#   pip install -r requirements.txt
#   streamlit run app.py
# ============================================================

import os
import numpy as np
import streamlit as st
from PIL import Image, ImageOps
import tensorflow as tf
from tensorflow import keras

# ------------------------------------------------------------
# 1. KONFIGURASI DASAR
# ------------------------------------------------------------
# Kelas HARUS berurutan sama seperti saat training (label 0..9).
CLASS_NAMES = [
    "T-shirt/top", "Trouser", "Pullover", "Dress", "Coat",
    "Sandal", "Shirt", "Sneaker", "Bag", "Ankle boot",
]

# Terjemahan Indonesia sekadar untuk tampilan (opsional).
CLASS_NAMES_ID = [
    "Kaos/atasan", "Celana panjang", "Sweter", "Dress", "Mantel",
    "Sandal", "Kemeja", "Sepatu kets", "Tas", "Sepatu bot",
]

# Kandidat lokasi file model. Taruh file .keras di folder yang sama
# dengan app.py, atau di subfolder "model/".
MODEL_CANDIDATES = [
    "cnn_fashion_mnist_optimized_v2.keras",
    "best_cnn_fashion_mnist_v2.keras",
    os.path.join("model", "cnn_fashion_mnist_optimized_v2.keras"),
    os.path.join("model", "best_cnn_fashion_mnist_v2.keras"),
]

st.set_page_config(
    page_title="Klasifikasi Fashion-MNIST (CNN)",
    page_icon="👕",
    layout="centered",
)


# ------------------------------------------------------------
# 2. MEMUAT MODEL (dengan cache agar tidak reload tiap interaksi)
# ------------------------------------------------------------
@st.cache_resource(show_spinner="Memuat model CNN...")
def load_cnn_model():
    for path in MODEL_CANDIDATES:
        if os.path.exists(path):
            model = keras.models.load_model(path, compile=False)
            return model, path
    return None, None


model, model_path = load_cnn_model()


# ------------------------------------------------------------
# 3. FUNGSI PREPROCESSING
# ------------------------------------------------------------
# Fashion-MNIST: objek TERANG di latar HITAM, ukuran 28x28 grayscale,
# nilai piksel dinormalisasi ke rentang [0, 1].
# Foto pengguna umumnya sebaliknya (objek gelap di latar terang),
# sehingga sering perlu di-INVERT agar cocok dengan konvensi dataset.
def preprocess_image(pil_image, invert_mode="auto"):
    # 1) Perbaiki orientasi EXIF (foto HP sering ter-rotate).
    pil_image = ImageOps.exif_transpose(pil_image)

    # 2) Ubah ke grayscale 1 channel.
    gray = pil_image.convert("L")

    # 3) Resize ke 28x28 (ukuran input model).
    gray = gray.resize((28, 28), Image.Resampling.LANCZOS)

    # 4) Normalisasi ke [0, 1].
    arr = np.asarray(gray, dtype="float32") / 255.0

    # 5) Tentukan perlu inversi atau tidak.
    #    Mode "auto": lihat rata-rata piksel tepi (asumsi = latar).
    #    Kalau latar terang (> 0.5), balik agar latar jadi gelap.
    border = np.concatenate(
        [arr[0, :], arr[-1, :], arr[:, 0], arr[:, -1]]
    )
    auto_inverted = False
    if invert_mode == "auto":
        if border.mean() > 0.5:
            arr = 1.0 - arr
            auto_inverted = True
    elif invert_mode == "always":
        arr = 1.0 - arr
        auto_inverted = True
    # invert_mode == "never" -> biarkan apa adanya.

    # 6) Bentuk batch: (1, 28, 28, 1).
    model_input = arr.reshape(1, 28, 28, 1)
    return model_input, arr, auto_inverted


# ------------------------------------------------------------
# 4. HEADER
# ------------------------------------------------------------
st.title("👕 Klasifikasi Pakaian — CNN Fashion-MNIST")
st.caption(
    "Unggah gambar sebuah item pakaian, model CNN akan memprediksi "
    "salah satu dari 10 kategori Fashion-MNIST."
)

if model is None:
    st.error(
        "Model tidak ditemukan. Letakkan file "
        "`cnn_fashion_mnist_optimized_v2.keras` (atau "
        "`best_cnn_fashion_mnist_v2.keras`) di folder yang sama "
        "dengan `app.py`, lalu jalankan ulang."
    )
    st.stop()
else:
    st.success(f"Model dimuat dari: `{model_path}`")


# ------------------------------------------------------------
# 5. SIDEBAR: PENGATURAN & INFO
# ------------------------------------------------------------
with st.sidebar:
    st.header("⚙️ Pengaturan")
    invert_choice = st.radio(
        "Mode inversi latar",
        options=["auto", "always", "never"],
        index=0,
        help=(
            "Fashion-MNIST = objek terang di latar hitam. "
            "'auto' otomatis membalik jika latar foto terang. "
            "Ubah ke 'always'/'never' bila hasil terlihat keliru."
        ),
    )

    st.markdown("---")
    st.subheader("ℹ️ Tentang")
    st.markdown(
        "- **Input model:** 28×28, grayscale, dinormalisasi /255\n"
        "- **Arsitektur:** CNN (Conv + BatchNorm + Dropout)\n"
        "- **Kelas:** 10 kategori pakaian\n"
        "- **Tips foto:** objek jelas, latar polos, "
        "seluruh item terlihat."
    )

    st.markdown("---")
    st.caption("Tugas Kelompok 2 — Optimasi & Implementasi CNN")


# ------------------------------------------------------------
# 6. INPUT GAMBAR
# ------------------------------------------------------------
uploaded = st.file_uploader(
    "Unggah gambar (JPG / PNG)",
    type=["jpg", "jpeg", "png", "bmp", "webp"],
)

if uploaded is None:
    st.info("Silakan unggah satu gambar item pakaian untuk memulai.")
    st.stop()

# Buka gambar yang diunggah.
try:
    pil_image = Image.open(uploaded)
except Exception as exc:  # noqa: BLE001
    st.error(f"Gagal membuka gambar: {exc}")
    st.stop()


# ------------------------------------------------------------
# 7. PREPROCESSING + PREDIKSI
# ------------------------------------------------------------
model_input, processed_2d, was_inverted = preprocess_image(
    pil_image, invert_mode=invert_choice
)

probabilities = model.predict(model_input, verbose=0)[0]
predicted_index = int(np.argmax(probabilities))
confidence = float(probabilities[predicted_index]) * 100.0


# ------------------------------------------------------------
# 8. TAMPILKAN GAMBAR (asli vs yang dilihat model)
# ------------------------------------------------------------
col_a, col_b = st.columns(2)
with col_a:
    st.markdown("**Gambar asli**")
    st.image(pil_image, use_container_width=True)
with col_b:
    st.markdown("**Yang 'dilihat' model (28×28)**")
    # Perbesar agar mudah dilihat mata (nearest agar tetap tajam per-piksel).
    preview = (processed_2d * 255).astype("uint8")
    preview_img = Image.fromarray(preview).resize(
        (196, 196), Image.Resampling.NEAREST
    )
    st.image(preview_img, use_container_width=True)

if invert_choice == "auto" and was_inverted:
    st.caption("🔁 Latar terang terdeteksi — gambar dibalik otomatis.")


# ------------------------------------------------------------
# 9. HASIL PREDIKSI
# ------------------------------------------------------------
st.markdown("## 🎯 Hasil Prediksi")
st.markdown(
    f"### {CLASS_NAMES[predicted_index]} "
    f"_({CLASS_NAMES_ID[predicted_index]})_"
)
st.progress(min(confidence / 100.0, 1.0))
st.metric("Tingkat keyakinan", f"{confidence:.2f}%")

# Tabel probabilitas semua kelas, diurutkan dari tertinggi.
st.markdown("#### Probabilitas seluruh kelas")
order = np.argsort(probabilities)[::-1]
table_rows = {
    "Kelas": [CLASS_NAMES[i] for i in order],
    "Probabilitas (%)": [f"{probabilities[i] * 100:.2f}" for i in order],
}
st.dataframe(table_rows, use_container_width=True, hide_index=True)

# Grafik batang probabilitas.
chart_data = {CLASS_NAMES[i]: float(probabilities[i]) for i in range(10)}
st.bar_chart(chart_data, horizontal=True)

# Peringatan bila model ragu (mungkin gambar di luar distribusi).
if confidence < 55.0:
    st.warning(
        "Keyakinan model rendah. Gambar mungkin kurang mirip data "
        "training (coba latar polos, ubah mode inversi, atau gunakan "
        "foto item pakaian yang lebih jelas)."
    )
