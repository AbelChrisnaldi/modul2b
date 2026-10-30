"""Klasifikasi Groq untuk Modul 2A/2B.

Mode 'guided_score' = LLM TERBIMBING skor linear yang dipelajari dari TRAIN;
ini bukan zero-shot LLM murni. Keputusan akhir/label dikembalikan API Groq.
"""
from __future__ import annotations

import json
import re
import time
from pathlib import Path

LABELS = ["Aman", "Waspada", "Berisiko"]
FEATURES = ["ipk", "kehadiran", "jam_belajar", "tugas_terkumpul", "nilai_uts", "aktif_organisasi"]
NAMA_FITUR = FEATURES  # Alias untuk kompatibilitas dengan app.py Modul 2B
PROMPT_GUIDED = """Kamu bertugas mengembalikan SATU label status akademik mahasiswa.
Pilihan label: Aman, Waspada, Berisiko.
Gunakan SKOR TERKALIBRASI yang diberikan pada pesan pengguna, dengan aturan tegas:
- skor di bawah batas pertama = Berisiko
- skor dari batas pertama sampai sebelum batas kedua = Waspada
- skor sama dengan atau lebih besar daripada batas kedua = Aman
Skor berasal dari pemodelan fitur pada DATA TRAIN, bukan dari label data yang sedang diuji.
Jangan mengganti kelas dari skor dengan tebakan atau asumsi umum.
Respons hanya JSON seperti {"label":"Aman"}.
"""
PROMPT_PURE = """Kamu adalah pengklasifikasi status akademik mahasiswa.
Pilih tepat satu: Aman, Waspada, Berisiko.
Pertimbangkan IPK, kehadiran, nilai UTS, tugas terkumpul, dan jam belajar secara gabungan.
Aktivitas organisasi tidak otomatis menentukan kelas. Jangan gunakan Waspada
sebagai jawaban default. Aman untuk kondisi gabungan baik, Berisiko untuk
kondisi gabungan rendah, dan Waspada untuk kondisi peralihan.
Jawab hanya JSON seperti {"label":"Aman"}.
"""


def load_config(path="config.json"):
    with Path(path).open(encoding="utf-8") as f:
        return json.load(f)


def _number(value, name):
    n = float(value)
    if not (n == n and abs(n) < float('inf')):
        raise ValueError(f"{name}: angka harus finite")
    return n


def format_input(data):
    missing = sorted(set(FEATURES) - set(data))
    if missing:
        raise ValueError(f"Fitur kurang: {missing}")
    organization = str(data['aktif_organisasi']).strip().lower()
    if organization not in ("ya", "tidak"):
        raise ValueError("aktif_organisasi harus Ya atau Tidak")
    # Pengetikan eksplisit (bukan string bebas) membantu mengurangi prompt injection.
    fields = [f"{k}: {_number(data[k], k):g}" for k in FEATURES if k != "aktif_organisasi"]
    fields.append("aktif_organisasi: " + ("Ya" if organization == "ya" else "Tidak"))
    return '\n'.join(fields)


def compute_guided_score(cfg, data):
    rubric = cfg['kalibrasi_train']
    x = {k: _number(data[k], k) for k in FEATURES if k != "aktif_organisasi"}
    x['aktif_organisasi'] = 1.0 if str(data['aktif_organisasi']).strip().lower() == 'ya' else 0.0
    return float(rubric['intercept']) + sum(float(v) * x[k] for k, v in rubric['koefisien'].items())


def build_messages(cfg, data):
    user = format_input(data)
    if cfg.get('metode') == 'guided_score':
        skor = compute_guided_score(cfg, data)
        lower, upper = cfg['kalibrasi_train']['batas']
        user += (f"\n\nSKOR TERKALIBRASI = {skor:.6f}"
                 f"\nBATAS PERTAMA = {lower:.6f}"
                 f"\nBATAS KEDUA = {upper:.6f}"
                 "\nPilih satu label hanya dari perbandingan skor dengan batas di atas.")
    msgs = [{"role": "system", "content": cfg['system_prompt']}]
    for example in cfg.get('few_shot', []):
        msgs.append({"role": "user", "content": format_input(example['data'])})
        msgs.append({"role": "assistant", "content": json.dumps({"label": example['label']}, ensure_ascii=False)})
    msgs.append({"role": "user", "content": user})
    return msgs


def parse_label(content):
    if not content:
        return "invalid"
    try:
        parsed = json.loads(content)
        raw = parsed.get('label') if isinstance(parsed, dict) else parsed
    except (TypeError, ValueError):
        raw = content
    if not isinstance(raw, str):
        return "invalid"
    normalized = re.sub(r'[^a-z]', '', raw.casefold())
    return next((label for label in LABELS if label.casefold() == normalized), 'invalid')


def classify(client, cfg, data, max_retry=4):
    """Return label only from Groq. API failures stop evaluation, never become invalid labels."""
    import groq
    schema = {
        "type": "object", "properties": {"label": {"type": "string", "enum": LABELS}},
        "required": ["label"], "additionalProperties": False
    }
    args = {
        "model": cfg['model'], "temperature": cfg.get('temperature', 0),
        "max_tokens": cfg.get('max_tokens', 160),
        "messages": build_messages(cfg, data),
        "response_format": {"type": "json_schema", "json_schema": {
            "name": "status_akademik", "strict": True, "schema": schema
        }}
    }
    if cfg.get('reasoning_effort'):
        args['reasoning_effort'] = cfg['reasoning_effort']
    for attempt in range(max_retry):
        try:
            result = client.chat.completions.create(**args)
            return parse_label(result.choices[0].message.content)
        except (groq.RateLimitError, groq.APIConnectionError, groq.APITimeoutError) as error:
            if attempt == max_retry - 1:
                raise RuntimeError("Groq terus membatasi/gagal tersambung. Jalankan ulang evaluasi setelah koneksi pulih.") from error
            delay = 12.0 * (attempt + 1)
            if isinstance(error, groq.RateLimitError) and error.response is not None:
                retry = error.response.headers.get('retry-after')
                if retry:
                    try:
                        delay = max(delay, float(retry))
                    except ValueError:
                        pass
            print(f"\nAPI sedang dibatasi. Menunggu {delay:g} detik...", flush=True)
            time.sleep(delay)
    raise RuntimeError("Gagal memperoleh prediksi")
