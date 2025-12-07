import sys
import textwrap
import gradio as gr
import subprocess
import os
from pathlib import Path
from datetime import datetime
from fpdf import FPDF
import yt_dlp
import math
import shlex


ROOT = Path(__file__).parent
WHISPER_BIN = ROOT / "build" / "bin" / "whisper-cli"
MODEL_PATH  = ROOT / "models" / "ggml-small.bin"
OUTPUT_DIR  = ROOT / "outputs"
OUTPUT_DIR.mkdir(exist_ok=True)


def log(msg):
    ts = datetime.now().strftime("[%H:%M:%S]")
    print(ts, msg)
    sys.stdout.flush()


def run_whisper(input_file: Path, language: str, make_srt: bool, make_json: bool, run_id: str, progress=None):
    run_dir = OUTPUT_DIR / run_id
    run_dir.mkdir(exist_ok=True)

    # Logs "live" sur le terminal
    log(f"Extraction audio : {input_file}")

    # 1) On extrait une piste audio propre (wav 16 kHz mono)
    if progress:
        progress(0.05, desc="Extraction audio...")
    audio_path = extract_audio_if_needed(input_file, run_dir)

    # 2) On regarde la durée & on découpe si très long (ex: > 2h)
    duration = get_media_duration_seconds(audio_path)
    log(f"Durée détectée : {duration/60:.1f} min")

    chunks = split_long_audio(audio_path, run_dir, max_chunk_sec=2 * 3600)
    n_chunks = len(chunks)
    log(f"Découpage en {n_chunks} morceaux")

    full_text = []
    txt_files = []
    srt_files = []
    json_files = []

    for idx, chunk in enumerate(chunks):
        base = run_dir / f"transcript_{idx:03d}"

        log(f"Traitement chunk {idx+1}/{n_chunks}")

        cmd = [
            str(WHISPER_BIN),
            "-m", str(MODEL_PATH),
            "-f", str(chunk),
            "-otxt",
            "-of", str(base),
        ]

        if language and language.strip():
            cmd += ["-l", language.strip()]

        if make_srt:
            cmd += ["-osrt"]

        if make_json:
            cmd += ["-oj"]

        if progress:
            ratio = (idx / max(n_chunks, 1))
            progress(0.1 + 0.8 * ratio, desc=f"Transcription chunk {idx+1}/{n_chunks}...")

        res = subprocess.run(cmd, capture_output=True, text=True)
        if res.returncode != 0:
            err = res.stderr or res.stdout
            raise RuntimeError(f"Erreur WhisperCLI (chunk {idx+1}):\n{err}")

        txt_file = base.with_suffix(".txt")
        if txt_file.exists():
            txt_files.append(txt_file)
            full_text.append(txt_file.read_text(encoding="utf-8", errors="ignore"))

        srt_file = base.with_suffix(".srt")
        if srt_file.exists():
            srt_files.append(srt_file)

        json_file = base.with_suffix(".json")
        if json_file.exists():
            json_files.append(json_file)

    # Concat simple du texte. (SRT & JSON restent chunkés pour l'instant)
    joined_text = "\n\n".join(full_text)
    if progress:
        progress(0.95, desc="Finalisation...")

    # On garde le répertoire pour pouvoir packer tout ça proprement
    return joined_text, txt_files, srt_files, json_files, run_dir, duration


def make_pdf_from_text(text: str, run_dir: Path) -> Path:
    """
    Crée un PDF propre avec :
    - logo en haut si présent
    - police Roboto (TTF) si dispo, sinon fallback Helvetica
    - gestion des longues lignes
    """
    pdf_path = run_dir / "transcript.pdf"

    pdf = FPDF()
    # marges raisonnables
    pdf.set_margins(left=15, top=20, right=15)
    pdf.set_auto_page_break(auto=True, margin=20)
    pdf.add_page()

    base_dir = Path(__file__).parent
    font_dir = base_dir / "fonts"

    # Ajout des polices TTF (en essayant d'être tolérant si un fichier manque)
    try:
        pdf.add_font("Roboto", "", str(font_dir / "Roboto-Regular.ttf"), uni=True)
    except Exception:
        pass
    try:
        pdf.add_font("Roboto", "B", str(font_dir / "Roboto-Bold.ttf"), uni=True)
    except Exception:
        pass
    try:
        pdf.add_font("RobotoMono", "", str(font_dir / "RobotoMono-Regular.ttf"), uni=True)
    except Exception:
        pass

    # Logo si présent
    logo_path = base_dir / "logo.png"
    if logo_path.exists():
        # largeur 30 mm, position en haut à gauche
        pdf.image(str(logo_path), x=15, y=15, w=30)
        pdf.ln(30)  # on descend sous le logo

    # Titre
    try:
        pdf.set_font("Roboto", "B", 14)
    except RuntimeError:
        # fallback si Roboto n'est pas connue
        pdf.set_font("Helvetica", "B", 14)

    pdf.cell(0, 10, "Transcription Whisper", ln=True)
    pdf.ln(5)

    # Corps du texte
    try:
        pdf.set_font("Roboto", "", 11)
    except RuntimeError:
        pdf.set_font("Helvetica", "", 11)

    # Largeur effective de texte (pour éviter le bug "not enough horizontal space...")
    effective_width = pdf.w - pdf.l_margin - pdf.r_margin
    if effective_width <= 0:
        effective_width = 100  # valeur de secours

    # On découpe les lignes trop longues (URLs, etc.)
    for paragraph in text.split("\n"):
        paragraph = paragraph.strip()
        if not paragraph:
            pdf.ln(4)
            continue

        # On "wrap" le texte pour éviter les lignes infinies
        wrapped_lines = textwrap.wrap(
            paragraph,
            width=100,  # nombre de caractères approx
            break_long_words=True,
            break_on_hyphens=False,
        )
        for line in wrapped_lines:
            pdf.multi_cell(effective_width, 6, line)
        pdf.ln(2)

    pdf.output(str(pdf_path))
    return pdf_path


def download_youtube_audio(url: str, run_id: str):
    run_dir = OUTPUT_DIR / run_id
    run_dir.mkdir(exist_ok=True)
    outtmpl = str(run_dir / "yt_audio.%(ext)s")

    ydl_opts = {
        "format": "bestaudio/best",
        "outtmpl": outtmpl,
        "noplaylist": True,
        "quiet": True,
        "no_warnings": True,
    }

    with yt_dlp.YoutubeDL(ydl_opts) as ydl:
        info = ydl.extract_info(url, download=True)
        downloaded = ydl.prepare_filename(info)
    return Path(downloaded)


def transcribe(media_file, youtube_url, language, make_srt, make_json, make_pdf, history, progress=gr.Progress()):
    if not WHISPER_BIN.exists():
        return ("Binaire introuvable : "
                f"{WHISPER_BIN}"), None, None, None, None, history, ""
    if not MODEL_PATH.exists():
        return ("Modèle introuvable : "
                f"{MODEL_PATH}"), None, None, None, None, history, ""

    run_id = datetime.now().strftime("%Y%m%d_%H%M%S")
    source_desc = ""
    input_path = None

    try:
        log("Début transcription")
        progress(0.01, desc="Préparation...")

        if youtube_url and youtube_url.strip():
            source_desc = f"YouTube: {youtube_url.strip()}"
            log(f"Source : {source_desc}")
            progress(0.05, desc="Téléchargement YouTube...")
            input_path = download_youtube_audio(youtube_url.strip(), run_id)
            log("Téléchargement YouTube terminé")
        elif media_file:
            source_desc = f"Fichier upload: {Path(media_file).name}"
            input_path = Path(media_file)
            log(f"Source : {source_desc}")
        else:
            return "Aucun fichier ni URL fournie.", None, None, None, None, history, ""

        # On lance Whisper (avec extraction audio, découpage, etc.)
        text, txt_files, srt_files, json_files, run_dir, duration = run_whisper(
            input_file=input_path,
            language=language,
            make_srt=make_srt,
            make_json=make_json,
            run_id=run_id,
            progress=progress,
        )

        # Estimation simple : small sur N100 ≈ 0.3x à 1x temps réel
        if duration > 0:
            est_factor = 0.7  # ajustable
            est_time = duration * est_factor
            est_min = int(est_time // 60)
            est_sec = int(est_time % 60)
            estimation_text = f"(Durée audio ~ {duration/60:.1f} min, temps de traitement estimé ~ {est_min} min {est_sec} s)\n\n"
        else:
            estimation_text = ""

        pdf_file = None
        if make_pdf:
            pdf_file = make_pdf_from_text(text, run_dir)
            log("PDF généré")

        # On ne renvoie qu’un TXT/SRT/JSON/PDF principal (le premier chunk) pour le téléchargement direct
        txt_file_out = txt_files[0] if txt_files else None
        srt_file_out = srt_files[0] if srt_files else None
        json_file_out = json_files[0] if json_files else None

        history = history or []
        history.insert(0, f"- {datetime.now().strftime('%H:%M:%S')} · {source_desc} · id={run_id}")
        history = history[:10]
        history_md = "\n".join(history)

        progress(1.0, desc="Terminé ✅")
        log("Transcription terminée")

        return estimation_text + (text or "(Transcription vide)"), \
               str(txt_file_out) if txt_file_out else None, \
               str(srt_file_out) if srt_file_out else None, \
               str(pdf_file) if pdf_file else None, \
               str(json_file_out) if json_file_out else None, \
               history, history_md

    except Exception as e:
        err = f"Erreur pendant le traitement : {e}"
        log(err)
        return err, None, None, None, None, history, ""


def get_media_duration_seconds(path: Path) -> float:
    """Retourne la durée en secondes via ffprobe."""
    cmd = [
        "ffprobe",
        "-v", "error",
        "-show_entries", "format=duration",
        "-of", "default=nw=1:nk=1",
        str(path),
    ]
    try:
        out = subprocess.check_output(cmd, stderr=subprocess.STDOUT).decode().strip()
        return float(out)
    except Exception:
        return 0.0


def extract_audio_if_needed(path: Path, run_dir: Path) -> Path:
    """
    Si c'est une vidéo, on extrait une piste audio propre.
    Pour simplifier, on convertit tout en wav mono 16 kHz.
    """
    audio_path = run_dir / "audio.wav"
    cmd = [
        "ffmpeg",
        "-y",
        "-i", str(path),
        "-vn",             # pas de vidéo
        "-acodec", "pcm_s16le",
        "-ar", "16000",
        "-ac", "1",
        str(audio_path),
    ]
    subprocess.run(cmd, check=True)
    return audio_path


def split_long_audio(audio_path: Path, run_dir: Path, max_chunk_sec: int = 2 * 3600):
    """
    Découpe l'audio en chunks si > max_chunk_sec (par défaut 2h).
    Retourne la liste des chemins de chunks.
    """
    duration = get_media_duration_seconds(audio_path)
    if duration <= max_chunk_sec or duration <= 0:
        return [audio_path]

    n_chunks = math.ceil(duration / max_chunk_sec)
    chunk_paths = []

    for i in range(n_chunks):
        start = i * max_chunk_sec
        out_path = run_dir / f"chunk_{i:03d}.wav"
        cmd = [
            "ffmpeg",
            "-y",
            "-i", str(audio_path),
            "-ss", str(start),
            "-t", str(max_chunk_sec),
            "-acodec", "copy",
            str(out_path),
        ]
        subprocess.run(cmd, check=True)
        chunk_paths.append(out_path)

    return chunk_paths


# 🔥 Créer l’interface explicitement
demo = gr.Blocks(css=None)

with demo:
    gr.Markdown("# 🧠 Whisper Studio Deluxe\n### Audio / Vidéo / YouTube → Texte, SRT, JSON, PDF — en local sur ton serveur N100")

    history_state = gr.State([])

    with gr.Tab("Transcription"):
        with gr.Row():
            media = gr.Audio(type="filepath", label="Fichier audio / vidéo")
        with gr.Row():
            yt = gr.Textbox(label="URL YouTube (optionnel)")
        with gr.Row():
            lang = gr.Textbox(label="Langue (fr, en — vide = auto)", value="fr")
        with gr.Row():
            make_srt = gr.Checkbox(label="Générer un fichier SRT", value=True)
            make_json = gr.Checkbox(label="Générer un JSON", value=False)
            make_pdf = gr.Checkbox(label="Générer un PDF", value=True)

        btn = gr.Button("🚀 Lancer la transcription")

        with gr.Row():
            txt_out = gr.Textbox(label="Transcription", lines=20)

        with gr.Row():
            txt_file_out = gr.File(label="Télécharger TXT")
            srt_file_out = gr.File(label="Télécharger SRT")
            pdf_file_out = gr.File(label="Télécharger PDF")
            json_file_out = gr.File(label="Télécharger JSON")

    with gr.Tab("Historique"):
        history_md = gr.Markdown("*(Aucun job encore)*")

    btn.click(
        fn=transcribe,
        inputs=[media, yt, lang, make_srt, make_json, make_pdf, history_state],
        outputs=[txt_out, txt_file_out, srt_file_out, pdf_file_out, json_file_out, history_state, history_md]
    )

demo.queue(max_size=10).launch(server_name="0.0.0.0", server_port=7860)
