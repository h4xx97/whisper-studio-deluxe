import gradio as gr
import subprocess
import os
import tempfile
from pathlib import Path

ROOT = Path(__file__).parent
WHISPER_BIN = ROOT / "build" / "bin" / "whisper-cli"
MODEL_PATH  = ROOT / "models" / "ggml-small.bin"

def transcribe(audio_file, language, make_srt):
    if not audio_file:
        return "Aucun fichier audio fourni."
    if not WHISPER_BIN.exists():
        return f"Binaire introuvable : {WHISPER_BIN}"
    if not MODEL_PATH.exists():
        return f"Modèle introuvable : {MODEL_PATH}"

    with tempfile.TemporaryDirectory() as tmpdir:
        tmpdir = Path(tmpdir)
        base = tmpdir / "out"

        cmd = [
            str(WHISPER_BIN),
            "-m", str(MODEL_PATH),
            "-f", audio_file,
            "-otxt",
            "-of", str(base),
        ]

        if language and language.strip():
            cmd += ["-l", language.strip()]

        if make_srt:
            cmd += ["-osrt"]

        try:
            subprocess.run(cmd, check=True, capture_output=True)
        except subprocess.CalledProcessError as e:
            return (e.stderr.decode(errors="ignore")
                    or e.stdout.decode(errors="ignore"))

        txt_file = base.with_suffix(".txt")
        if not txt_file.exists():
            return "La transcription n'a pas été générée."

        return txt_file.read_text(encoding="utf-8", errors="ignore")

with gr.Blocks() as demo:
    gr.Markdown("# 🎧 Whisper.cpp WebUI — Model small — Local & rapide")
    audio = gr.Audio(type="filepath", label="Audio")
    lang  = gr.Textbox(label="Langue (fr, en… vide = auto)", value="fr")
    srt   = gr.Checkbox(label="Générer un SRT", value=False)
    btn   = gr.Button("Transcrire")
    out   = gr.Textbox(label="Résultat", lines=20)
    btn.click(transcribe, [audio, lang, srt], out)

if __name__ == "__main__":
    demo.launch(server_name="0.0.0.0", server_port=7860)
