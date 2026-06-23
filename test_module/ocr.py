import base64
import io
import requests
import gradio as gr
from PIL import Image, ImageDraw, ImageFont
import tempfile

OCR_URL = "http://87.205.21.33:44953/ocr"


def _draw_boxes(image: Image.Image, blocks: list) -> Image.Image:
    img = image.copy().convert("RGB")
    draw = ImageDraw.Draw(img)
    try:
        font = ImageFont.truetype("/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf", 14)
    except Exception:
        font = ImageFont.load_default()
    for block in blocks:
        x1, y1, x2, y2 = block["box"]
        draw.rectangle([x1, y1, x2, y2], outline="red", width=2)
        label = f"{block['score']:.2f}"
        draw.text((x1, max(0, y1 - 16)), label, fill="red", font=font)
    return img


def run_ocr(image: Image.Image):
    if image is None:
        return None, "Upload an image"

    with tempfile.NamedTemporaryFile(suffix=".png", delete=True) as f:
        image.save(f.name)
        with open(f.name, "rb") as fp:
            resp = requests.post(
                OCR_URL,
                files={"image": ("image.png", fp, "image/png")},
                timeout=120,
            )

    try:
        resp.raise_for_status()
        data = resp.json()
    except Exception as e:
        return None, f"Request failed:\n{e}\n\n{resp.text}"

    blocks = data.get("blocks", [])
    img_b64 = data.get("image")
    display = Image.open(io.BytesIO(base64.b64decode(img_b64))).convert("RGB") if img_b64 else image

    if not blocks:
        return display, "(no text detected)"

    annotated = _draw_boxes(display, blocks)
    text = "\n\n".join(
        f"[{b['score']:.2f}] {b['text']}" for b in blocks
    )
    return annotated, text


with gr.Blocks(title="PaddleOCR Test") as app:
    gr.Markdown("# PaddleOCR Server Test")

    with gr.Row():
        image_in = gr.Image(type="pil", label="Input Image")
        image_out = gr.Image(type="pil", label="Detected Regions")

    run_btn = gr.Button("Run OCR")
    output = gr.Textbox(label="Text Blocks", lines=20)

    run_btn.click(fn=run_ocr, inputs=image_in, outputs=[image_out, output])
    image_in.change(fn=run_ocr, inputs=image_in, outputs=[image_out, output])

app.launch(server_name="0.0.0.0", server_port=7860)
