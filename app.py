import os
import re
import base64
import io
from flask import Flask, request, jsonify
from anthropic import Anthropic

# OCR is intentionally kept in this backend so the HTML only captures frames.
try:
    from PIL import Image, ImageOps, ImageEnhance, ImageFilter
    PIL_OK = True
except Exception:
    PIL_OK = False

try:
    import pytesseract
    TESSERACT_OK = True
except Exception:
    TESSERACT_OK = False

app = Flask(__name__)

# CORS: allow the HTML tool to call this backend from another domain.
@app.after_request
def add_cors_headers(response):
    response.headers["Access-Control-Allow-Origin"] = "*"
    response.headers["Access-Control-Allow-Headers"] = "Content-Type, Authorization, x-api-key"
    response.headers["Access-Control-Allow-Methods"] = "GET, POST, OPTIONS"
    return response

API_KEY = os.environ.get("ANTHROPIC_API_KEY", "").strip()
MODEL = os.environ.get("ANTHROPIC_MODEL", "claude-3-5-haiku-latest").strip()




def _decode_image(value):
    if not value:
        raise ValueError("Thiếu ảnh OCR")
    raw = str(value)
    if raw.startswith("data:"):
        raw = raw.split(",", 1)[1]
    return base64.b64decode(raw)


def _extract_session(text):
    text = str(text or "")
    # Chuẩn hóa một vài lỗi OCR phổ biến.
    t = text.replace("\n", " ")
    t = re.sub(r"[OoQq]", "0", t)
    t = re.sub(r"[Il|]", "1", t)
    t = re.sub(r"[Ss]", "5", t)
    t = re.sub(r"[Bb]", "8", t)

    # Ưu tiên chuỗi đứng sau # / PHIÊN / SESSION.
    m = re.search(r"(?:#|PHI\s*[EÊÈÉ]N|SESSION|MA\s*PHI\s*[EÊÈÉ]N)\s*[:#-]?\s*((?:\d[\s-]*){5,12})", t, re.I)
    if m:
        n = re.sub(r"\D", "", m.group(1))
        if 5 <= len(n) <= 12:
            return n

    candidates = []
    for x in re.findall(r"(?<!\d)(?:\d[\s-]?){4,11}\d(?!\d)", t):
        n = re.sub(r"\D", "", x)
        if 5 <= len(n) <= 12:
            candidates.append(n)
    if candidates:
        return max(candidates, key=len)
    return None


def _ocr_image(image_bytes):
    if not PIL_OK:
        raise RuntimeError("Thiếu Pillow cho OCR")
    if not TESSERACT_OK:
        raise RuntimeError("Thiếu pytesseract cho OCR")

    img = Image.open(io.BytesIO(image_bytes)).convert("RGB")
    # Phóng to và tạo nhiều biến thể để tăng khả năng đọc mã phiên.
    scale = 2.0 if max(img.size) < 1800 else 1.0
    if scale != 1.0:
        img = img.resize((int(img.width * scale), int(img.height * scale)))

    gray = ImageOps.grayscale(img)
    gray = ImageEnhance.Contrast(gray).enhance(1.8)
    gray = gray.filter(ImageFilter.SHARPEN)
    variants = [img, gray, gray.point(lambda p: 255 if p > 145 else 0)]

    texts = []
    configs = [
        "--psm 6 -c tessedit_char_whitelist=0123456789#",
        "--psm 11 -c tessedit_char_whitelist=0123456789#"
    ]
    for variant in variants:
        for config in configs:
            try:
                out = pytesseract.image_to_string(variant, config=config)
                if out:
                    texts.append(out)
            except Exception:
                pass

    combined = "\n".join(texts)
    return _extract_session(combined), combined


def get_client():
    if not API_KEY:
        raise RuntimeError("ANTHROPIC_API_KEY chưa được cấu hình trên Render")
    return Anthropic(api_key=API_KEY)


@app.route("/ocr", methods=["POST", "OPTIONS"])
def ocr():
    if request.method == "OPTIONS":
        return ("", 204)

    data = request.get_json(silent=True) or {}
    image = data.get("image") or data.get("frame")
    if not image:
        return jsonify({"ok": False, "error": "Thiếu image/frame"}), 400

    try:
        image_bytes = _decode_image(image)
        session, raw_text = _ocr_image(image_bytes)
        return jsonify({
            "ok": True,
            "session": session,
            "raw_text": raw_text[:2000],
            "ocr": True
        })
    except Exception as e:
        # Nếu OCR local chưa được cài, dùng chính AI backend để đọc ảnh.
        # Đây là fallback, không đoán mã khi ảnh không rõ.
        try:
            clean = str(image).split(",", 1)[-1]
            client = get_client()
            message = client.messages.create(
                model=MODEL,
                max_tokens=32,
                temperature=0,
                messages=[{
                    "role": "user",
                    "content": [
                        {"type": "image", "source": {"type": "base64", "media_type": "image/jpeg", "data": clean}},
                        {"type": "text", "text": "Đọc mã phiên nằm sau dấu #. Chỉ trả về 5-12 chữ số nếu nhìn rõ; nếu không rõ trả về NONE. Không đoán."}
                    ]
                }]
            )
            text = "".join(block.text for block in message.content if getattr(block, "type", "") == "text").strip()
            session = _extract_session(text)
            return jsonify({"ok": True, "session": session, "raw_text": text, "ocr": False, "ai_fallback": True})
        except Exception as ai_error:
            return jsonify({"ok": False, "error": str(e), "ai_fallback_error": str(ai_error)}), 500


@app.get("/")
def root():
    return jsonify({
        "ok": True,
        "service": "AI backend",
        "status": "online",
        "endpoint": "/analyze"
    })


@app.get("/health")
def health():
    return jsonify({"ok": True, "service": "AI backend", "status": "online"})


@app.route("/analyze", methods=["POST", "OPTIONS"])
def analyze():
    if request.method == "OPTIONS":
        return ("", 204)

    data = request.get_json(silent=True) or {}
    prompt = str(data.get("prompt", "")).strip()

    if not prompt:
        return jsonify({"ok": False, "error": "Thiếu prompt"}), 400

    try:
        client = get_client()
        message = client.messages.create(
            model=MODEL,
            max_tokens=500,
            temperature=0,
            messages=[
                {"role": "user", "content": prompt}
            ]
        )

        text = "".join(
            block.text for block in message.content
            if getattr(block, "type", "") == "text"
        ).strip()

        return jsonify({
            "ok": True,
            "result": text,
            "model": MODEL
        })

    except Exception as e:
        return jsonify({
            "ok": False,
            "error": str(e)
        }), 500


if __name__ == "__main__":
    port = int(os.environ.get("PORT", "10000"))
    app.run(host="0.0.0.0", port=port)
