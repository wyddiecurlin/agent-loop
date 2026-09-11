"""Small JPEGs from photo bytes; all input/output stays with the caller."""

import base64
from io import BytesIO
from threading import Lock

from PIL import Image, ImageOps
from pillow_heif import register_heif_opener

register_heif_opener()

MAX_INPUT_BYTES = 100_000_000
MAX_PIXELS = 80_000_000
MAX_IMAGE_BYTES = 50_000
MAX_EDGE = 512
MAX_PAGES = 200
_PDF_LOCK = Lock()  # PDFium forbids concurrent calls, even for separate documents.


def jpeg(image: Image.Image) -> dict:
	"""Resize before encoding: small files alone do not imply few vision tokens."""
	if image.width * image.height > MAX_PIXELS:
		raise ValueError(f"image exceeds {MAX_PIXELS} decoded pixels")
	image = ImageOps.exif_transpose(image)
	image.thumbnail((MAX_EDGE, MAX_EDGE))
	if image.mode in ("RGBA", "LA", "P"):
		rgba = image.convert("RGBA")
		image = Image.new("RGB", rgba.size, "white")
		image.paste(rgba, mask=rgba.getchannel("A"))
	else:
		image = image.convert("RGB")
	quality = 40
	while True:
		out = BytesIO()
		image.save(out, format="JPEG", quality=quality, optimize=True)
		data = out.getvalue()
		if len(data) <= MAX_IMAGE_BYTES:
			return {"type": "input_image", "image_url": "data:image/jpeg;base64," + base64.b64encode(data).decode(),
			        "detail": "low"}
		if quality > 20:
			quality -= 10
		else:
			image = image.resize((max(1, image.width * 3 // 4), max(1, image.height * 3 // 4)))


def convert(data: bytes) -> list[dict]:
	"""Decode photos/RAW or split PDFs into ordered, bounded page images."""
	if not data or len(data) > MAX_INPUT_BYTES:
		raise ValueError(f"input must contain 1–{MAX_INPUT_BYTES} bytes")
	if data.startswith(b"%PDF-"):
		import pypdfium2 as pdfium
		parts = []
		with _PDF_LOCK, pdfium.PdfDocument(data) as document:
			if not 0 < len(document) <= MAX_PAGES:
				raise ValueError(f"PDF must contain 1–{MAX_PAGES} pages")
			for index in range(len(document)):
				page = document[index]
				try:
					width, height = page.get_size()
					if min(width, height) <= 0:
						raise ValueError(f"invalid PDF page {index + 1}")
					bitmap = page.render(scale=min(1, MAX_EDGE / max(width, height)))
					try:
						parts.extend([{"type": "input_text", "text": f"Page {index + 1}"}, jpeg(bitmap.to_pil())])
					finally:
						bitmap.close()
				finally:
					page.close()
		return parts
	try:
		with Image.open(BytesIO(data)) as image:
			# RAWs often expose a tiny TIFF preview. Decode the sensor data first.
			if image.format != "TIFF":
				return [jpeg(image)]
	except OSError:
		pass
	import rawpy
	try:
		with rawpy.imread(BytesIO(data)) as raw:
			if raw.sizes.raw_width * raw.sizes.raw_height > MAX_PIXELS:
				raise ValueError(f"RAW exceeds {MAX_PIXELS} decoded pixels")
			pixels = raw.postprocess(half_size=True, use_camera_wb=True, output_bps=8)
		return [jpeg(Image.fromarray(pixels))]
	except rawpy.LibRawError:
		try:
			with Image.open(BytesIO(data)) as image:
				return [jpeg(image)]
		except OSError as exc:
			raise ValueError("invalid or unsupported photo/RAW format") from exc
