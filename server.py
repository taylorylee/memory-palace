# “””
Memory Palace — Backend (cross-platform, no GPU required)

Pipeline:

1. Receive uploaded image
1. Run Depth-Anything-V2 (CPU-compatible) to get a dense depth map
1. Back-project image pixels into 3D using the depth map → colored point cloud
1. Write as .ply and stream back to the browser

Works on Windows, macOS, Linux. No CUDA required.

Setup (one time):
pip install fastapi uvicorn python-multipart pillow numpy torch torchvision transformers

Run:
uvicorn server:app –host 0.0.0.0 –port 8000 –reload

Upgrade path (when you want true Gaussian splats on GPU):

- Deploy on Modal.com or Replicate with a GPU
- Swap estimate_depth() + depth_to_ply() for a SHARP inference call
- The frontend stays identical — it just receives a .ply either way
  “””

import io
import struct
import uuid
import numpy as np
from PIL import Image
from fastapi import FastAPI, File, UploadFile, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import Response

app = FastAPI(title=“Memory Palace API”)

app.add_middleware(
CORSMiddleware,
allow_origins=[”*”],
allow_methods=[“POST”, “GET”],
allow_headers=[”*”],
)

# Lazy-load the depth model once on first request

_depth_model = None
_depth_processor = None

MAX_SIDE = 518   # Depth-Anything-V2 native resolution
MAX_MB   = 20

def get_depth_model():
global _depth_model, _depth_processor
if _depth_model is None:
from transformers import AutoImageProcessor, AutoModelForDepthEstimation
model_id = “depth-anything/Depth-Anything-V2-Small-hf”
print(f”Loading {model_id} …”)
_depth_processor = AutoImageProcessor.from_pretrained(model_id)
_depth_model = AutoModelForDepthEstimation.from_pretrained(model_id)
_depth_model.eval()
print(“Depth model ready.”)
return _depth_processor, _depth_model

# ─────────────────────────────────────────────────────────────

# DEPTH INFERENCE

# ─────────────────────────────────────────────────────────────

def estimate_depth(img: Image.Image) -> np.ndarray:
“”“Returns a (H, W) float32 depth array, normalised 0-1 (1 = near).”””
import torch

```
processor, model = get_depth_model()

# Resize so longest side = MAX_SIDE and dims are multiples of 14
w, h = img.size
scale = MAX_SIDE / max(w, h)
new_w = ((int(w * scale)) // 14) * 14
new_h = ((int(h * scale)) // 14) * 14
img_resized = img.resize((new_w, new_h), Image.LANCZOS)

inputs = processor(images=img_resized, return_tensors="pt")
with torch.no_grad():
    outputs = model(**inputs)
    depth = outputs.predicted_depth.squeeze().numpy()

# Normalise and invert so near = 1, far = 0
d_min, d_max = depth.min(), depth.max()
if d_max - d_min > 1e-6:
    depth = (depth - d_min) / (d_max - d_min)
return (1.0 - depth).astype(np.float32)
```

# ─────────────────────────────────────────────────────────────

# POINT CLOUD → PLY

# ─────────────────────────────────────────────────────────────

def depth_to_ply(img: Image.Image, depth: np.ndarray, max_points: int = 300_000) -> bytes:
“”“Back-projects pixels into 3D via pinhole model. Returns binary PLY bytes.”””
dh, dw = depth.shape
pixels = np.array(img.resize((dw, dh), Image.LANCZOS).convert(“RGB”), dtype=np.float32)

```
xs = np.arange(dw, dtype=np.float32)
ys = np.arange(dh, dtype=np.float32)
xx, yy = np.meshgrid(xs, ys)

# Pinhole camera: assume ~60 degree FOV
fx = fy = dw / (2 * np.tan(np.radians(30)))
cx, cy = dw / 2.0, dh / 2.0

# Map normalised depth [0,1] to world Z in range [0.5, 5.0]
z = 0.5 + depth * 4.5

X = ((xx - cx) / fx * z).ravel()
Y = (-(yy - cy) / fy * z).ravel()
Z = (-z).ravel()
R = pixels[:, :, 0].ravel().astype(np.uint8)
G = pixels[:, :, 1].ravel().astype(np.uint8)
B = pixels[:, :, 2].ravel().astype(np.uint8)

# Subsample if over budget
n_total = len(X)
if n_total > max_points:
    idx = np.random.choice(n_total, max_points, replace=False)
    X, Y, Z, R, G, B = X[idx], Y[idx], Z[idx], R[idx], G[idx], B[idx]

n = len(X)

header = (
    "ply\n"
    "format binary_little_endian 1.0\n"
    f"element vertex {n}\n"
    "property float x\n"
    "property float y\n"
    "property float z\n"
    "property uchar red\n"
    "property uchar green\n"
    "property uchar blue\n"
    "end_header\n"
).encode("ascii")

# 3 floats (12 bytes) + 3 uchars (3 bytes) = 15 bytes per vertex
data = bytearray(n * 15)
for i in range(n):
    off = i * 15
    struct.pack_into("<fff", data, off, float(X[i]), float(Y[i]), float(Z[i]))
    data[off + 12] = int(R[i])
    data[off + 13] = int(G[i])
    data[off + 14] = int(B[i])

return header + bytes(data)
```

# ─────────────────────────────────────────────────────────────

# ROUTES

# ─────────────────────────────────────────────────────────────

@app.get(”/health”)
def health():
return {“status”: “ok”, “pipeline”: “depth-anything-v2 → point-cloud PLY”}

@app.post(”/process”)
async def process_image(image: UploadFile = File(…)):
if not image.content_type.startswith(“image/”):
raise HTTPException(400, “File must be an image (JPEG / PNG / WebP)”)

```
contents = await image.read()
if len(contents) > MAX_MB * 1024 * 1024:
    raise HTTPException(400, f"Image must be under {MAX_MB} MB")

try:
    img = Image.open(io.BytesIO(contents)).convert("RGB")

    # Cap input for speed
    w, h = img.size
    scale = min(1.0, 1024 / max(w, h))
    if scale < 1.0:
        img = img.resize((int(w * scale), int(h * scale)), Image.LANCZOS)

    depth = estimate_depth(img)
    ply_bytes = depth_to_ply(img, depth)

    return Response(
        content=ply_bytes,
        media_type="application/octet-stream",
        headers={
            "Content-Disposition": f'attachment; filename="memory_{uuid.uuid4().hex[:8]}.ply"'
        },
    )

except Exception as e:
    raise HTTPException(500, str(e))
```
