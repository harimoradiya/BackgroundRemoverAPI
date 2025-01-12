from fastapi import FastAPI, File, UploadFile, HTTPException
from fastapi.responses import FileResponse, JSONResponse
from PIL import Image
import torch
from torchvision import transforms
from transformers import AutoModelForImageSegmentation
import io
import os
from datetime import datetime
import uuid
from pathlib import Path
import logging
from typing import Optional

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

app = FastAPI(title="Background Remover API", version="1.0.0")

# Create absolute paths for directories
BASE_DIR = Path(__file__).resolve().parent
UPLOAD_DIR = BASE_DIR / "temp_uploads"
OUTPUT_DIR = BASE_DIR / "temp_outputs"

# Create directories if they don't exist
UPLOAD_DIR.mkdir(parents=True, exist_ok=True)
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

logger.info(f"Upload directory: {UPLOAD_DIR}")
logger.info(f"Output directory: {OUTPUT_DIR}")

# Initialize model (done at startup)
@app.on_event("startup")
async def load_model():
    global model
    try:
        model = AutoModelForImageSegmentation.from_pretrained('briaai/RMBG-2.0', trust_remote_code=True)
        device = 'cuda' if torch.cuda.is_available() else 'cpu'
        model.to(device)
        if device == 'cuda':
            torch.set_float32_matmul_precision('high')
        model.eval()
        logger.info(f"Model loaded successfully on {device}")
    except Exception as e:
        logger.error(f"Failed to load model: {str(e)}")
        raise

# Configure image transformation
image_size = (1024, 1024)
transform_image = transforms.Compose([
    transforms.Resize(image_size),
    transforms.ToTensor(),
    transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
])

async def save_upload_file(upload_file: UploadFile, destination: Path) -> Path:
    try:
        with destination.open("wb") as buffer:
            content = await upload_file.read()
            buffer.write(content)
        return destination
    except Exception as e:
        logger.error(f"Error saving file: {str(e)}")
        raise

def process_image(image: Image.Image) -> Image.Image:
    """Process the image and remove background."""
    try:
        device = 'cuda' if torch.cuda.is_available() else 'cpu'
        input_images = transform_image(image).unsqueeze(0).to(device)
        
        with torch.no_grad():
            preds = model(input_images)[-1].sigmoid().cpu()
        
        pred = preds[0].squeeze()
        pred_pil = transforms.ToPILImage()(pred)
        mask = pred_pil.resize(image.size)
        image.putalpha(mask)
        return image
    except Exception as e:
        logger.error(f"Error processing image: {str(e)}")
        raise

@app.post("/remove-background/")
async def remove_background(file: UploadFile = File(...)):
    """
    Remove background from uploaded image.
    Returns processed image with transparent background.
    """
    if not file.content_type.startswith('image/'):
        return JSONResponse(
            status_code=400,
            content={"error": "File must be an image"}
        )

    try:
        # Create unique filenames
        unique_id = str(uuid.uuid4())
        input_path = UPLOAD_DIR / f"{unique_id}_input.png"
        output_path = OUTPUT_DIR / f"{unique_id}_output.png"

        # Save input file
        image_data = await file.read()
        image = Image.open(io.BytesIO(image_data))
        image = image.convert('RGB')
        image.save(input_path)
        
        # Process image
        processed_image = process_image(image)
        processed_image.save(output_path, "PNG")
        
        # Verify file exists
        if not output_path.exists():
            raise FileNotFoundError(f"Failed to save processed image at {output_path}")
        
        logger.info(f"Successfully processed image. Output saved at: {output_path}")
        
        return FileResponse(
            path=str(output_path),
            media_type="image/png",
            filename=f"removed_background_{unique_id}.png"
        )
        
    except Exception as e:
        logger.error(f"Error: {str(e)}")
        return JSONResponse(
            status_code=500,
            content={"error": "Failed to process image", "detail": str(e)}
        )
        
    finally:
        # Cleanup files
        try:
            if input_path.exists():
                input_path.unlink()
            logger.info("Cleaned up input file")
        except Exception as e:
            logger.error(f"Error cleaning up input file: {str(e)}")

@app.get("/health")
async def health_check():
    """Health check endpoint."""
    return {
        "status": "healthy",
        "gpu": torch.cuda.is_available(),
        "upload_dir": str(UPLOAD_DIR),
        "output_dir": str(OUTPUT_DIR)
    }

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)