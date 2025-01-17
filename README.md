# Background Remover API

A simple and efficient API for removing image backgrounds, built using FastAPI. This project provides a fast and user-friendly way to integrate background removal capabilities into your applications.

## Features

- Remove backgrounds from images with ease.
- Supports multiple image formats (e.g., JPEG, PNG).
- Fast and scalable API design using FastAPI.
- Easy-to-use endpoints for integration.
- Extensible for additional image processing features.

## Getting Started

### Prerequisites

- Python 3.9 or higher
- pip (Python package installer)

### Installation

1. Clone the repository:

   ```bash
   git clone https://github.com/harimoradiya/BackgroundRemoverAPI.git
   cd BackgroundRemoverAPI
   ```

2. Create and activate a virtual environment:

   ```bash
   python -m venv env
   source env/bin/activate  # On Windows use `env\\Scripts\\activate`
   ```

3. Install the required dependencies:

   ```bash
   pip install -r requirements.txt
   ```

4. Start the API server:

   ```bash
   uvicorn app.main:app --reload
   ```

   The API will be available at [http://127.0.0.1:8000](http://127.0.0.1:8000).

## Usage

### API Endpoints

#### 1. Upload an Image

**Endpoint:** `POST /remove-background`

**Description:** Removes the background from the uploaded image.

**Request:**

- Content-Type: `multipart/form-data`
- Body: 
  - `file`: Image file to process.

**Response:**

- `200 OK`: Returns the image with the background removed.
- `400 Bad Request`: If the input file is invalid or not supported.

**Example cURL Request:**

```bash
curl -X POST \
  -F "file=@path_to_your_image.jpg" \
  http://127.0.0.1:8000/remove-background
```

#### 2. Health Check

**Endpoint:** `GET /health`

**Description:** Returns the status of the API.

**Response:**

- `200 OK`: API is running.

### Response Example

```json
{
  "status": "success",
  "message": "Background removed successfully",
  "data": {
    "image_url": "http://127.0.0.1:8000/static/processed_image.png"
  }
}
```

## Project Structure

```
BackgroundRemoverAPI/
├── app/
│   ├── main.py          # Entry point for the API
├── requirements.txt      # Dependencies
└── README.md             # Project documentation
```

## Dependencies

- FastAPI
- Uvicorn
- Pillow
- rembg

Install all dependencies using:

```bash
pip install -r requirements.txt
```

## Contributing

Contributions are welcome! Please feel free to submit a Pull Request.

## License

This project is licensed under the MIT License. See the LICENSE file for details.

## Author

[Hari Moradiya](https://github.com/harimoradiya)

---

