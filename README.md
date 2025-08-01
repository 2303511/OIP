## Running Locally

1. Install LM Studio: https://lmstudio.ai
2. Download the model "DeepSeek-R1-0528-Qwen3-8B" in LM Studio.
3. Enable the LM Studio Local Server (Settings > Developer > Enable Local Server).
4. Run the backend:

# To be edited
Run backend 'uvicorn backend:app --reload --port 8000'
Run frotnend 'npm start'

To be edited for LLM engine portion -- need user to download locally and run the server.
```bash
cd backend
pip install -r requirements.txt
python app.py
