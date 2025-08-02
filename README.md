## Running Locally

1. Install LM Studio: https://lmstudio.ai
2. Download the model "DeepSeek-R1-0528-Qwen3-8B" in LM Studio.
3. Enable the LM Studio Local Server (Settings > Developer > Enable Local Server).
4. Run the backend:

# To be edited
Run backend 'uvicorn backend:app --reload --port 8000' -> backend dir
Run frotnend 'npm start' (Pre-requiste: run the command 'npm install') -> client dir


```bash
cd backend
pip install -r requirements.txt
uvicorn backend:app --reload --port 8000
```

Second Command Prompt
```bash
cd client
npm install
npm start

