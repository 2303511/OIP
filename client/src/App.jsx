import React, { useState } from 'react';
import ReactMarkdown from 'react-markdown';
import './App.css';

function App() {
  const [file, setFile] = useState(null);
  const [crop, setCrop] = useState('tomato');
  const [result, setResult] = useState(null);
  const [loading, setLoading] = useState(false);
  const [imagePreview, setImagePreview] = useState(null);
  const [error, setError] = useState(null);

  const handleFileChange = (e) => {
    const selectedFile = e.target.files[0];
    setFile(selectedFile);
    setImagePreview(URL.createObjectURL(selectedFile));
    setResult(null);
    setError(null);
  };

  const handleUpload = async () => {
    if (!file) {
      setError('Please select an image first.');
      return;
    }

    const formData = new FormData();
    formData.append('file', file);
    formData.append('crop', crop);

    setLoading(true);
    setError(null);

    try {
      const res = await fetch('http://localhost:8000/predict', {
        method: 'POST',
        body: formData,
      });

      if (!res.ok) throw new Error('Server error. Please try again.');

      const data = await res.json();
      setResult(data);
    } catch (err) {
      setError(err.message);
    } finally {
      setLoading(false);
    }
  };

  return (
    <div className="app-container">
      <div className="card">
        <h1>🪲 Insect Identifier</h1>

        <input type="file" accept="image/*" onChange={handleFileChange} className="file-input" />

        <div className="image-preview">
          <img
            src={imagePreview || 'https://via.placeholder.com/400x250.png?text=Upload+Insect+Image'}
            alt="Preview"
          />
        </div>

        <label className="label">
          Crop Type:
          <select value={crop} onChange={(e) => setCrop(e.target.value)} className="select-input">
            <option value="tomato">Tomato</option>
            <option value="corn">Corn</option>
            <option value="soybean">Soybean</option>
          </select>
        </label>

        <button onClick={handleUpload} className="upload-btn">
          🚀 Upload & Identify
        </button>

        {loading && <p className="loading">🔍 Processing image...</p>}
        {error && <p className="error">⚠️ {error}</p>}

        {result && (
          <div className="result">
            <h3>✅ Identification Result</h3>
            <p><strong>Insect:</strong> {result.predicted_class}</p>
            <p><strong>Confidence:</strong> {Math.round(result.confidence * 100)}%</p>
            <div className="markdown">
              <strong>LLM Response:</strong>
              <ReactMarkdown>{result.llm_response}</ReactMarkdown>
            </div>
          </div>
        )}
      </div>
    </div>
  );
}

export default App;
