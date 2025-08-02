// src/App.jsx
import React, { useState } from 'react';
import ReactMarkdown from "react-markdown";

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
    <div style={{ padding: 30, fontFamily: 'Arial', maxWidth: 600, margin: 'auto' }}>
      <h2>🌿 Insect Identifier</h2>

      <input type="file" accept="image/*" onChange={handleFileChange} /><br /><br />

      <img
        src={imagePreview || "https://via.placeholder.com/300x200.png?text=Upload+Insect+Image"}
        alt="Preview"
        style={{ width: '100%', maxHeight: 300, objectFit: 'contain' }}
      />

      <br /><br />
      <label>
        Crop type:&nbsp;
        <select value={crop} onChange={e => setCrop(e.target.value)}>
          <option value="tomato">Tomato</option>
          <option value="corn">Corn</option>
          <option value="soybean">Soybean</option>
        </select>
      </label>

      <br /><br />
      <button onClick={handleUpload}>Upload and Identify</button>

      {loading && <p>🔍 Processing image... please wait.</p>}
      {error && <p style={{ color: 'red' }}>⚠️ {error}</p>}

      {result && (
        <div style={{ marginTop: 20, border: '1px solid #ccc', padding: 20, borderRadius: 8 }}>
          <h3>✅ Identification Result</h3>
          <p><strong>Insect:</strong> {result.predicted_class}</p>
          <p><strong>Confidence:</strong> {Math.round(result.confidence * 100)}%</p>
          <p><strong>Crop Context:</strong> {result.crop_context_status}</p>
          <div>
            <strong>LLM Response:</strong>
            <ReactMarkdown>{result.llm_response}</ReactMarkdown>
          </div>
        </div>
      )}
    </div>
  );
}

export default App;
