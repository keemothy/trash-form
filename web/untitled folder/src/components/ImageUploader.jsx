import React, { useState } from 'react';

const API_URL = 'http://localhost:9000/predict';

export default function ImageUploader() {
  const [image, setImage] = useState(null);
  const [preview, setPreview] = useState(null);
  const [result, setResult] = useState('');
  const [loading, setLoading] = useState(false);

  const handleImageChange = (e) => {
    const file = e.target.files[0];
    setImage(file);
    setPreview(file ? URL.createObjectURL(file) : null);
    setResult('');
  };

  const handleSubmit = async (e) => {
    e.preventDefault();
    if (!image) return;
    setLoading(true);
    setResult('');
    const formData = new FormData();
    formData.append('image', image);
    try {
      const response = await fetch(API_URL, {
        method: 'POST',
        body: formData,
      });
      if (!response.ok) throw new Error('Prediction failed');
      const data = await response.json();
      setResult(`Prediction: ${data.class}`);
    } catch (err) {
      setResult('Error: ' + err.message);
    } finally {
      setLoading(false);
    }
  };

  return (
    <div style={{ maxWidth: 600, margin: '40px auto', background: 'rgba(255,255,255,0.95)', padding: 40, borderRadius: 16, boxShadow: '0 4px 24px #b3c6e0' }}>
      <h2 style={{ textAlign: 'center', color: '#2d6cdf', fontSize: '2.5em', fontWeight: 700, marginBottom: 10 }}>Trashform <span role="img" aria-label="leaf">🍃</span></h2>
      <p style={{ textAlign: 'center', color: '#4a6fa1', fontSize: '1.1em', marginBottom: 30 }}>
        Revolutionizing recycling with artificial intelligence. Identify materials, get recycling guidance, and make a positive environmental impact.
      </p>
      <div style={{ background: 'linear-gradient(90deg, #e3f0ff 0%, #f6faff 100%)', borderRadius: 12, padding: 32, border: '2px dashed #b3c6e0', textAlign: 'center', marginBottom: 24 }}>
        <form onSubmit={handleSubmit}>
          <label htmlFor="file-upload" style={{ display: 'block', marginBottom: 16 }}>
            <div style={{ display: 'flex', flexDirection: 'column', alignItems: 'center', cursor: 'pointer' }}>
              <div style={{ background: '#e3f0ff', borderRadius: '50%', width: 80, height: 80, display: 'flex', alignItems: 'center', justifyContent: 'center', marginBottom: 12, fontSize: 36, color: '#2d6cdf' }}>
                <span role="img" aria-label="upload">🖼️</span>
              </div>
              <span style={{ fontWeight: 500, color: '#2d6cdf', fontSize: '1.1em' }}>Drop your image here, or click to browse</span>
              <span style={{ color: '#7a8fa6', fontSize: '0.95em', marginTop: 4 }}>Supports: JPG, PNG, GIF up to 10MB</span>
            </div>
            <input id="file-upload" type="file" accept="image/*" onChange={handleImageChange} style={{ display: 'none' }} />
          </label>
          <button type="button" onClick={() => document.getElementById('file-upload').click()} style={{ background: '#2d6cdf', color: '#fff', border: 'none', borderRadius: 8, padding: '12px 28px', fontSize: '1.1em', fontWeight: 600, margin: '12px 8px 0 0', cursor: 'pointer', boxShadow: '0 2px 8px #b3c6e0', transition: 'background 0.2s' }}>
            Choose File
          </button>
          <button type="submit" disabled={loading || !image} style={{ background: '#4ad991', color: '#fff', border: 'none', borderRadius: 8, padding: '12px 28px', fontSize: '1.1em', fontWeight: 600, margin: '12px 0 0 8px', cursor: loading || !image ? 'not-allowed' : 'pointer', boxShadow: '0 2px 8px #b3c6e0', transition: 'background 0.2s' }}>
            {loading ? 'Classifying...' : 'Classify Image'}
          </button>
        </form>
        {preview && <img src={preview} alt="Preview" style={{ display: 'block', margin: '24px auto 0', maxWidth: '100%', maxHeight: 220, borderRadius: 10, boxShadow: '0 2px 8px #b3c6e0' }} />}
      </div>
      <div style={{ background: '#f6faff', borderRadius: 8, padding: 18, marginBottom: 24, color: '#2d6cdf', fontSize: '1em', boxShadow: '0 1px 4px #e3f0ff' }}>
        <strong>Tips for better recognition:</strong>
        <ul style={{ margin: '10px 0 0 18px', color: '#4a6fa1', fontSize: '0.98em' }}>
          <li>Ensure good lighting and clear focus</li>
          <li>Center the item in the frame</li>
          <li>Avoid shadows and reflections</li>
          <li>Clean the item surface if possible</li>
        </ul>
      </div>
      <div style={{ marginTop: 20, fontSize: '1.2em', textAlign: 'center', color: result.startsWith('Error') ? '#e74c3c' : '#2d6cdf', fontWeight: 500 }}>{result}</div>
    </div>
  );
}
