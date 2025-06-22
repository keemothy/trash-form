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
    <div className="modern-uploader-container" style={{ minHeight: '100vh', width: '100vw', display: 'flex', flexDirection: 'column', alignItems: 'center', justifyContent: 'center', background: 'none', boxShadow: 'none', padding: 0, margin: 0 }}>
      <div style={{ width: '100%', maxWidth: 600, background: 'rgba(255,255,255,0.98)', padding: 40, borderRadius: 24, boxShadow: '0 8px 32px #b3c6e0', margin: '32px 0' }}>
        <h2 style={{ textAlign: 'center', color: '#2d6cdf', fontSize: '2.7em', fontWeight: 800, marginBottom: 10, letterSpacing: '-1px', textShadow: '0 2px 8px #e3f0ff' }}>
          Trashform <span role="img" aria-label="leaf">🍃</span>
        </h2>
        <p style={{ textAlign: 'center', color: '#4a6fa1', fontSize: '1.15em', marginBottom: 30, fontWeight: 500 }}>
          Revolutionizing recycling with artificial intelligence. Identify materials, get recycling guidance, and make a positive environmental impact.
        </p>
        <div style={{ background: 'linear-gradient(90deg, #e3f0ff 0%, #f6faff 100%)', borderRadius: 16, padding: 32, border: '2.5px dashed #b3c6e0', textAlign: 'center', marginBottom: 28, boxShadow: '0 2px 12px #e3f0ff' }}>
          <form onSubmit={handleSubmit}>
            <label htmlFor="file-upload" style={{ display: 'block', marginBottom: 18 }}>
              <div style={{ display: 'flex', flexDirection: 'column', alignItems: 'center', cursor: 'pointer' }}>
                <div style={{ background: '#e3f0ff', borderRadius: '50%', width: 90, height: 90, display: 'flex', alignItems: 'center', justifyContent: 'center', marginBottom: 14, fontSize: 40, color: '#2d6cdf', boxShadow: '0 2px 8px #b3c6e0' }}>
                  <span role="img" aria-label="upload">🖼️</span>
                </div>
                <span style={{ fontWeight: 600, color: '#2d6cdf', fontSize: '1.15em' }}>Drop your image here, or click to browse</span>
                <span style={{ color: '#7a8fa6', fontSize: '0.98em', marginTop: 4 }}>Supports: JPG, PNG, GIF up to 10MB</span>
              </div>
              <input id="file-upload" type="file" accept="image/*" onChange={handleImageChange} style={{ display: 'none' }} />
            </label>
            <button type="button" onClick={() => document.getElementById('file-upload').click()} style={{ background: '#2d6cdf', color: '#fff', border: 'none', borderRadius: 10, padding: '14px 32px', fontSize: '1.13em', fontWeight: 700, margin: '12px 10px 0 0', cursor: 'pointer', boxShadow: '0 2px 8px #b3c6e0', transition: 'background 0.2s' }}>
              Choose File
            </button>
            <button type="submit" disabled={loading || !image} style={{ background: loading || !image ? '#b3c6e0' : '#4ad991', color: '#fff', border: 'none', borderRadius: 10, padding: '14px 32px', fontSize: '1.13em', fontWeight: 700, margin: '12px 0 0 10px', cursor: loading || !image ? 'not-allowed' : 'pointer', boxShadow: '0 2px 8px #b3c6e0', transition: 'background 0.2s' }}>
              {loading ? 'Classifying...' : 'Classify Image'}
            </button>
          </form>
          {preview && <img src={preview} alt="Preview" style={{ display: 'block', margin: '28px auto 0', maxWidth: '100%', maxHeight: 260, borderRadius: 14, boxShadow: '0 2px 12px #b3c6e0', border: '2px solid #e3f0ff' }} />}
        </div>
        <div style={{ background: '#f6faff', borderRadius: 12, padding: 22, marginBottom: 28, color: '#2d6cdf', fontSize: '1.08em', boxShadow: '0 1px 6px #e3f0ff', fontWeight: 500 }}>
          <strong>Tips for better recognition:</strong>
          <ul style={{ margin: '12px 0 0 20px', color: '#4a6fa1', fontSize: '1em' }}>
            <li>Ensure good lighting and clear focus</li>
            <li>Center the item in the frame</li>
            <li>Avoid shadows and reflections</li>
            <li>Clean the item surface if possible</li>
          </ul>
        </div>
        <div style={{ marginTop: 24, fontSize: '1.25em', textAlign: 'center', color: result.startsWith('Error') ? '#e74c3c' : '#2d6cdf', fontWeight: 600, minHeight: 32 }}>{result}</div>
      </div>
      <footer style={{ marginTop: 16, color: '#7a8fa6', fontSize: '1em', textAlign: 'center', letterSpacing: '0.02em' }}>
        &copy; {new Date().getFullYear()} Trashform. All rights reserved.
      </footer>
    </div>
  );
}
