import React, { useState, useEffect } from 'react';
import axios from 'axios';

function App() {
  const [prediction, setPrediction] = useState('');
  const [prompt, setPrompt] = useState('');
  const [markets, setMarkets] = useState([]);
  const [loading, setLoading] = useState(false);

  useEffect(() => {
    // Fetch initial markets data
    fetchMarkets();
  }, []);

  const fetchMarkets = async () => {
    try {
      const response = await axios.get('http://localhost:5000/kalshi/markets');
      setMarkets(response.data);
    } catch (error) {
      console.error('Error fetching markets:', error);
    }
  };

  const handlePredict = async (e) => {
    e.preventDefault();
    setLoading(true);
    try {
      const response = await axios.post('http://localhost:5000/predict', {
        prompt: prompt,
        mode: 'fast',
        timeframe: 'short'
      });
      setPrediction(response.data.prediction_result);
    } catch (error) {
      console.error('Error making prediction:', error);
      setPrediction('Error making prediction');
    }
    setLoading(false);
  };

  return (
    <div style={{ padding: '20px', maxWidth: '800px', margin: '0 auto' }}>
      <h1>AI World Predictions</h1>

      {/* Prediction Form */}
      <div style={{ marginBottom: '40px' }}>
        <h2>Make a Prediction</h2>
        <form onSubmit={handlePredict}>
          <textarea
            value={prompt}
            onChange={(e) => setPrompt(e.target.value)}
            placeholder="Enter your prediction prompt..."
            style={{ width: '100%', height: '100px', marginBottom: '10px' }}
          />
          <button type="submit" disabled={loading}>
            {loading ? 'Predicting...' : 'Get Prediction'}
          </button>
        </form>
        {prediction && (
          <div style={{ marginTop: '20px' }}>
            <h3>Prediction Result:</h3>
            <p>{prediction}</p>
          </div>
        )}
      </div>

      {/* Markets Display */}
      <div>
        <h2>Available Markets</h2>
        <div style={{ display: 'grid', gap: '20px' }}>
          {markets.map((market, index) => (
            <div key={index} style={{ border: '1px solid #ccc', padding: '10px' }}>
              <h3>{market.title || market.ticker}</h3>
              {market.description && <p>{market.description}</p>}
            </div>
          ))}
        </div>
      </div>
    </div>
  );
}

export default App;
