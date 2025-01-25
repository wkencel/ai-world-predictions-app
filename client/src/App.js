import React, { useState, useEffect } from 'react';
import axios from 'axios';

function App() {
  const [prediction, setPrediction] = useState('');
  const [prompt, setPrompt] = useState('');
  const [markets, setMarkets] = useState([]);
  const [loading, setLoading] = useState(false);

  useEffect(() => {
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
    <div className="container">
      <header>
        <h1>AI World Predictions</h1>
        <p className="subtitle">Bold Predictions for Tomorrow's Reality</p>
      </header>

      <main>
        {/* Prediction Section */}
        <section className="prediction-section">
          <h2>Make a Prediction</h2>
          <form onSubmit={handlePredict} className="prediction-form">
            <textarea
              value={prompt}
              onChange={(e) => setPrompt(e.target.value)}
              placeholder="Ask anything about future outcomes (sports, markets, events)..."
              className="prediction-input"
            />
            <button
              type="submit"
              disabled={loading}
              className={`predict-button ${loading ? 'loading' : ''}`}
            >
              {loading ? 'Generating Bold Prediction...' : 'Get Bold Prediction'}
            </button>
          </form>

          {prediction && (
            <div className="prediction-result">
              <h3>Bold Prediction</h3>
              <div className="result-box">
                <div className="prediction-content">
                  {prediction}
                </div>
                <div className="prediction-disclaimer">
                  This is a speculative prediction. Make your own financial decisions.
                </div>
              </div>
            </div>
          )}
        </section>

        {/* Markets Section */}
        <section className="markets-section">
          <h2>Available Markets</h2>
          <div className="markets-grid">
            {markets.map((market, index) => (
              <div key={index} className="market-card">
                <h3>{market.title || market.ticker}</h3>
                {market.description && <p>{market.description}</p>}
                <div className="market-footer">
                  <span className="market-ticker">{market.ticker}</span>
                  {market.status && (
                    <span className={`market-status ${market.status.toLowerCase()}`}>
                      {market.status}
                    </span>
                  )}
                </div>
              </div>
            ))}
          </div>
        </section>
      </main>
    </div>
  );
}

export default App;
