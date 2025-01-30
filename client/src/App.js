import React, { useState, useEffect } from 'react';
import axios from 'axios';

const BACKEND_PORT = process.env.REACT_APP_BACKEND_API_PORT || 5000;
const BACKEND_URL = `http://localhost:${BACKEND_PORT}`;

function App() {
  const [predictions, setPredictions] = useState({
    fast: '',
    deep: '',
    council: ''
  });
  const [prompt, setPrompt] = useState('');
  const [markets, setMarkets] = useState([]);
  const [marketsError, setMarketsError] = useState(null);
  const [loading, setLoading] = useState({
    fast: false,
    deep: false,
    council: false
  });

  useEffect(() => {
    (async () => await fetchMarkets())();
  }, []);

  const fetchMarkets = async () => {
    try {
      const response = await axios.get(`${BACKEND_URL}/kalshi/markets`);
      if (response.data && Array.isArray(response.data)) {
        setMarkets(response.data);
        setMarketsError(null);
      } else {
        throw new Error('Invalid markets data format');
      }
    } catch (error) {
      console.error('Error fetching markets:', error);
      setMarketsError(error.response?.data?.message || 'Unable to load markets at this time');
      setMarkets([]);
    }
  };

  const handlePredict = async (e) => {
    e.preventDefault();

    // Reset predictions
    setPredictions({
      fast: '',
      deep: '',
      council: ''
    });

    // Make all three predictions in parallel
    const modes = ['fast', 'deep', 'council'];
    for (const mode of modes) {
      setLoading(prev => ({ ...prev, [mode]: true }));
      try {
        const response = await axios.post(`${BACKEND_URL}/predict`, {
          prompt: prompt,
          mode: mode,
          timeframe: 'short'
        });

        // Handle the response based on mode
        if (mode === 'council') {
          setPredictions(prev => ({
            ...prev,
            [mode]: {
              consensus: response.data.consensus,
              discussion: response.data.discussion
            }
          }));
        } else {
          setPredictions(prev => ({
            ...prev,
            [mode]: response.data.prediction_result
          }));
        }
      } catch (error) {
        console.error(`Error making ${mode} prediction:`, error);
        setPredictions(prev => ({
          ...prev,
          [mode]: `Error making ${mode} prediction: ${error.message}`
        }));
      }
      setLoading(prev => ({ ...prev, [mode]: false }));
    }
  };

  const renderPredictionBox = (mode) => (
    <div className={`prediction-box ${mode}`}>
      <h3>{mode.charAt(0).toUpperCase() + mode.slice(1)} Prediction</h3>
      <div className="result-box">
        <div className="prediction-content">
          {loading[mode] ? (
            <div className="loading-indicator">
              Generating {mode} prediction...
            </div>
          ) : predictions[mode] ? (
            mode === 'council' ? (
              <div>
                <div className="council-consensus">
                  <h4>Final Consensus</h4>
                  <p>{predictions[mode].consensus?.final_prediction || 'Consensus building...'}</p>
                  <p>Confidence: {predictions[mode].consensus?.confidence_level || 0}%</p>
                </div>

                {predictions[mode].discussion && predictions[mode].discussion.length > 0 && (
                  <div className="council-experts">
                    <h4>Expert Opinions</h4>
                    {predictions[mode].discussion.map((expert, index) => (
                      <div key={index} className="expert-opinion">
                        <h5>{expert.expert}</h5>
                        <div className="expert-analysis">
                          <p><strong>Prediction:</strong> {expert.analysis.prediction}</p>
                          <p><strong>Confidence:</strong> {expert.analysis.confidence}%</p>
                          {expert.analysis.factors && (
                            <div className="factors">
                              <p><strong>Key Factors:</strong></p>
                              <ul>
                                {expert.analysis.factors.map((factor, idx) => (
                                  <li key={idx}>{factor}</li>
                                ))}
                              </ul>
                            </div>
                          )}
                          {expert.analysis.risks && (
                            <div className="risks">
                              <p><strong>Risk Factors:</strong></p>
                              <ul>
                                {expert.analysis.risks.map((risk, idx) => (
                                  <li key={idx}>{risk}</li>
                                ))}
                              </ul>
                            </div>
                          )}
                        </div>
                      </div>
                    ))}
                  </div>
                )}
              </div>
            ) : (
              <div className="prediction-text">
                {predictions[mode]}
              </div>
            )
          ) : null}
        </div>
      </div>
    </div>
  );

  return (
    <div className="container">
      <header>
        <h1>AI World Predictions</h1>
        <p className="subtitle">Multi-Model Prediction Analysis</p>
      </header>

      <main>
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
              disabled={Object.values(loading).some(Boolean)}
              className="predict-button"
            >
              Get Predictions
            </button>
          </form>

          <div className="predictions-grid">
            {renderPredictionBox('fast')}
            {renderPredictionBox('deep')}
            {renderPredictionBox('council')}
          </div>
        </section>

        {/* Temporarily comment out markets section until backend is fixed */}
        {/* <section className="markets-section">...</section> */}
      </main>
    </div>
  );
}

export default App;
