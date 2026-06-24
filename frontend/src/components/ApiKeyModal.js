import React, { useState } from "react";
import "./ApiKeyModal.css";

export default function ApiKeyModal({ onKeySubmit }) {
  const [inputKey, setInputKey] = useState("");
  const [showKey, setShowKey] = useState(false);
  const [error, setError] = useState("");

  const validate = (key) => {
    if (!key.trim()) return "API key cannot be empty.";
    if (!key.trim().startsWith("sk-")) return 'Key should start with "sk-". Check your OpenAI dashboard.';
    if (key.trim().length < 40) return "Key looks too short. Please paste the full key.";
    return "";
  };

  const handleSubmit = (e) => {
    e.preventDefault();
    const err = validate(inputKey);
    if (err) {
      setError(err);
      return;
    }
    setError("");
    onKeySubmit(inputKey.trim());
  };

  return (
    <div className="api-key-overlay">
      <div className="api-key-modal" role="dialog" aria-modal="true" aria-labelledby="api-key-title">
        <div className="api-key-modal-icon">🔑</div>

        <h2 id="api-key-title">Enter your OpenAI API Key</h2>
        <p>
          This app uses OpenAI to answer your health indicator questions.
          Your key is used only for this session and is never stored on our servers.
        </p>

        <form onSubmit={handleSubmit} noValidate>
          <div className="api-key-input-group">
            <label htmlFor="api-key-input">OpenAI API Key</label>
            <div className="api-key-input-wrapper">
              <input
                id="api-key-input"
                type={showKey ? "text" : "password"}
                value={inputKey}
                onChange={(e) => {
                  setInputKey(e.target.value);
                  if (error) setError("");
                }}
                placeholder="sk-proj-..."
                autoComplete="off"
                spellCheck={false}
              />
              <button
                type="button"
                className="api-key-toggle-btn"
                onClick={() => setShowKey((v) => !v)}
                aria-label={showKey ? "Hide key" : "Show key"}
              >
                {showKey ? "🙈" : "👁️"}
              </button>
            </div>
          </div>

          {error && (
            <div className="api-key-error" role="alert">
              ⚠️ {error}
            </div>
          )}

          <p className="api-key-hint">
            Don't have a key?{" "}
            <a href="https://platform.openai.com/api-keys" target="_blank" rel="noreferrer">
              Get one at platform.openai.com
            </a>
          </p>

          <div className="api-key-session-notice">
            <span>🔒</span>
            <span>
              Your key is stored only in browser session memory and cleared when you close the tab.
            </span>
          </div>

          <button
            type="submit"
            className="api-key-submit-btn"
            disabled={!inputKey.trim()}
          >
            Start Chatting →
          </button>
        </form>
      </div>
    </div>
  );
}
