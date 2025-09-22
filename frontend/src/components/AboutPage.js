import React from 'react';
import './AboutPage.css';

export default function AboutPage({ onNavigateToHome }) {
  return (
    <div className="about-container">
      {/* Header */}
      <div className="about-header">
        <div className="about-nav">
          <button onClick={onNavigateToHome} className="nav-button home-button">
            🏠 Home
          </button>
          <h1 className="about-title">About India Policy Insights</h1>
        </div>
      </div>

      {/* Main Content */}
      <div className="about-main">
        {/* Hero Section */}
        <section className="about-hero">
          <div className="hero-content">
            <div className="hero-text">
              <h2 className="hero-title">India Policy Insights</h2>
              <p className="hero-description">
                India Policy Insights aims to empower policymakers with precision 
                health and development data that strengthens policy discussion and 
                action, helping to prioritize areas for intervention and improve the 
                quality of life for India's people and communities.
              </p>
              <div className="hero-cta">
                
              </div>
            </div>
            <div className="hero-image">
              <div className="image-placeholder">
              <img 
              src="/images/India.png" 
              alt="India" 
              className="image-placeholder"
              onError={(e) => {
                e.target.style.display = 'none';
                e.target.nextElementSibling.style.display = 'flex';
              }}
            />
              </div>
            </div>
          </div>
        </section>

        {/* Mission Statement */}
        <section className="mission-section">
          <div className="mission-banner">
            <div className="children-image-placeholder">
              
            </div>
            <div className="mission-overlay">
              <h3 className="mission-title">
                The current chatbot helps to democratize the health and development data collected at the district level in India, through the natural language query.
              </h3>
            </div>
          </div>
        </section>

        

        {/* Features Section */}
        <section className="features-section">
          <h3 className="section-title">What We Provide</h3>
          <div className="features-grid">
            <div className="feature-card">
              <div className="feature-icon">📊</div>
              <h4>Comprehensive Data Analysis</h4>
              <p>Access to 722 Districts with 122 Health Indicators covering 2016-2021 data</p>
            </div>
            <div className="feature-card">
              <div className="feature-icon">🗺️</div>
              <h4>Interactive Maps</h4>
              <p>Visualize health indicators across different geographic boundaries and administrative levels</p>
            </div>
            <div className="feature-card">
              <div className="feature-icon">🔍</div>
              <h4>District Insights</h4>
              <p>Deep dive into specific districts to understand local health challenges and opportunities</p>
            </div>
            <div className="feature-card">
              <div className="feature-icon">🤖</div>
              <h4>AI-Powered Analysis</h4>
              <p>Conversational analytics powered by artificial intelligence for intuitive data exploration</p>
            </div>
          </div>
        </section>

        {/* Call to Action */}
        <section className="cta-section">
          <div className="cta-content">
            <h3>Ready to Explore Health Data?</h3>
            <p>Start analyzing India's health indicators with our interactive platform</p>
            <button onClick={onNavigateToHome} className="cta-button-large">
              Access Health Analytics →
            </button>
          </div>
        </section>
      </div>

     
    </div>
  );
}
