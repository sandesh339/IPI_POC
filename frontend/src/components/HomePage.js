import React from 'react';
import './HomePage.css';

const HomePage = ({ onNavigateToChat, onNavigateToAbout }) => {
  return (
    <div className="homepage-container">
      {/* Background with gradient overlay */}
      <div className="homepage-background"></div>
      
      {/* Header with logos */}
      <header className="homepage-header">
        <div className="header-spacer"></div>
        <div className="university-logos">
          <div className="logo-container harvard-logo">
            <img 
              src="/images/Harvard-Logo.png" 
              alt="Harvard University" 
              className="university-logo"
              onError={(e) => {
                e.target.style.display = 'none';
                e.target.nextElementSibling.style.display = 'flex';
              }}
            />
            <div className="logo-placeholder harvard" style={{ display: 'none' }}>
              <span className="logo-text">HARVARD</span>
              <span className="logo-subtext">UNIVERSITY</span>
            </div>
            
          </div>
          
          <div className="logo-container vt-logo">
            <img 
              src="/images/Virginia-Tech-Logo.png" 
              alt="Virginia Tech" 
              className="university-logo"
              onError={(e) => {
                e.target.style.display = 'none';
                e.target.nextElementSibling.style.display = 'flex';
              }}
            />
            <div className="logo-placeholder vt" style={{ display: 'none' }}>
              <span className="logo-text">VIRGINIA</span>
              <span className="logo-subtext">TECH</span>
            </div>
            
          </div>
        </div>
        <div className="header-navigation">
          <button onClick={onNavigateToAbout} className="nav-button about-button">
            📋 About
          </button>
        </div>
      </header>

            {/* Hero Banner with Background Image */}
      <section 
        className="hero-banner"
        style={{
          backgroundImage: `url(${process.env.PUBLIC_URL}/images/Food%20Market-Original.jpg)`
        }}
      >
        <div className="hero-banner-overlay">
          <div className="hero-banner-content">
            <h1 className="hero-banner-title">Ask IPI</h1>
            <p className="hero-banner-subtitle">
              Comprehensive Health Indicators Analysis for Indian Districts
            </p>
          </div>
        </div>
      </section>

      {/* Main content */}
      <main className="homepage-main">
        <div className="hero-section">
          <div className="hero-content">
            <div className="cta-section">
              <button 
                className="start-analysis-btn"
                onClick={onNavigateToChat}
              >
                Access Chatbot Here
                <span className="btn-arrow">→</span>
              </button>
              
              <div className="data-info">
                <span className="data-badge">
                  <span className="badge-icon">🏥</span>
                  722 Districts • 122 Health Indicators • 2016-2021 Data
                </span>
              </div>
              
              <div className="description-container">
                <p className="main-description">
                  Analyze health indicator performance, create interactive visualizations, 
                  and discover insights from 2016 and 2021 health survey data.
                </p>
                
                <div className="features-grid">
                  <div className="feature-item">
                    <div className="feature-icon">📊</div>
                    <div className="feature-text">
                      <h3>Interactive Analytics</h3>
                      <p>Generate dynamic charts and visualizations</p>
                    </div>
                  </div>
                  
                  <div className="feature-item">
                    <div className="feature-icon">🗺️</div>
                    <div className="feature-text">
                      <h3>Geographic Insights</h3>
                      <p>Explore data through interactive maps</p>
                    </div>
                  </div>
                  
                  <div className="feature-item">
                    <div className="feature-icon">🏥</div>
                    <div className="feature-text">
                      <h3>Health Analytics</h3>
                      <p>Comprehensive health indicator analysis</p>
                    </div>
                  </div>
                </div>
              </div>
            </div>
          </div>
        </div>
      </main>

     
    </div>
  );
};

export default HomePage;
