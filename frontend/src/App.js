import React, { useState, useEffect } from 'react';
import HealthConversationView from "./components/HealthConversationView";
import HomePage from "./components/HomePage";
import AboutPage from "./components/AboutPage";

function App() {
  // Initialize view state from localStorage or default to 'home'
  const [currentView, setCurrentView] = useState(() => {
    try {
      const savedView = localStorage.getItem('healthApp_currentView');
      return savedView ? savedView : 'home';
    } catch (error) {
      console.error('Error loading view state from localStorage:', error);
      return 'home';
    }
  });

  // Persist view state to localStorage whenever it changes
  useEffect(() => {
    try {
      localStorage.setItem('healthApp_currentView', currentView);
    } catch (error) {
      console.error('Error saving view state to localStorage:', error);
    }
  }, [currentView]);

  const navigateToChat = () => {
    setCurrentView('chat');
  };

  const navigateToHome = () => {
    setCurrentView('home');
  };

  const navigateToAbout = () => {
    setCurrentView('about');
  };

  // Function to completely reset app state (for debugging or troubleshooting)
  const resetAppState = () => {
    try {
      localStorage.removeItem('healthApp_currentView');
      localStorage.removeItem('healthApp_chatMessages');
      setCurrentView('home');
      console.log('App state reset successfully');
    } catch (error) {
      console.error('Error resetting app state:', error);
    }
  };

  // Expose reset function globally for debugging (optional)
  useEffect(() => {
    window.resetHealthApp = resetAppState;
    return () => {
      delete window.resetHealthApp;
    };
  }, []);

  return (
    <div className="App">
      {currentView === 'home' ? (
        <HomePage onNavigateToChat={navigateToChat} onNavigateToAbout={navigateToAbout} />
      ) : currentView === 'about' ? (
        <AboutPage onNavigateToHome={navigateToHome} />
      ) : (
        <HealthConversationView onNavigateToHome={navigateToHome} />
      )}
    </div>
  );
}

export default App;
