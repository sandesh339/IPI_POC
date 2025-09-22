import { useState, useRef, useEffect, useCallback } from "react";
import SDGVisualization from "./SDGVisualization";
import DistrictClassification from "./DistrictClassification";
import IndividualDistrictAnalysis from "./IndividualDistrictAnalysis";
import StateWiseAnalysis from "./StateWiseAnalysis";
import CrossSDGAnalysis from "./CrossSDGAnalysis";
import NeighboringDistrictsAnalysis from "./NeighboringDistrictsAnalysis";
import StateWiseExtremes from "./StateWiseExtremes";
import ImprovementDistrictsAnalysis from "./ImprovementDistrictsAnalysis";
import BorderDistrictsAnalysis from "./BorderDistrictsAnalysis";
import RadiusAnalysis from "./RadiusAnalysis";
import TimeSeriesAnalysis from "./TimeSeriesAnalysis";
import { 
  saveMapVisualization,
  saveChartAsImage,
  saveElementAsImage,
  generateFilename,
  initializeMapForCapture,
  debugMapSaveReadiness,
  saveReactMapGLVisualization
} from '../utils/saveUtils';
import "./SaveVisualization.css";

// Chart.js registration is now handled in SDGVisualization component

export default function ConversationView({ onNavigateToHome }) {
  const [query, setQuery] = useState("");
  const [messages, setMessages] = useState([]);
  const [isLoading, setIsLoading] = useState(false);
  const chatContainerRef = useRef(null);
  const messagesRef = useRef([]);
  
  // Modal states
  const [modalOpen, setModalOpen] = useState(false);
  const [modalType, setModalType] = useState(null); // 'chart' or 'map'
  const [modalData, setModalData] = useState(null);
  
  // Save functionality states
  const [isSaving, setIsSaving] = useState(false);
  const [saveStatus, setSaveStatus] = useState(null); // 'success', 'error', null
  const [saveMessage, setSaveMessage] = useState('');
  const modalContentRef = useRef(null);
  const visualizationRef = useRef(null);
  
  // Audio transcription states
  const [isRecording, setIsRecording] = useState(false);
  const [audioBlob, setAudioBlob] = useState(null);
  const mediaRecorderRef = useRef(null);
  
  // Reaction states
  const [activeReactionBar, setActiveReactionBar] = useState(null);

  // Typewriter effect states
  const [isTyping, setIsTyping] = useState(false);
  const [currentTypingMessageId, setCurrentTypingMessageId] = useState(null);
  const typewriterTimeoutRef = useRef(null);

  console.log("ConversationView rendered, messages length:", messages.length);

  useEffect(() => {
    console.log("ConversationView mounted successfully");
  }, []);

  // Update messages ref when messages change
  useEffect(() => {
    messagesRef.current = messages;
  }, [messages]);

  useEffect(() => {
    if (chatContainerRef.current) {
      const scrollContainer = chatContainerRef.current;
      scrollContainer.scrollTo({
        top: scrollContainer.scrollHeight,
        behavior: "smooth",
      });
    }
  }, [messages]);

  // Close reaction bar when clicking outside
  useEffect(() => {
    const handleClickOutside = (event) => {
      if (activeReactionBar && !event.target.closest('.reaction-bar') && !event.target.closest('.reaction-trigger')) {
        setActiveReactionBar(null);
      }
    };

    document.addEventListener('click', handleClickOutside);
    return () => {
      document.removeEventListener('click', handleClickOutside);
    };
  }, [activeReactionBar]);

  // Cleanup typewriter timeout on unmount
  useEffect(() => {
    return () => {
      if (typewriterTimeoutRef.current) {
        clearTimeout(typewriterTimeoutRef.current);
      }
    };
  }, []);

  // Typewriter effect function - simplified to avoid stack overflow
  const typewriterEffect = useCallback((messageId, fullText, visualizations = null) => {
    setIsTyping(true);
    setCurrentTypingMessageId(messageId);
    
    let currentIndex = 0;
    const typingSpeed = 15; // milliseconds per character (adjust for speed)
    
    const typeNextCharacter = () => {
      if (currentIndex < fullText.length) {
        const partialText = fullText.substring(0, currentIndex + 1);
        
        setMessages(prev => prev.map(msg => 
          msg.id === messageId 
            ? { ...msg, text: partialText, isTyping: true }
            : msg
        ));
        
        currentIndex++;
        typewriterTimeoutRef.current = setTimeout(typeNextCharacter, typingSpeed);
      } else {
        // Typing complete
        setMessages(prev => prev.map(msg => 
          msg.id === messageId 
            ? { 
                ...msg, 
                text: fullText, 
                isTyping: false,
                visualizations: visualizations
              }
            : msg
        ));
        setIsTyping(false);
        setCurrentTypingMessageId(null);
      }
    };
    
    typeNextCharacter();
  }, []);

  const handleQuery = async () => {
    if (!query.trim() || isLoading) return;

    // If currently typing, stop the typewriter effect
    if (isTyping && typewriterTimeoutRef.current) {
      clearTimeout(typewriterTimeoutRef.current);
      setIsTyping(false);
      setCurrentTypingMessageId(null);
    }

    const newUserMessage = { 
      id: Date.now(), 
      text: query, 
      type: "user" 
    };
    
    setMessages(prev => [...prev, newUserMessage]);
    setQuery("");
    setIsLoading(true);

    try {
      const res = await fetch("http://127.0.0.1:8000/chatbot/", {
        method: "POST",
        headers: { 
          "Content-Type": "application/json",
          "Cache-Control": "no-cache, no-store, must-revalidate",
          "Pragma": "no-cache",
          "Expires": "0"
        },
        body: JSON.stringify({ 
          query,
          history: messages.map((msg) => ({
            role: msg.type === "user" ? "user" : "assistant",
            content: msg.text,
          })),
          timestamp: Date.now() // Add timestamp for cache busting
        }),
      });

      const data = await res.json();
      
      // Add data validation and logging for debugging
      console.log("API Response Data:", data);
      if (data.data && Array.isArray(data.data)) {
        console.log("Top 5 Districts from API:", data.data.slice(0, 5).map((d, i) => 
          `${i+1}. ${d.district || d.district_name} - ${d.indicator_value || d.performance_percentile}`
        ));
      }
      
      const messageId = Date.now() + 1;
      const visualizations = (data.boundary || data.data || data.chart_data) ? {
        boundary: data.boundary || [],
        data: data.data || data,
        chart_data: data.chart_data || null,
        function_calls: data.function_calls
      } : null;

      // Create initial message with empty text
      const newAssistantMessage = {
        id: messageId,
        text: "",
        type: "assistant",
        isTyping: true,
        visualizations: null // Will be added after typing completes
      };

      console.log("API Response:", data);
      console.log("Visualization data:", visualizations);

      setMessages(prev => [...prev, newAssistantMessage]);
      setIsLoading(false);
      
      // Start typewriter effect
      typewriterEffect(messageId, data.response || "Hello! How can I help you today?", visualizations);

    } catch (error) {
      setIsLoading(false);
      const errorMessageId = Date.now() + 1;
      const errorMessage = {
        id: errorMessageId,
        text: "",
        type: "assistant",
        isTyping: true
      };
      
      setMessages(prev => [...prev, errorMessage]);
      typewriterEffect(errorMessageId, "Error: Unable to reach chatbot. Please check your connection and try again.");
    }
  };

  // 🎤 Audio transcription functions
  const startRecording = async () => {
    setIsRecording(true);
    setAudioBlob(null);

    try {
      const stream = await navigator.mediaDevices.getUserMedia({ audio: true });
      const mediaRecorder = new MediaRecorder(stream);
      const audioChunks = [];

      mediaRecorder.ondataavailable = (event) => {
        if (event.data.size > 0) {
          audioChunks.push(event.data);
        }
      };

      mediaRecorder.onstop = () => {
        const audioBlob = new Blob(audioChunks, { type: "audio/wav" });
        setAudioBlob(audioBlob);
        sendAudioToOpenAI(audioBlob);
        // Stop all tracks to release microphone
        stream.getTracks().forEach(track => track.stop());
      };

      mediaRecorderRef.current = mediaRecorder;
      mediaRecorder.start();
    } catch (error) {
      console.error("Error accessing microphone:", error);
      setIsRecording(false);
    }
  };

  const stopRecording = () => {
    setIsRecording(false);
    if (mediaRecorderRef.current) {
      mediaRecorderRef.current.stop();
    }
  };

  const sendAudioToOpenAI = async (audioBlob) => {
    const formData = new FormData();
    formData.append("file", audioBlob, "audio.wav");
    formData.append("model", "whisper-1");

    try {
      const res = await fetch("https://api.openai.com/v1/audio/transcriptions", {
        method: "POST",
        headers: {
          Authorization: `Bearer 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`,
        },
        body: formData,
      });

      const data = await res.json();
      if (data.text) {
        setQuery(data.text);
        // Auto-send transcribed query
        setTimeout(() => handleQuery(), 100);
      }
    } catch (error) {
      console.error("Error transcribing audio:", error);
    }
  };

  // 👍 Reaction functions
  const handleReaction = async (messageId, reactionType) => {
    // Find the assistant message and corresponding user message
    const messageIndex = messages.findIndex(msg => msg.id === messageId);
    const assistantMsg = messages[messageIndex];
    const userMsg = messageIndex > 0 && messages[messageIndex - 1]?.type === "user" 
      ? messages[messageIndex - 1] : null;

    // Optimistically update UI
    const updatedMessages = [...messages];
    updatedMessages[messageIndex] = { ...assistantMsg, reaction: reactionType };
    setMessages(updatedMessages);

    // Send to backend
    try {
      await fetch("http://127.0.0.1:8000/store_reaction/", {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({
          user_query: userMsg ? userMsg.text : "",
          bot_response: assistantMsg.text,
          reaction_type: reactionType,
        }),
      });
    } catch (error) {
      console.error("Error storing reaction:", error);
    }
  };

  const toggleVisualization = (messageId, type) => {
    const messageData = messages.find(msg => msg.id === messageId)?.visualizations;
    setModalOpen(true);
    setModalType(type);
    setModalData(messageData);
  };

  const closeModal = useCallback(() => {
    setModalOpen(false);
    setModalType(null);
    setModalData(null);
    setSaveStatus(null);
    setSaveMessage('');
  }, []);

  // Save visualization functionality
  const handleSaveVisualization = async () => {
    if (!modalData) return;
    
    setIsSaving(true);
    setSaveStatus(null);
    setSaveMessage('');
    
    try {
      // Generate filename based on visualization type and data
      const visualizationType = modalType === 'chart' ? 'Chart' : 'Map';
      const filename = generateFilename(visualizationType, {
        sdg_goal_number: modalData.data?.[0]?.result?.sdg_goal_number || modalData.sdg_goal_number,
        indicator_name: modalData.data?.[0]?.result?.indicator_name || modalData.indicator_name,
        state_name: modalData.data?.[0]?.result?.state_name || modalData.state_name
      });
      
      console.log('🎯 Starting save process for:', filename);
      
      // Debug map instances before save
      if (window.mapInstances) {
        console.log('Available map instances:', Array.from(window.mapInstances.keys()));
      }
      
      let imageResult;
      
      // IMPROVED LOGIC: Detect actual visualization type based on available data
      // First, check if there's actually a map available to save
      const hasActualMapData = modalData.boundary && modalData.boundary.length > 0;
      const hasMapCanvas = document.querySelector('.modal-map-content .mapboxgl-map') || 
                          document.querySelector('.modal-map-content .map-container');
      
      // Only treat as map if we have both map data AND a map canvas
      const isActualMapVisualization = hasActualMapData && hasMapCanvas;
      
      console.log('🔍 Visualization type detection:', {
        modalType,
        hasActualMapData,
        hasMapCanvas: !!hasMapCanvas,
        isActualMapVisualization
      });
      
      if (isActualMapVisualization) {
        console.log('🗺️  Saving map visualization...');
        setSaveMessage('Capturing map visualization...');
        
        // Check if this is a neighboring districts analysis (React Map GL)
        const isNeighboringDistricts = isNeighboringDistrictsRequest(modalData);
        
        try {
          if (isNeighboringDistricts) {
            console.log('🏘️ Using React Map GL save method for neighboring districts...');
            imageResult = await saveReactMapGLVisualization('modal-map-content', filename);
          } else {
            // Use standard map save for other types
            imageResult = await saveMapVisualization('modal-map-content', filename);
          }
          setSaveStatus('success');
          setSaveMessage(`Map saved successfully as ${filename}.png`);
        } catch (error) {
          console.warn('Map save failed, using fallback:', error);
          
          // Fallback to element capture
          const visualizationElement = 
            document.querySelector('.modal-map-content .mapboxgl-map') ||
            document.querySelector('.modal-map-content .map-container') ||
            document.querySelector('.modal-map-content');
          
          if (visualizationElement) {
            imageResult = await saveElementAsImage(visualizationElement, filename);
            setSaveStatus('success');
            setSaveMessage(`Map saved successfully as ${filename}.png (fallback)`);
          } else {
            throw new Error('Map visualization element not found');
          }
        }
      } else {
        // Always save as chart if no actual map is available
        console.log('📊 Saving chart visualization...');
        setSaveMessage('Capturing chart visualization...');
        
        // Find chart canvases first (more precise approach)
        const allChartCanvases = document.querySelectorAll('.modal-chart-content canvas[role="img"], .modal-chart-content canvas');
        
        if (allChartCanvases.length > 1) {
          console.log(`Detected ${allChartCanvases.length} charts, saving all charts...`);
          
          // Find the smallest container that includes all charts but excludes modal UI
          const chartContainer = 
            document.querySelector('.modal-chart-content .charts-container') ||
            document.querySelector('.modal-chart-content .chart-grid') ||
            document.querySelector('.modal-chart-content .visualization-area') ||
            document.querySelector('.modal-chart-content > div:first-of-type');
          
          if (chartContainer) {
            const { saveMultipleCharts } = await import('../utils/saveUtils');
            imageResult = await saveMultipleCharts(chartContainer, filename);
            setSaveStatus('success');
            setSaveMessage(`Successfully saved ${allChartCanvases.length} charts as separate files`);
          } else {
            throw new Error('Chart container not found for multiple charts');
          }
        } else if (allChartCanvases.length === 1) {
          console.log('Single chart detected, saving canvas directly');
          const chartCanvas = allChartCanvases[0];
          
          // Save the canvas directly to avoid capturing any surrounding UI
          imageResult = await saveChartAsImage({ current: { canvas: chartCanvas } }, filename);
          setSaveStatus('success');
          setSaveMessage(`Chart saved successfully as ${filename}.png`);
        } else {
          // If no chart canvas found, look for chart-specific containers (avoid modal chrome)
          const chartContainer = 
            document.querySelector('.modal-chart-content .recharts-wrapper') ||
            document.querySelector('.modal-chart-content .chart-container') ||
            document.querySelector('.modal-chart-content .visualization-content') ||
            document.querySelector('.modal-chart-content [class*="chart"]') ||
            document.querySelector('.modal-chart-content > div > div'); // Go deeper to avoid modal header/footer
          
          if (chartContainer) {
            console.log('No canvas found, saving chart container (excluding modal UI)');
            imageResult = await saveElementAsImage(chartContainer, filename);
            setSaveStatus('success');
            setSaveMessage(`Visualization saved successfully as ${filename}.png`);
          } else {
            throw new Error('No chart content found to save');
          }
        }
      }
      
      // Save data to backend
      try {
        const { saveVisualizationToBackend } = await import('../utils/saveUtils');
        const backendResult = await saveVisualizationToBackend(modalData, {
          filename: filename,
          type: isActualMapVisualization ? 'Map' : 'Chart',
          modal_type: modalType,
          export_timestamp: new Date().toISOString()
        });
        console.log('Backend save result:', backendResult);
      } catch (backendError) {
        console.warn('Backend save failed:', backendError);
        // Don't fail the entire save process if backend save fails
      }
      
      // Auto-hide success message after 3 seconds
      setTimeout(() => {
        setSaveStatus(null);
        setSaveMessage('');
      }, 3000);
      
    } catch (error) {
      console.error('❌ Save failed:', error);
      setSaveStatus('error');
      setSaveMessage(`Error saving visualization: ${error.message}`);
      
      // Auto-hide error message after 5 seconds
      setTimeout(() => {
        setSaveStatus(null);
        setSaveMessage('');
      }, 5000);
    } finally {
      setIsSaving(false);
    }
  };

  // Function to close modal and trigger new query - stabilized with useCallback
  const closeModalAndQuery = useCallback(async (queryText) => {
    if (!queryText.trim() || isLoading) return;
    
    closeModal();
    
    // Create user message
    const newUserMessage = { 
      id: Date.now(), 
      text: queryText, 
      type: "user" 
    };
    
    setMessages(prev => [...prev, newUserMessage]);
    setIsLoading(true);

    try {
      const res = await fetch("http://127.0.0.1:8000/chatbot/", {
        method: "POST",
        headers: { 
          "Content-Type": "application/json",
          "Cache-Control": "no-cache, no-store, must-revalidate",
          "Pragma": "no-cache",
          "Expires": "0"
        },
        body: JSON.stringify({ 
          query: queryText,
          history: messagesRef.current.map((msg) => ({
            role: msg.type === "user" ? "user" : "assistant",
            content: msg.text,
          })),
          timestamp: Date.now() // Add timestamp for cache busting
        }),
      });

      const data = await res.json();
      
      const messageId = Date.now() + 1;
      const visualizations = (data.boundary || data.data || data.chart_data) ? {
        boundary: data.boundary || [],
        data: data.data || data,
        chart_data: data.chart_data || null,
        function_calls: data.function_calls
      } : null;

      const newAssistantMessage = {
        id: messageId,
        text: "",
        type: "assistant",
        isTyping: true,
        visualizations: null
      };

      setMessages(prev => [...prev, newAssistantMessage]);
      setIsLoading(false);
      
      // Use typewriter effect
      typewriterEffect(messageId, data.response || "Analysis complete!", visualizations);

    } catch (error) {
      setIsLoading(false);
      const errorMessageId = Date.now() + 1;
      const errorMessage = {
        id: errorMessageId,
        text: "",
        type: "assistant",
        isTyping: true
      };
      
      setMessages(prev => [...prev, errorMessage]);
      typewriterEffect(errorMessageId, "Error: Unable to process request. Please try again.");
    }
  }, [isLoading, closeModal, typewriterEffect]); // Only depend on stable values

  // Expose the function globally for modal components
  useEffect(() => {
    window.closeModalAndQuery = closeModalAndQuery;
    return () => {
      delete window.closeModalAndQuery;
    };
  }, [closeModalAndQuery]);

  // Listen for custom events from modal components
  useEffect(() => {
    const handleNewQuery = (event) => {
      const { query: queryText } = event.detail;
      closeModalAndQuery(queryText);
    };

    window.addEventListener('newQuery', handleNewQuery);
    return () => {
      window.removeEventListener('newQuery', handleNewQuery);
    };
  }, []);

  // Close modal on escape key
  useEffect(() => {
    const handleEscape = (event) => {
      if (event.key === 'Escape' && modalOpen) {
        closeModal();
      }
    };

    document.addEventListener('keydown', handleEscape);
    return () => {
      document.removeEventListener('keydown', handleEscape);
    };
  }, [modalOpen]);

  // Check if this is a classification request
  const isClassificationRequest = (visualizations) => {
    if (!visualizations) return false;
    
    const { function_calls, data } = visualizations;
    
    // Check for classification function calls
    if (function_calls) {
      const classificationFunctions = [
        "get_sdg_goal_classification",
        "get_aac_classification"
      ];
      
      if (function_calls.some(fc => classificationFunctions.includes(fc.function))) {
        return true;
      }
    }
    
    // Check for classification data structure patterns
    if (data) {
      // Check for function call results with classification data
      if (Array.isArray(data) && data.length > 0 && data[0].result) {
        const result = data[0].result;
        if (result.map_type === "sdg_goal_classification" || 
            result.map_type === "aac_classification" ||
            result.available_indicators ||
            result.indicator_data_map) {
          return true;
        }
      }
      
      // Check for direct classification data
      if (data.map_type === "sdg_goal_classification" || 
          data.map_type === "aac_classification" ||
          data.available_indicators ||
          data.indicator_data_map) {
        return true;
      }
    }
    
    return false;
  };

  // Check if this is a state-wise analysis request
  const isStateWiseRequest = (visualizations) => {
    if (!visualizations) return false;
    
    const { function_calls, data } = visualizations;
    
    // Check for state-wise function calls
    if (function_calls) {
      if (function_calls.some(fc => fc.function === "get_state_wise_summary")) {
        return true;
      }
    }
    
    // Check for state-wise data structure patterns
    if (data) {
      // Check for function call results with state-wise data
      if (Array.isArray(data) && data.length > 0 && data[0].result) {
        const result = data[0].result;
        if (result.analysis_type === "state_wise_summary" || 
            result.map_type === "state_wise_analysis") {
          return true;
        }
      }
      
      // Check for direct state-wise data
      if (data.analysis_type === "state_wise_summary" || 
          data.map_type === "state_wise_analysis") {
        return true;
      }
    }
    
    return false;
  };

  // Check if this is an individual district request
  const isIndividualDistrictRequest = (visualizations) => {
    if (!visualizations) return false;
    
    const { function_calls, data } = visualizations;
    
    // Check for individual district function calls
    if (function_calls) {
      const individualDistrictFunctions = [
        "get_individual_district_sdg_data",
        "get_district_indicator_selection_prompt",
        "get_best_worst_district_for_indicator"
      ];
      
      if (function_calls.some(fc => individualDistrictFunctions.includes(fc.function))) {
        return true;
      }
    }
    
          // Check for individual district data structure patterns
      if (data) {
        // Check for function call results with individual district data
        if (Array.isArray(data) && data.length > 0 && data[0].result) {
          const result = data[0].result;
          if (result.analysis_type === "individual_district" || 
              result.analysis_type === "best_worst_district" ||
              result.analysis_type === "overall_sdg_performance" ||
              result.map_type === "individual_district_analysis" ||
              result.map_type === "best_worst_district_analysis" ||
              result.map_type === "best_worst_overall_sdg" ||
              result.needs_indicator_selection ||
              (result.district && result.sdg_goals_data) ||
              (result.performance_type && (result.performance_type === "Best" || result.performance_type === "Worst"))) {
            return true;
          }
        }
        
        // Check for direct individual district data
        if (data.analysis_type === "individual_district" || 
            data.analysis_type === "best_worst_district" ||
            data.analysis_type === "overall_sdg_performance" ||
            data.map_type === "individual_district_analysis" ||
            data.map_type === "best_worst_district_analysis" ||
            data.map_type === "best_worst_overall_sdg" ||
            data.needs_indicator_selection ||
            (data.district && data.sdg_goals_data) ||
            (data.performance_type && (data.performance_type === "Best" || data.performance_type === "Worst"))) {
          return true;
        }
      }
    
    return false;
  };

  // Check if this is a cross-SDG analysis request
  const isCrossSDGRequest = (visualizations) => {
    if (!visualizations) return false;
    
    const { function_calls, data } = visualizations;
    
    // Check for cross-SDG function calls
    if (function_calls) {
      if (function_calls.some(fc => fc.function === "get_cross_sdg_analysis")) {
        return true;
      }
    }
    
    // Check for cross-SDG data structure patterns
    if (data) {
      // Check for function call results with cross-SDG data
      if (Array.isArray(data) && data.length > 0 && data[0].result) {
        const result = data[0].result;
        if (result.analysis_type === "cross_sdg_analysis" || 
            result.map_type === "cross_sdg_analysis") {
          return true;
        }
      }
      
      // Check for direct cross-SDG data
      if (data.analysis_type === "cross_sdg_analysis" || 
          data.map_type === "cross_sdg_analysis") {
        return true;
      }
    }
    
    return false;
  };

  // Check if this is a neighboring districts request
  const isNeighboringDistrictsRequest = (visualizations) => {
    if (!visualizations) return false;
    
    const { function_calls, data } = visualizations;
    
    // Check for neighboring districts function calls
    if (function_calls) {
      if (function_calls.some(fc => fc.function === "get_neighboring_districts_comparison")) {
        return true;
      }
    }
    
    // Check for neighboring districts data structure patterns
    if (data) {
      // Check for function call results with neighboring districts data
      if (Array.isArray(data) && data.length > 0 && data[0].result) {
        const result = data[0].result;
        if (result.map_type === "neighbor_comparison" || 
            (result.target_district && result.neighbors)) {
          return true;
        }
      }
      
      // Check for direct neighboring districts data
      if (data.map_type === "neighbor_comparison" || 
          (data.target_district && data.neighbors)) {
        return true;
      }
    }
    
    return false;
  };

  // Check if this is a state-wise extremes request
  const isStateWiseExtremesRequest = (visualizations) => {
    console.log('isStateWiseExtremesRequest called with:', {
      visualizations,
      hasVisualizations: !!visualizations,
      visualizationKeys: visualizations ? Object.keys(visualizations) : 'no viz'
    });

    if (!visualizations) {
      console.log('isStateWiseExtremesRequest: No visualizations');
      return false;
    }
    
    const { function_calls, data } = visualizations;
    
    // Check for state-wise extremes function calls
    if (function_calls) {
      const hasStateWiseExtremesFunction = function_calls.some(fc => fc.function === "get_state_wise_indicator_extremes");
      console.log('isStateWiseExtremesRequest: Function calls check:', {
        function_calls,
        hasStateWiseExtremesFunction
      });
      if (hasStateWiseExtremesFunction) {
        return true;
      }
    }
    
    // Check for state-wise extremes data structure patterns
    if (data) {
      console.log('isStateWiseExtremesRequest: Data structure check:', {
        dataType: typeof data,
        isArray: Array.isArray(data),
        dataKeys: typeof data === 'object' ? Object.keys(data) : 'not object'
      });

      // Check for function call results with state-wise extremes data
      if (Array.isArray(data) && data.length > 0 && data[0].result) {
        const result = data[0].result;
        console.log('isStateWiseExtremesRequest: Array result check:', {
          result,
          resultKeys: Object.keys(result),
          mapType: result.map_type,
          hasStateResults: !!result.state_results,
          hasIndicatorName: !!result.indicator_name
        });
        if (result.map_type === "state_wise_extremes" || 
            (result.state_results && result.indicator_name)) {
          console.log('isStateWiseExtremesRequest: Found in array result');
          return true;
        }
      }
      
      // Check for direct state-wise extremes data
      if (data.map_type === "state_wise_extremes" || 
          (data.state_results && data.indicator_name)) {
        console.log('isStateWiseExtremesRequest: Found in direct data');
        return true;
      }
    }
    
    console.log('isStateWiseExtremesRequest: No match found, returning false');
    return false;
  };

  // Check if this is an improvement districts request
  const isImprovementDistrictsRequest = (visualizations) => {
    console.log('isImprovementDistrictsRequest called with:', visualizations);

    if (!visualizations) {
      return false;
    }
    
    const { function_calls, data } = visualizations;
    
    // Check for improvement districts function calls
    if (function_calls) {
      const hasImprovementFunction = function_calls.some(fc => fc.function === "get_most_least_improved_districts");
      console.log('isImprovementDistrictsRequest: Function calls check:', {
        function_calls,
        hasImprovementFunction
      });
      if (hasImprovementFunction) {
        return true;
      }
    }
    
    // Check for improvement districts data structure patterns
    if (data) {
      // Check for function call results with improvement data
      if (Array.isArray(data) && data.length > 0 && data[0].result) {
        const result = data[0].result;
        console.log('isImprovementDistrictsRequest: Array result check:', {
          result,
          resultKeys: Object.keys(result),
          hasDistricts: !!result.districts,
          hasData: !!result.data,
          hasQueryType: !!result.query_type,
          queryType: result.query_type,
          mapType: result.map_type
        });
        
        // Check for improvement analysis map type
        if (result.map_type === 'improvement_analysis') {
          console.log('isImprovementDistrictsRequest: Found improvement_analysis map_type');
          return true;
        }
        
        // Check for districts with query_type
        if (result.districts && (result.query_type === 'most_improved' || result.query_type === 'least_improved')) {
          console.log('isImprovementDistrictsRequest: Found in array result with query_type');
          return true;
        }
        
        // Check for data array with districts (nested structure)
        if (result.data && Array.isArray(result.data) && result.data.length > 0) {
          const hasDistrictData = result.data.some(item => item.district_name || item.district);
          if (hasDistrictData && (result.query_type === 'most_improved' || result.query_type === 'least_improved')) {
            console.log('isImprovementDistrictsRequest: Found in nested data array');
            return true;
          }
        }
      }
      
      // Check for direct improvement data
      if (data.districts && (data.query_type === 'most_improved' || data.query_type === 'least_improved')) {
        console.log('isImprovementDistrictsRequest: Found in direct data');
        return true;
      }
      
      // Check for map_type improvement_analysis
      if (data.map_type === 'improvement_analysis') {
        console.log('isImprovementDistrictsRequest: Found direct improvement_analysis map_type');
        return true;
      }
    }
    
    console.log('isImprovementDistrictsRequest: No match found, returning false');
    return false;
  };

  // Check if this is a border districts request
  const isBorderDistrictsRequest = (visualizations) => {
    console.log('isBorderDistrictsRequest called with:', visualizations);

    if (!visualizations) {
      return false;
    }
    
    const { function_calls, data } = visualizations;
    
    // Check for border districts function calls
    if (function_calls) {
      const hasBorderFunction = function_calls.some(fc => fc.function === "get_border_districts");
      console.log('isBorderDistrictsRequest: Function calls check:', {
        function_calls,
        hasBorderFunction
      });
      if (hasBorderFunction) {
        return true;
      }
    }
    
    // Check for border districts data structure patterns
    if (data) {
      // Check for function call results with border districts data
      if (Array.isArray(data) && data.length > 0 && data[0].result) {
        const result = data[0].result;
        console.log('isBorderDistrictsRequest: Array result check:', {
          result,
          resultKeys: Object.keys(result),
          hasData: !!result.data,
          hasQueryType: !!result.query_type,
          queryType: result.query_type,
          mapType: result.map_type
        });
        
        // Check for border districts map type
        if (result.map_type === 'border_districts') {
          console.log('isBorderDistrictsRequest: Found border_districts map_type');
          return true;
        }
        
        // Check for border districts query type
        if (result.query_type === 'border_districts') {
          console.log('isBorderDistrictsRequest: Found border_districts query_type');
          return true;
        }
      }
      
      // Check for direct border districts data
      if (data.query_type === 'border_districts' || data.map_type === 'border_districts') {
        console.log('isBorderDistrictsRequest: Found direct border districts data');
        return true;
      }
    }
    
    console.log('isBorderDistrictsRequest: No match found, returning false');
    return false;
  };

  // Check if this is a radius analysis request
  const isRadiusAnalysisRequest = (visualizations) => {
    console.log('isRadiusAnalysisRequest called with:', visualizations);

    if (!visualizations) {
      return false;
    }
    
    const { function_calls, data } = visualizations;
    
    // Check for radius analysis function calls
    if (function_calls) {
      const hasRadiusFunction = function_calls.some(fc => fc.function === "get_districts_within_radius");
      console.log('isRadiusAnalysisRequest: Function calls check:', {
        function_calls,
        hasRadiusFunction
      });
      if (hasRadiusFunction) {
        return true;
      }
    }
    
    // Check for radius analysis data structure patterns
    if (data) {
      // Check for function call results with radius analysis data
      if (Array.isArray(data) && data.length > 0 && data[0].result) {
        const result = data[0].result;
        console.log('isRadiusAnalysisRequest: Array result check:', {
          result,
          resultKeys: Object.keys(result),
          hasData: !!result.data,
          hasQueryType: !!result.query_type,
          queryType: result.query_type,
          mapType: result.map_type
        });
        
        // Check for radius analysis map type
        if (result.map_type === 'radius_analysis') {
          console.log('isRadiusAnalysisRequest: Found radius_analysis map_type');
          return true;
        }
        
        // Check for radius analysis query type
        if (result.query_type === 'radius_analysis') {
          console.log('isRadiusAnalysisRequest: Found radius_analysis query_type');
          return true;
        }

        // Check for radius-specific properties
        if (result.center_point && result.radius_km && result.districts) {
          console.log('isRadiusAnalysisRequest: Found radius analysis properties');
          return true;
        }
      }
      
      // Check for direct radius analysis data
      if (data.query_type === 'radius_analysis' || data.map_type === 'radius_analysis') {
        console.log('isRadiusAnalysisRequest: Found direct radius analysis data');
        return true;
      }

      // Check for radius-specific properties in direct data
      if (data.center_point && data.radius_km && data.districts) {
        console.log('isRadiusAnalysisRequest: Found direct radius analysis properties');
        return true;
      }
    }
    
    console.log('isRadiusAnalysisRequest: No match found, returning false');
    return false;
  };

  const isTimeSeriesRequest = (visualizations) => {
    if (!visualizations) return false;
    
    // Check for time series specific data structure
    const hasTimeSeriesData = visualizations.data && Array.isArray(visualizations.data) && 
      visualizations.data.some(d => d.nfhs_4_value !== undefined || d.nfhs_5_value !== undefined);
    
    // Check for time series function calls (fix: check call.function)
    const hasTimeSeriesCall = visualizations.function_calls && 
      visualizations.function_calls.some(call => 
        typeof call.function === 'string' && (
          call.function.includes('get_time_series_comparison') || 
          call.function.includes('time_series') ||
          call.function.includes('annual_change')
        )
      );
    
    // Check for time series indicators in the data
    const hasTimeSeriesIndicators = visualizations.data && 
      visualizations.data.some(d => d.annual_change !== undefined || d.trend_direction !== undefined);
    
    return hasTimeSeriesData || hasTimeSeriesCall || hasTimeSeriesIndicators;
  };

  const hasChart = (visualizations) => {
    if (!visualizations) return false;
    
    // Check if the data contains chart-worthy information
    const { data, chart_data, function_calls } = visualizations;
    
    // If there's explicit chart data
    if (chart_data) return true;
    
    // If there's data that could be charted
    if (data) {
      if (Array.isArray(data) && data.length > 0) return true;
      if (data.districts || data.ranking_type || data.indicator_name) return true;
      
      // Check for classification data structure - function call results
      if (Array.isArray(data) && data.length > 0 && data[0].result) {
        const result = data[0].result;
        if (result.classification_summary || 
            result.map_type === "district_classification" ||
            result.map_type === "sdg_goal_classification" ||
            result.map_type === "aac_classification" ||
            result.map_type === "improvement_analysis" ||
            result.available_indicators ||
            result.districts ||
            (result.data && Array.isArray(result.data))) {
          return true;
        }
      }
      
      // Check for direct classification data structure (new format)
      if (data.map_type === "sdg_goal_classification" || 
          data.map_type === "aac_classification" ||
          data.map_type === "improvement_analysis" ||
          data.classification_summary || 
          data.available_indicators) {
        return true;
      }
    }
    
    // Check function calls for chart-worthy functions
    if (function_calls) {
      const chartSupportedFunctions = [
        "get_multiple_districts_and_indicators_data",
        "classify_districts_by_indicator", 
        "filter_districts_by_indicator",
        "get_district_indicator_data",
        "get_indicator_value_for_district",
        "get_sdg_goal_data",
        "get_sdg_goal_classification",
        "get_aac_classification",
        "get_individual_district_sdg_data",
        "get_best_worst_district_for_indicator",
        "get_cross_sdg_analysis",
        "get_state_wise_summary",
        "get_time_series_comparison",
        "get_aspirational_district_tracking",
        "get_neighboring_districts_comparison",
        "get_state_wise_indicator_extremes",
        "get_most_least_improved_districts",
          "get_border_districts",
          "get_districts_within_radius"
      ];
      
      if (function_calls.some(fc => chartSupportedFunctions.includes(fc.function))) {
        return true;
      }
    }
    
    return false;
  };

  // NEW: Helper function to determine if there's actual map data available
  const hasActualMapData = (visualizations) => {
    if (!visualizations) return false;
    
    // Check if there's boundary data available
    const hasBoundaryData = visualizations.boundary && visualizations.boundary.length > 0;
    
    // If there's boundary data, allow map display
    if (hasBoundaryData) {
      return true;
    }
    
    // Also check if it's a known map-specific request type that should have maps
    const isMapRequest = isClassificationRequest(visualizations) ||
                        isStateWiseRequest(visualizations) ||
                        isCrossSDGRequest(visualizations) ||
                        isNeighboringDistrictsRequest(visualizations) ||
                        isImprovementDistrictsRequest(visualizations) ||
                        isBorderDistrictsRequest(visualizations) ||
                        isRadiusAnalysisRequest(visualizations) ||
                        isTimeSeriesRequest(visualizations) ||
                        isIndividualDistrictRequest(visualizations);
    
    // Check for general SDG goal data requests (these often have maps)
    const isGeneralSDGRequest = visualizations.function_calls && 
      visualizations.function_calls.some(fc => 
        fc.function === "get_sdg_goal_data" || 
        fc.function === "get_overall_sdg_goal_data"
      );
    
    // Return true for known map request types or general SDG requests
    return isMapRequest || isGeneralSDGRequest;
  };

  const formatMessage = (text) => {
    if (!text) return "";
    
    try {
      // Decode Unicode escape sequences to proper emojis
      const decodeUnicode = (str) => {
        return str.replace(/\\u([0-9a-fA-F]{4})/g, (match, grp) => {
          return String.fromCharCode(parseInt(grp, 16));
        });
      };
      
      // Apply Unicode decoding first
      const decodedText = decodeUnicode(text);
      
      // Split text into lines and process each line
      const lines = decodedText.split('\n');
      const formattedLines = lines.map(line => {
        // Handle numbered lists (1. 2. 3. etc.)
        if (/^\d+\.\s/.test(line.trim())) {
          const match = line.match(/^\d+\./);
          return <div key={Math.random()} style={{ marginBottom: "8px", paddingLeft: "16px" }}>
            <strong>{match ? match[0] : ''}</strong> {line.replace(/^\d+\.\s/, '')}
          </div>;
        }
        
        // Handle bullet points (- or *)
        if (/^[-*]\s/.test(line.trim())) {
          return <div key={Math.random()} style={{ marginBottom: "6px", paddingLeft: "16px" }}>
            <span style={{ color: "#10a37f", fontWeight: "bold", marginRight: "8px" }}>•</span>
            {line.replace(/^[-*]\s/, '')}
          </div>;
        }
        
        // Handle bold text (**text**)
        const processedLine = line.replace(/\*\*(.*?)\*\*/g, '<strong>$1</strong>');
        
        // Return regular line
        return <div key={Math.random()} style={{ marginBottom: line.trim() === "" ? "12px" : "4px" }}>
          <span dangerouslySetInnerHTML={{ __html: processedLine }} />
        </div>;
      });
      
      return <div>{formattedLines}</div>;
    } catch (error) {
      console.error("Error formatting message:", error);
      return <div>{text}</div>;
    }
  };

  const openModal = (visualizations, type) => {
    setModalOpen(true);
    setModalType(type);
    setModalData(visualizations);
  };

  const renderVisualization = (visualizations) => {
    if (isTimeSeriesRequest(visualizations)) {
      return (
        <TimeSeriesAnalysis 
          data={visualizations.data} 
          boundary={visualizations.boundary || []} 
          chartOnly={false} 
          isModal={false}
        />
      );
    }
    // ... rest of the visualization rendering logic ...
  };

  // Enhanced map initialization when visualization data changes
  useEffect(() => {
    if (modalData && (modalData.boundary && modalData.boundary.length > 0 || modalData.boundary_data)) {
      console.log('🗺️  Modal data contains map boundary data');
      
      // Wait a bit for the map to render, then initialize for capture
      const initTimeout = setTimeout(() => {
        // Try to find and initialize any map instances
        const mapContainers = document.querySelectorAll('.mapboxgl-map');
        mapContainers.forEach((container, index) => {
          const mapInstance = container._map;
          if (mapInstance) {
            const mapId = `modal-map-${index}`;
            console.log(`Initializing map ${mapId} for capture`);
            initializeMapForCapture(mapInstance, mapId);
            
            // For RadiusAnalysis maps, add additional initialization
            if (isRadiusAnalysisRequest(modalData)) {
              console.log('🎯 Radius analysis map detected, adding extra initialization');
              setTimeout(() => {
                initializeMapForCapture(mapInstance, 'radius-analysis-map');
                initializeMapForCapture(mapInstance, 'modal-map-content');
                mapInstance.triggerRepaint();
              }, 1500);
            }
            
            // For BorderDistricts maps, add additional initialization
            if (isBorderDistrictsRequest(modalData)) {
              console.log('🏰 Border districts map detected, adding extra initialization');
              setTimeout(() => {
                initializeMapForCapture(mapInstance, 'border-districts-map');
                initializeMapForCapture(mapInstance, 'modal-map-content');
                mapInstance.triggerRepaint();
              }, 1500);
            }
            
            // For NeighboringDistricts maps, add additional initialization
            if (isNeighboringDistrictsRequest(modalData)) {
              console.log('🏘️ Neighboring districts map detected, adding extra initialization');
              setTimeout(() => {
                initializeMapForCapture(mapInstance, 'neighboring-districts-map');
                initializeMapForCapture(mapInstance, 'modal-map-content');
                mapInstance.triggerRepaint();
              }, 1500);
            }
            
            // Debug map readiness
            setTimeout(() => {
              debugMapSaveReadiness(mapInstance);
            }, 2000);
          }
        });
      }, 1000);
      
      return () => clearTimeout(initTimeout);
    }
  }, [modalData]);

  return (
    <div className="app-container">
      {/* Enhanced Header */}
      <div style={{
        borderBottom: "1px solid #e8d5b7",
        padding: "28px 40px",
        background: "linear-gradient(135deg, #ffffff 0%, #faf9f7 100%)",
        textAlign: "center",
        position: "relative",
        boxShadow: "0 2px 8px rgba(139, 69, 19, 0.06)"
      }}>
        <h1 style={{
          margin: 0,
          fontSize: "32px",
          fontWeight: "700",
          background: "linear-gradient(135deg, #5d4e37 0%, #4a3728 100%)",
          WebkitBackgroundClip: "text",
          WebkitTextFillColor: "transparent",
          backgroundClip: "text"
        }}>
          ASK SDG
        </h1>
        <p style={{
          margin: "10px 0 0 0",
          fontSize: "18px",
          color: "#8b7355",
          fontWeight: "400"
        }}>
          Conversational Analytics on India's Sustainable Development Goals Data
        </p>
        {/* Home Button */}
        {onNavigateToHome && (
          <button
            onClick={onNavigateToHome}
            style={{
              position: "absolute",
              top: "50%",
              left: "40px",
              transform: "translateY(-50%)",
              background: "linear-gradient(135deg, #3b82f6 0%, #dc2626 100%)",
              color: "white",
              border: "none",
              borderRadius: "12px",
              padding: "10px 16px",
              fontSize: "14px",
              fontWeight: "600",
              cursor: "pointer",
              display: "flex",
              alignItems: "center",
              gap: "8px",
              transition: "all 0.3s ease",
              boxShadow: "0 4px 16px rgba(59, 130, 246, 0.3)",
              zIndex: 10
            }}
            onMouseEnter={(e) => {
              e.target.style.transform = "translateY(-50%) translateY(-2px)";
              e.target.style.boxShadow = "0 6px 20px rgba(59, 130, 246, 0.4)";
            }}
            onMouseLeave={(e) => {
              e.target.style.transform = "translateY(-50%)";
              e.target.style.boxShadow = "0 4px 16px rgba(59, 130, 246, 0.3)";
            }}
            title="Return to Home Page"
          >
            <span>🏠</span>
            <span>Home</span>
          </button>
        )}

        {/* Enhanced User Avatar */}
        <div style={{
          position: "absolute",
          top: "50%",
          right: "40px",
          transform: "translateY(-50%)",
          width: "52px",
          height: "52px",
          borderRadius: "14px",
          background: "linear-gradient(135deg, #d4a574 0%, #c19a6b 50%, #b08d5b 100%)",
          display: "flex",
          alignItems: "center",
          justifyContent: "center",
          color: "white",
          fontWeight: "700",
          fontSize: "20px",
          boxShadow: "0 4px 16px rgba(212, 165, 116, 0.3)",
          border: "2px solid rgba(255, 255, 255, 0.3)",
          transition: "all 0.3s ease"
        }}>
          U
        </div>
      </div>

      {/* Conversation Container */}
      <div className="conversation-container" ref={chatContainerRef}>
        {/* Enhanced Welcome Message */}
        {messages.length === 0 && (
          <div className="conversation-message assistant" style={{
            background: "linear-gradient(135deg, #f5f3f0 0%, #e8d5b7 100%)",
            border: "2px solid #8b7355",
            margin: "32px 40px",
            borderRadius: "20px",
            boxShadow: "0 8px 24px rgba(139, 115, 85, 0.15)"
          }}>
            <div className="message-content">
              <div className="message-avatar assistant" style={{
                background: "linear-gradient(135deg, #8b7355 0%, #a0845c 50%, #b8956b 100%)",
                boxShadow: "0 6px 20px rgba(139, 115, 85, 0.4)",
                width: "42px",
                height: "42px",
                fontSize: "17px"
              }}>AI</div>
              <div className="message-text">
                <div style={{ 
                  marginBottom: "24px",
                  fontSize: "24px",
                  fontWeight: "700",
                  color: "#5d4e37"
                }}>
                SDG Assistant
                </div>
                
                <div style={{ 
                  background: "linear-gradient(135deg, #faf9f7 0%, #f5f3f0 100%)", 
                  padding: "24px", 
                  borderRadius: "16px", 
                  marginBottom: "24px",
                  border: "1px solid #e8d5b7",
                  boxShadow: "0 2px 8px rgba(139, 69, 19, 0.06)"
                }}>
                  <div style={{ 
                    fontWeight: "700", 
                    marginBottom: "16px", 
                    color: "#5d4e37",
                    fontSize: "18px",
                    display: "flex",
                    alignItems: "center",
                    gap: "10px"
                  }}>
                    Some example queries:
                  </div>
                  <div style={{ 
                    fontSize: "16px", 
                    color: "#6d5a47", 
                    lineHeight: "1.7",
                    fontWeight: "400"
                  }}>
                    • <strong>Show me the top 5 districts with highest diabetes prevalence in 2021</strong><br/>
                    • <strong>Show correlation analysis between SDG 3, 4, and 5 for 2021</strong><br/>
                    • <strong>Compare Delhi's malnutrition indicators with touching boundary districts</strong><br/>
                    • <strong>Show districts bordering Maharashtra with SDG 3 health indicators for 2021</strong> <br/>
                    • <strong>List all districts within 100 km of Delhi and their SDG 1 performance</strong> 
                  </div>
                </div>
                
              </div>
            </div>
          </div>
        )}
        
        {messages.map((message) => (
          <div key={message.id} className={`conversation-message ${message.type}`}>
            <div className="message-content">
              <div className={`message-avatar ${message.type}`}>
                {message.type === "user" ? "U" : "AI"}
              </div>
              <div className="message-text">
                <div style={{ 
                  whiteSpace: "pre-wrap", 
                  marginBottom: "12px",
                  lineHeight: "1.6",
                  color: "#374151",
                  position: "relative"
                }}>
                  {formatMessage(message.text)}
                  
                  {/* Show typewriter cursor for messages that are currently typing */}
                  {message.isTyping && (
                    <span style={{
                      display: "inline-block",
                      width: "2px",
                      height: "20px",
                      backgroundColor: "#8b7355",
                      marginLeft: "2px",
                      animation: "blink 1s infinite"
                    }}>
                    </span>
                  )}
                  
                  {/* Reaction buttons for assistant messages - only show when not typing */}
                  {message.type === "assistant" && !message.isTyping && (
                    <div style={{ 
                      position: "relative",
                      marginTop: "12px",
                      display: "flex",
                      alignItems: "center",
                      justifyContent: "flex-start"
                    }}>
                      {/* Only show reaction trigger if not already reacted */}
                      {!message.reaction && (
                         <button
                           className="reaction-trigger"
                           style={{ 
                             background: "linear-gradient(135deg, #f5f3f0 0%, #e8d5b7 100%)",
                             border: "1px solid #d4c4a8",
                             borderRadius: "20px",
                             padding: "6px 12px",
                             cursor: "pointer",
                             fontSize: "14px",
                             fontWeight: "500",
                             color: "#5d4e37",
                             display: "flex",
                             alignItems: "center",
                             gap: "6px",
                             transition: "all 0.3s ease",
                             boxShadow: "0 2px 8px rgba(139, 69, 19, 0.1)",
                             userSelect: "none",
                             position: "relative",
                             overflow: "hidden"
                           }}
                           onClick={() =>
                             setActiveReactionBar(activeReactionBar === message.id ? null : message.id)
                           }
                           title="Add Reaction"
                           onMouseEnter={(e) => {
                             e.target.style.background = "linear-gradient(135deg, #e8d5b7 0%, #d4c4a8 100%)";
                             e.target.style.transform = "translateY(-1px)";
                             e.target.style.boxShadow = "0 4px 12px rgba(139, 69, 19, 0.15)";
                           }}
                           onMouseLeave={(e) => {
                             e.target.style.background = "linear-gradient(135deg, #f5f3f0 0%, #e8d5b7 100%)";
                             e.target.style.transform = "translateY(0)";
                             e.target.style.boxShadow = "0 2px 8px rgba(139, 69, 19, 0.1)";
                           }}
                         >
                           <span style={{ fontSize: "16px" }}>😊</span>
                           <span>React</span>
                         </button>
                      )}
                      
                      {/* Enhanced floating reaction bar */}
                       {activeReactionBar === message.id && (
                         <div 
                           className="reaction-bar" 
                           style={{
                             position: "absolute",
                             top: "-60px",
                             left: "0",
                             zIndex: 1000,
                             background: "linear-gradient(135deg, #ffffff 0%, #fefdfb 100%)",
                             borderRadius: "25px",
                             boxShadow: "0 12px 40px rgba(139, 69, 19, 0.25)",
                             padding: "8px 4px",
                             border: "2px solid #e8d5b7",
                             backdropFilter: "blur(10px)",
                             display: "flex",
                             alignItems: "center",
                             gap: "4px",
                             animation: "reactionBarSlideIn 0.3s ease-out"
                           }}
                         >
                           <button
                             onClick={() => {
                               handleReaction(message.id, "like");
                               setActiveReactionBar(null);
                             }}
                             style={{ 
                               background: "transparent", 
                               border: "none", 
                               fontSize: "24px", 
                               cursor: "pointer", 
                               padding: "8px 10px",
                               borderRadius: "15px",
                               transition: "all 0.2s ease",
                               display: "flex",
                               alignItems: "center",
                               justifyContent: "center"
                             }}
                             title="Like this response"
                             onMouseEnter={(e) => {
                               e.target.style.backgroundColor = "#f0f9ff";
                               e.target.style.transform = "scale(1.2)";
                             }}
                             onMouseLeave={(e) => {
                               e.target.style.backgroundColor = "transparent";
                               e.target.style.transform = "scale(1)";
                             }}
                           >
                             👍
                           </button>
                           <button
                             onClick={() => {
                               handleReaction(message.id, "dislike");
                               setActiveReactionBar(null);
                             }}
                             style={{ 
                               background: "transparent", 
                               border: "none", 
                               fontSize: "24px", 
                               cursor: "pointer", 
                               padding: "8px 10px",
                               borderRadius: "15px",
                               transition: "all 0.2s ease",
                               display: "flex",
                               alignItems: "center",
                               justifyContent: "center"
                             }}
                             title="Dislike this response"
                             onMouseEnter={(e) => {
                               e.target.style.backgroundColor = "#fef2f2";
                               e.target.style.transform = "scale(1.2)";
                             }}
                             onMouseLeave={(e) => {
                               e.target.style.backgroundColor = "transparent";
                               e.target.style.transform = "scale(1)";
                             }}
                           >
                             👎
                           </button>
                           <button
                             onClick={() => {
                               handleReaction(message.id, "love");
                               setActiveReactionBar(null);
                             }}
                             style={{ 
                               background: "transparent", 
                               border: "none", 
                               fontSize: "24px", 
                               cursor: "pointer", 
                               padding: "8px 10px",
                               borderRadius: "15px",
                               transition: "all 0.2s ease",
                               display: "flex",
                               alignItems: "center",
                               justifyContent: "center"
                             }}
                             title="Love this response"
                             onMouseEnter={(e) => {
                               e.target.style.backgroundColor = "#fdf2f8";
                               e.target.style.transform = "scale(1.2)";
                             }}
                             onMouseLeave={(e) => {
                               e.target.style.backgroundColor = "transparent";
                               e.target.style.transform = "scale(1)";
                             }}
                           >
                             ❤️
                           </button>
                           <button
                             onClick={() => {
                               handleReaction(message.id, "helpful");
                               setActiveReactionBar(null);
                             }}
                             style={{ 
                               background: "transparent", 
                               border: "none", 
                               fontSize: "24px", 
                               cursor: "pointer", 
                               padding: "8px 10px",
                               borderRadius: "15px",
                               transition: "all 0.2s ease",
                               display: "flex",
                               alignItems: "center",
                               justifyContent: "center"
                             }}
                             title="Very helpful response"
                             onMouseEnter={(e) => {
                               e.target.style.backgroundColor = "#f0fdf4";
                               e.target.style.transform = "scale(1.2)";
                             }}
                             onMouseLeave={(e) => {
                               e.target.style.backgroundColor = "transparent";
                               e.target.style.transform = "scale(1)";
                             }}
                           >
                             ⭐
                           </button>
                         </div>
                      )}
                      
                      {/* Enhanced reaction summary */}
                      {message.reaction && (
                        <div style={{
                          display: "flex",
                          alignItems: "center",
                          gap: "8px",
                          padding: "8px 16px",
                          background: "linear-gradient(135deg, #f5f3f0 0%, #e8d5b7 100%)",
                          borderRadius: "20px",
                          border: "1px solid #d4c4a8",
                          boxShadow: "0 2px 8px rgba(139, 69, 19, 0.1)",
                          animation: "reactionSummarySlideIn 0.4s ease-out"
                        }}>
                          <span style={{ fontSize: "20px" }}>
                            {message.reaction === "like" ? "👍" : 
                             message.reaction === "dislike" ? "👎" 
                             : message.reaction === "love" ? "❤️" 
                             : message.reaction === "helpful" ? "⭐" 
                             : "👍"}
                          </span>
                          <span style={{ 
                            fontSize: "13px", 
                            color: "#5d4e37",
                            fontWeight: "600"
                          }}>
                            {message.reaction === "like" ? "Thanks for the feedback!" :
                             message.reaction === "dislike" ? "We'll improve next time" :
                             message.reaction === "love" ? "Glad you loved it!" :
                             message.reaction === "helpful" ? "Happy to help!" : "Thanks!"}
                          </span>
                        </div>
                      )}
                    </div>
                  )}
                </div>
                
                {/* Visualization Controls - only show when typing is complete */}
                {message.visualizations && !message.isTyping && (
                  <div className="visualization-controls">
                    {(() => {
                      const hasChartResult = hasChart(message.visualizations);
                      const hasMapResult = hasActualMapData(message.visualizations);
                      const isStateWiseExtremesResult = isStateWiseExtremesRequest(message.visualizations);
                      const isImprovementAnalysis = message.visualizations?.data?.[0]?.result?.map_type === 'improvement_analysis';
                      
                      console.log('Button display debug:', {
                        hasChartResult,
                        hasMapResult,
                        isStateWiseExtremesResult,
                        isImprovementAnalysis,
                        messageViz: message.visualizations,
                        mapType: message.visualizations?.data?.[0]?.result?.map_type,
                        hasBoundaryData: !!(message.visualizations?.boundary && message.visualizations.boundary.length > 0),
                        isIndividualDistrict: isIndividualDistrictRequest(message.visualizations),
                        functionCalls: message.visualizations?.function_calls
                      });
                      
                      // Show buttons based on what's actually available
                      const showChartButton = hasChartResult || isImprovementAnalysis;
                      const showMapButton = hasMapResult;
                      
                      if (!showChartButton && !showMapButton) {
                        return null;
                      }
                      
                      return (
                        <div style={{ marginBottom: '10px' }}>
                          {showChartButton && (
                            <button
                              className="expand-button"
                              onClick={() => openModal(message.visualizations, 'chart')}
                            >
                              📊 View Charts
                            </button>
                          )}
                          {showMapButton && (
                            <button
                              className="expand-button"
                              onClick={() => openModal(message.visualizations, 'map')}
                            >
                              🗺️ View Map
                            </button>
                          )}
                        </div>
                      );
                    })()}
                  </div>
                )}
              </div>
            </div>
          </div>
        ))}
        
        {/* Loading indicator */}
        {isLoading && (
          <div className="conversation-message assistant">
            <div className="message-content">
              <div className="message-avatar assistant">AI</div>
              <div className="message-text">
                <div style={{ 
                  opacity: 0.7, 
                  display: "flex", 
                  alignItems: "center", 
                  gap: "8px" 
                }}>
                  <div className="typing-indicator">
                    <span></span>
                    <span></span>
                    <span></span>
                  </div>
                  Analyzing your query...
                </div>
              </div>
            </div>
          </div>
        )}
      </div>

      {/* Input Section */}
      <div className="chat-input-section">
        <div className="chat-input-wrapper">
          <div className="chat-input-container">
            <textarea
              value={query}
              onChange={(e) => setQuery(e.target.value)}
              placeholder={isRecording ? "🎤 Listening... Speak your question" : "Message your AI health data assistant... (Press Enter to send, Shift+Enter for new line, 🎤 for voice)"}
              className="chat-input"
              onKeyDown={(e) => {
                if (e.key === 'Enter' && !e.shiftKey) {
                  e.preventDefault();
                  handleQuery();
                }
              }}
              disabled={isLoading}
              rows={1}
              style={{
                resize: "none",
                overflowY: query.length > 100 ? "auto" : "hidden"
              }}
              onInput={(e) => {
                e.target.style.height = "auto";
                e.target.style.height = Math.min(e.target.scrollHeight, 120) + "px";
              }}
            />
            {/* Microphone button */}
            {isRecording ? (
              <button 
                onClick={stopRecording} 
                className="chat-button"
                style={{
                  background: "linear-gradient(135deg, #d4a574 0%, #c19a6b 100%)",
                  marginLeft: "8px",
                  minWidth: "48px",
                  animation: "pulse 1.5s infinite"
                }}
                title="Stop recording"
              >
                ⏹️
              </button>
            ) : (
              <button 
                onClick={startRecording} 
                className="chat-button"
                disabled={isLoading}
                style={{
                  background: "linear-gradient(135deg, #a0845c 0%, #8b7355 100%)",
                  marginLeft: "8px",
                  minWidth: "48px"
                }}
                title="Voice input (Click to start recording)"
              >
                🎤
              </button>
            )}
            
            {/* Send button */}
            <button 
              onClick={handleQuery} 
              className="chat-button"
              disabled={isLoading || !query.trim()}
              title={isLoading ? "Processing..." : "Send message (Enter)"}
            >
              {isLoading ? (
                <div className="typing-indicator" style={{ transform: "scale(0.7)" }}>
                  <span></span>
                  <span></span>
                  <span></span>
                </div>
              ) : (
                "Send"
              )}
            </button>
          </div>
          <div style={{
            textAlign: "center",
            fontSize: "13px",
            color: "#8b7355",
            marginTop: "16px",
            opacity: 0.8,
            fontWeight: "500",
            letterSpacing: "0.5px"
          }}>
            ✨ AI-powered health data analysis • 📊 Interactive charts • 🗺️ Geographic insights
          </div>
        </div>
      </div>

      {/* Modal Popup */}
      {modalOpen && modalData && (
        <div className="modal-overlay" onClick={closeModal}>
          <div className="modal-container" onClick={(e) => e.stopPropagation()}>
            <div className="modal-header">
              <h3 className="modal-title">
                {modalType === 'chart' ? '📊 Interactive Chart View' : '🗺️ Geographic Map View'}
              </h3>
              <div className="modal-header-actions">
                {/* Save Status Message */}
                {saveStatus && (
                  <div className={`save-status ${saveStatus}`} style={{
                    padding: '8px 12px',
                    borderRadius: '6px',
                    fontSize: '14px',
                    fontWeight: '500',
                    marginRight: '16px',
                    backgroundColor: saveStatus === 'success' ? '#d4edda' : '#f8d7da',
                    color: saveStatus === 'success' ? '#155724' : '#721c24',
                    border: `1px solid ${saveStatus === 'success' ? '#c3e6cb' : '#f5c6cb'}`,
                    animation: 'fadeIn 0.3s ease-in-out'
                  }}>
                    {saveStatus === 'success' ? '✅' : '❌'} {saveMessage}
                  </div>
                )}
                
                {/* Save Button */}
                <button 
                  className="modal-save-button"
                  onClick={handleSaveVisualization}
                  disabled={isSaving}
                  style={{
                    background: isSaving ? '#6c757d' : 'linear-gradient(135deg, #28a745 0%, #20c997 100%)',
                    color: 'white',
                    border: 'none',
                    borderRadius: '8px',
                    padding: '10px 16px',
                    fontSize: '14px',
                    fontWeight: '600',
                    cursor: isSaving ? 'not-allowed' : 'pointer',
                    display: 'flex',
                    alignItems: 'center',
                    gap: '8px',
                    transition: 'all 0.3s ease',
                    boxShadow: '0 2px 4px rgba(40, 167, 69, 0.2)',
                    marginRight: '12px'
                  }}
                  title="Save visualization as image and data"
                >
                  {isSaving ? (
                    <>
                      <div className="spinner" style={{
                        width: '16px',
                        height: '16px',
                        border: '2px solid #ffffff',
                        borderTop: '2px solid transparent',
                        borderRadius: '50%',
                        animation: 'spin 1s linear infinite'
                      }}></div>
                      Saving...
                    </>
                  ) : (
                    <>
                      💾 Save
                    </>
                  )}
                </button>
                
                {/* Close Button */}
                <button className="modal-close-button" onClick={closeModal}>
                  ✕ Close
                </button>
              </div>
            </div>
            <div className="modal-content" ref={modalContentRef}>
              {modalType === 'chart' && hasChart(modalData) && (
                <div className="modal-chart-content">
                  <div className="visualization-container">
                    {(() => {
                    const isStateWiseExtremes = isStateWiseExtremesRequest(modalData);
                    const isStateWise = isStateWiseRequest(modalData);
                    const isIndividualDistrict = isIndividualDistrictRequest(modalData);
                    const isCrossSDG = isCrossSDGRequest(modalData);
                    const isNeighboringDistricts = isNeighboringDistrictsRequest(modalData);
                    const isClassification = isClassificationRequest(modalData);
                    const isImprovementDistricts = isImprovementDistrictsRequest(modalData);
                    const isBorderDistricts = isBorderDistrictsRequest(modalData);
                    console.log('modalData for radius analysis check:', modalData);
                    const isRadiusAnalysis = isRadiusAnalysisRequest(modalData);
                    console.log('isRadiusAnalysis result:', isRadiusAnalysis);
                    const isTimeSeries = isTimeSeriesRequest(modalData);
                    console.log('isTimeSeries result:', isTimeSeries);
                    
                    console.log('Modal component selection debug:', {
                      modalData,
                      isStateWiseExtremes,
                      isStateWise,
                      isIndividualDistrict,
                      isCrossSDG,
                      isNeighboringDistricts,
                      isClassification,
                      isImprovementDistricts,
                      isBorderDistricts,
                      modalDataKeys: modalData ? Object.keys(modalData) : 'no modal data',
                      functionCalls: modalData?.function_calls,
                      dataStructure: modalData?.data,
                      firstResultData: modalData?.data?.[0]?.result
                    });
                    
                    // Force improvement districts for debugging
                    if (modalData?.data?.[0]?.result?.map_type === 'improvement_analysis') {
                      console.log('FORCING improvement districts rendering due to improvement_analysis map_type');
                      
                      const extractedData = modalData.data?.[0]?.result || modalData.data;
                      console.log('Extracted data for processing:', extractedData);
                      
                      // Try multiple ways to find the districts data
                      let districts = [];
                      
                      if (extractedData.districts && Array.isArray(extractedData.districts)) {
                        districts = extractedData.districts;
                        console.log('Found districts in extractedData.districts', districts.length);
                      } else if (extractedData.data && Array.isArray(extractedData.data)) {
                        districts = extractedData.data;
                        console.log('Found districts in extractedData.data', districts.length);
                      } else if (modalData.data && Array.isArray(modalData.data) && modalData.data.length > 1) {
                        // Check if modalData.data has multiple items (not just the result wrapper)
                        districts = modalData.data.slice(1); // Skip the first item which might be the result wrapper
                        console.log('Found districts in modalData.data (skipping first)', districts.length);
                      }
                      
                      console.log('Final districts array:', districts);
                      console.log('Sample district:', districts[0]);
                      
                      // Handle the nested data structure where districts are in result.data
                      const improvementData = {
                        ...extractedData,
                        districts: districts,
                        boundary_data: modalData.boundary || modalData.boundary_data || extractedData.boundary_data || []
                      };
                      
                      console.log('FORCED improvement data for component:', improvementData);
                      
                      return (
                        <ImprovementDistrictsAnalysis 
                          improvementData={improvementData}
                          mapOnly={false}
                          chartOnly={true}
                        />
                      );
                    }
                    
                    if (isImprovementDistricts) {
                      console.log('Rendering ImprovementDistrictsAnalysis component with data:', {
                        modalData,
                        extractedData: modalData.data?.[0]?.result || modalData.data
                      });
                      
                      const extractedData = modalData.data?.[0]?.result || modalData.data;
                      
                      // Handle the nested data structure where districts are in result.data
                      const improvementData = {
                        ...extractedData,
                        districts: extractedData.districts || extractedData.data || [],
                        boundary_data: modalData.boundary || modalData.boundary_data || extractedData.boundary_data || []
                      };
                      
                      console.log('Final improvement data for component:', improvementData);
                      
                      return (
                        <ImprovementDistrictsAnalysis 
                          improvementData={improvementData}
                          mapOnly={false}
                          chartOnly={true}
                        />
                      );
                    } else if (isStateWiseExtremes) {
                      console.log('Rendering StateWiseExtremes component with data:', {
                        extremesData: {
                          ...(modalData.data?.[0]?.result || modalData.data),
                          boundary_data: modalData.boundary || modalData.boundary_data || (modalData.data?.[0]?.result?.boundary_data) || []
                        }
                      });
                      return (
                        <StateWiseExtremes 
                          extremesData={{
                            ...(modalData.data?.[0]?.result || modalData.data),
                            boundary_data: modalData.boundary || modalData.boundary_data || (modalData.data?.[0]?.result?.boundary_data) || []
                          }}
                          mapOnly={false}
                          chartOnly={true}
                        />
                      );
                    } else if (isStateWise) {
                      return (
                        <StateWiseAnalysis 
                          data={modalData} 
                          boundary={modalData.boundary || modalData.boundary_data || []}
                          chartOnly={true}
                          isModal={true}
                        />
                      );
                    } else if (isIndividualDistrict) {
                      return (
                        <IndividualDistrictAnalysis 
                          data={modalData} 
                          boundary={modalData.boundary || []}
                          chartOnly={true}
                          isModal={true}
                        />
                      );
                    } else if (isTimeSeries) {
                      console.log('Rendering TimeSeriesAnalysis component with data:', {
                        modalData,
                        extractedData: modalData.data?.[0]?.result || modalData.data
                      });
                      
                      return (
                        <TimeSeriesAnalysis 
                          data={modalData} 
                          boundary={modalData.boundary || modalData.boundary_data || []}
                          chartOnly={true}
                          isModal={true}
                        />
                      );
                    } else if (isCrossSDG) {
                      return (
                        <CrossSDGAnalysis 
                          data={modalData} 
                          boundary={modalData.boundary || modalData.boundary_data || []}
                          chartOnly={true}
                          isModal={true}
                        />
                      );
                    } else if (isBorderDistricts) {
                      console.log('Rendering BorderDistrictsAnalysis component with data:', {
                        modalData,
                        multipleResults: modalData.data,
                        boundaryLength: modalData.boundary?.length
                      });
                      
                      // Handle multiple function calls (e.g., Bihar + Goa)
                      let combinedData = [];
                      let allBoundaryData = modalData.boundary || modalData.boundary_data || [];
                      let targetState = '';
                      let sdgGoalNumber = null;
                      let year = 2021;
                      
                      if (Array.isArray(modalData.data) && modalData.data.length > 0) {
                        // Multiple function call results - combine them
                        modalData.data.forEach(functionResult => {
                          const result = functionResult.result;
                          if (result && result.data) {
                            combinedData = combinedData.concat(result.data);
                            if (!targetState) targetState = result.state;
                            if (!sdgGoalNumber) sdgGoalNumber = result.sdg_goal_number;
                            if (result.year) year = result.year;
                            
                            // Combine boundary data
                            if (result.boundary_data && Array.isArray(result.boundary_data)) {
                              allBoundaryData = allBoundaryData.concat(result.boundary_data);
                            }
                          }
                        });
                      } else {
                        // Single function call result
                      const extractedData = modalData.data?.[0]?.result || modalData.data;
                        combinedData = extractedData.data || [];
                        targetState = extractedData.state || '';
                        sdgGoalNumber = extractedData.sdg_goal_number;
                        year = extractedData.year || 2021;
                      }
                      
                      // Handle the border districts data structure
                      const borderData = {
                        success: true,
                        query_type: "border_districts",
                        state: targetState,
                        sdg_goal_number: sdgGoalNumber,
                        year: year,
                        data: combinedData,
                        boundary_data: allBoundaryData,
                        map_type: "border_districts"
                      };
                      
                      console.log('Final border data for component:', borderData);
                      
                      return (
                        <BorderDistrictsAnalysis 
                          borderData={borderData}
                          mapOnly={false}
                          chartOnly={true}
                        />
                      );
                    } else if (isRadiusAnalysis) {
                      console.log('Rendering RadiusAnalysis component with data:', {
                        modalData,
                        extractedData: modalData.data?.[0]?.result || modalData.data
                      });
                      
                      const extractedData = modalData.data?.[0]?.result || modalData.data;
                      
                      // Handle the radius analysis data structure
                      console.log('RadiusAnalysis Chart: Extracted data:', extractedData);
                      
                      const radiusData = {
                        ...extractedData,
                        districts: extractedData.districts || extractedData.data || [],
                        boundary_data: modalData.boundary || modalData.boundary_data || extractedData.boundary_data || []
                      };
                      
                      console.log('RadiusAnalysis Chart: Final radiusData:', radiusData);
                      console.log('RadiusAnalysis Chart: Districts count:', radiusData.districts.length);
                      
                      return (
                        <RadiusAnalysis 
                          radiusData={radiusData}
                          chartOnly={true}
                        />
                      );
                    } else if (isNeighboringDistricts) {
                      return (
                        <NeighboringDistrictsAnalysis 
                          data={modalData} 
                          boundary={modalData.boundary || modalData.boundary_data || []}
                          chartOnly={true}
                          isModal={true}
                        />
                      );
                    } else if (isClassification) {
                      return (
                        <DistrictClassification 
                          data={modalData} 
                          boundary={modalData.boundary || []}
                          chartOnly={true}
                        />
                      );
                    } else {
                      console.log('Falling back to SDGVisualization component');
                      return (
                        <SDGVisualization 
                          data={modalData} 
                          boundary={modalData.boundary || []}
                          chartOnly={true}
                        />
                      );
                    }
                  })()}
                  </div>
                </div>
              )}
              
              {modalType === 'map' && (
                <div className="modal-map-content">
                  <div className="visualization-container">
                    {modalData?.data?.[0]?.result?.map_type === 'improvement_analysis' ? (
                    (() => {
                      console.log('FORCING map rendering for improvement_analysis');
                      const extractedData = modalData.data?.[0]?.result || modalData.data;
                      
                      // Try multiple ways to find the districts data
                      let districts = [];
                      
                      if (extractedData.districts && Array.isArray(extractedData.districts)) {
                        districts = extractedData.districts;
                      } else if (extractedData.data && Array.isArray(extractedData.data)) {
                        districts = extractedData.data;
                      } else if (modalData.data && Array.isArray(modalData.data) && modalData.data.length > 1) {
                        districts = modalData.data.slice(1);
                      }
                      
                      const improvementData = {
                        ...extractedData,
                        districts: districts,
                        boundary_data: modalData.boundary || modalData.boundary_data || extractedData.boundary_data || []
                      };
                      
                      return (
                        <ImprovementDistrictsAnalysis 
                          improvementData={improvementData}
                          mapOnly={true}
                        />
                      );
                    })()
                  ) : isImprovementDistrictsRequest(modalData) ? (
                    (() => {
                      const extractedData = modalData.data?.[0]?.result || modalData.data;
                      const improvementData = {
                        ...extractedData,
                        districts: extractedData.districts || extractedData.data || [],
                        boundary_data: modalData.boundary || modalData.boundary_data || extractedData.boundary_data || []
                      };
                      
                      return (
                        <ImprovementDistrictsAnalysis 
                          improvementData={improvementData}
                          mapOnly={true}
                        />
                      );
                    })()
                  ) : isStateWiseRequest(modalData) ? (
                    <StateWiseAnalysis 
                      data={modalData} 
                      boundary={modalData.boundary || modalData.boundary_data || []}
                      chartOnly={false}
                      isModal={true}
                    />
                  ) : isIndividualDistrictRequest(modalData) ? (
                    <IndividualDistrictAnalysis 
                      data={modalData} 
                      boundary={modalData.boundary || []}
                      chartOnly={false}
                      isModal={true}
                    />
                  ) : isCrossSDGRequest(modalData) ? (
                    <CrossSDGAnalysis 
                      data={modalData} 
                      boundary={modalData.boundary || modalData.boundary_data || []}
                      chartOnly={false}
                      isModal={true}
                    />
                  ) : isBorderDistrictsRequest(modalData) ? (
                    (() => {
                      // Handle multiple function calls for map view too
                      let combinedData = [];
                      let allBoundaryData = modalData.boundary || modalData.boundary_data || [];
                      let targetState = '';
                      let sdgGoalNumber = null;
                      let year = 2021;
                      
                      if (Array.isArray(modalData.data) && modalData.data.length > 0) {
                        // Multiple function call results - combine them
                        modalData.data.forEach(functionResult => {
                          const result = functionResult.result;
                          if (result && result.data) {
                            combinedData = combinedData.concat(result.data);
                            if (!targetState) targetState = result.state;
                            if (!sdgGoalNumber) sdgGoalNumber = result.sdg_goal_number;
                            if (result.year) year = result.year;
                            
                            // Combine boundary data
                            if (result.boundary_data && Array.isArray(result.boundary_data)) {
                              allBoundaryData = allBoundaryData.concat(result.boundary_data);
                            }
                          }
                        });
                      } else {
                        // Single function call result
                      const extractedData = modalData.data?.[0]?.result || modalData.data;
                        combinedData = extractedData.data || [];
                        targetState = extractedData.state || '';
                        sdgGoalNumber = extractedData.sdg_goal_number;
                        year = extractedData.year || 2021;
                      }
                      
                      const borderData = {
                        success: true,
                        query_type: "border_districts",
                        state: targetState,
                        sdg_goal_number: sdgGoalNumber,
                        year: year,
                        data: combinedData,
                        boundary_data: allBoundaryData,
                        map_type: "border_districts"
                      };
                      
                      return (
                        <BorderDistrictsAnalysis 
                          borderData={borderData}
                          mapOnly={true}
                        />
                      );
                    })()
                  ) : isRadiusAnalysisRequest(modalData) ? (
                    (() => {
                      console.log('RadiusAnalysis: Processing modalData:', modalData);
                      
                      const extractedData = modalData.data?.[0]?.result || modalData.data;
                      console.log('RadiusAnalysis: Extracted data:', extractedData);
                      
                      const radiusData = {
                        ...extractedData,
                        districts: extractedData.districts || extractedData.data || [],
                        boundary_data: modalData.boundary || modalData.boundary_data || extractedData.boundary_data || []
                      };
                      
                      console.log('RadiusAnalysis: Final radiusData:', radiusData);
                      console.log('RadiusAnalysis: Districts count:', radiusData.districts.length);
                      
                      return (
                        <RadiusAnalysis 
                          radiusData={radiusData}
                          mapOnly={true}
                        />
                      );
                    })()
                  ) : isNeighboringDistrictsRequest(modalData) ? (
                    <NeighboringDistrictsAnalysis 
                      data={modalData} 
                      boundary={modalData.boundary || modalData.boundary_data || []}
                      chartOnly={false}
                      isModal={true}
                      mapOnly={true}
                    />
                  ) : isStateWiseExtremesRequest(modalData) ? (
                    <StateWiseExtremes 
                      extremesData={{
                        ...(modalData.data?.[0]?.result || modalData.data),
                        boundary_data: modalData.boundary || modalData.boundary_data || (modalData.data?.[0]?.result?.boundary_data) || []
                      }}
                      mapOnly={true}
                    />
                  ) : isClassificationRequest(modalData) ? (
                    <DistrictClassification 
                      data={modalData} 
                      boundary={modalData.boundary || []}
                      chartOnly={false}
                    />
                  ) : (
                    <SDGVisualization 
                      data={modalData} 
                      boundary={modalData.boundary || []}
                      chartOnly={false}
                      isModal={true}
                    />
                  )}
                  </div>
                </div>
              )}
            </div>
          </div>
        </div>
      )}
    </div>
  );
} 