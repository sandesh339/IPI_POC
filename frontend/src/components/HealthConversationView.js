import React, { useState, useRef, useEffect, useCallback } from "react";
import "../components/SaveVisualization.css";

// Import visualization components
import RadiusAnalysis from "./RadiusAnalysis";
import BorderDistrictsAnalysis from "./BorderDistrictsAnalysis";
import StateWiseExtremes from "./StateWiseExtremes";
import MultiDistrictAnalysis from "./MultiDistrictAnalysis";
import ConstraintBasedAnalysis from "./ConstraintBasedAnalysis";
import TopBottomAnalysis from "./TopBottomAnalysis";
import IndicatorChangeAnalysis from "./IndicatorChangeAnalysis";
import DistrictComparisonAnalysis from "./DistrictComparisonAnalysis";
import MultiIndicatorPerformance from "./MultiIndicatorPerformance";
import StateMultiIndicatorPerformance from "./StateMultiIndicatorPerformance";
import DistrictSimilarityAnalysis from "./DistrictSimilarityAnalysis";
import DistrictClassification from "./DistrictClassification";
import DistrictClassificationChange from "./DistrictClassificationChange";

// Import save utilities
import { 
  saveMapVisualization, 
  saveElementAsImage, 
  generateFilename 
} from "../utils/saveUtils";

export default function HealthConversationView({ onNavigateToHome }) {
  const [query, setQuery] = useState("");
  
  // Initialize messages from localStorage or empty array
  const [messages, setMessages] = useState(() => {
    try {
      const savedMessages = localStorage.getItem('healthApp_chatMessages');
      return savedMessages ? JSON.parse(savedMessages) : [];
    } catch (error) {
      console.error('Error loading chat messages from localStorage:', error);
      return [];
    }
  });
  
  const [isLoading, setIsLoading] = useState(false);
  const chatContainerRef = useRef(null);
  const messagesRef = useRef([]);
  
  // Modal states
  const [modalOpen, setModalOpen] = useState(false);
  const [modalType, setModalType] = useState(null);
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

  

  useEffect(() => {
    console.log("HealthConversationView mounted successfully");
  }, []);

  // Function to clear chat messages with confirmation
  const clearChat = () => {
    const confirmed = window.confirm(
      "Are you sure you want to clear all chat messages? This action cannot be undone."
    );
    
    if (confirmed) {
      setMessages([]);
      try {
        localStorage.removeItem('healthApp_chatMessages');
        console.log('Chat messages cleared successfully');
      } catch (error) {
        console.error('Error clearing chat messages from localStorage:', error);
      }
    }
  };

  // Update messages ref when messages change
  useEffect(() => {
    messagesRef.current = messages;
  }, [messages]);

  // Persist messages to localStorage whenever they change
  useEffect(() => {
    try {
      localStorage.setItem('healthApp_chatMessages', JSON.stringify(messages));
    } catch (error) {
      console.error('Error saving chat messages to localStorage:', error);
    }
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

  // Typewriter effect function
  const typewriterEffect = useCallback((messageId, fullText, visualizations = null) => {
    setIsTyping(true);
    setCurrentTypingMessageId(messageId);
    let currentIndex = 0;
    
    const typeNextCharacter = () => {
      if (currentIndex < fullText.length) {
        const currentText = fullText.substring(0, currentIndex + 1);
        
        setMessages(prev => prev.map(msg => 
          msg.id === messageId 
            ? { ...msg, text: currentText }
            : msg
        ));
        
        currentIndex++;
        typewriterTimeoutRef.current = setTimeout(typeNextCharacter, 20);
      } else {
        setIsTyping(false);
        setCurrentTypingMessageId(null);
        
        // Add visualizations after typing is complete
        if (visualizations) {
          setMessages(prev => prev.map(msg => 
            msg.id === messageId 
              ? { ...msg, visualizations: visualizations }
              : msg
          ));
        }
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
      const res = await fetch("https://ipi-poc.onrender.com/chatbot/", {
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
          timestamp: Date.now()
        }),
      });

      const data = await res.json();
      
      
      
      // Check if this is a radius analysis response
      if (data.districts && data.boundary_data) {
        
        if (data.districts && data.districts.length > 0) {
          const sample = data.districts[0];
         
        }
      } else {
        
        console.log("Available keys:", Object.keys(data));
      }

      
      const messageId = Date.now() + 1;
      // Handle different response structures for different functions
      let visualizations = null;

      if (data.districts && data.boundary_data) {
        
        // New structure for radius analysis
        visualizations = {
          boundary: data.boundary_data || [],
          boundary_data: data.boundary_data || [],
          data: data.districts || [],
          districts: data.districts || [],
          center_point: data.center_point,
          center_type: data.center_type,
          radius_km: data.radius_km,
          query_type: data.query_type,
          chart_data: data.chart_data || null,
          function_calls: data.function_calls
        };
      } else if (data.districts || data.boundary_data) {
        
        // Handle partial radius analysis data AND multi-district data
        visualizations = {
          boundary: data.boundary_data || data.boundary || [],
          boundary_data: data.boundary_data || data.boundary || [],
          data: data.districts || [],
          districts: data.districts || [],
          center_point: data.center_point,
          center_type: data.center_type,
          radius_km: data.radius_km,
          query_type: data.query_type,
          chart_data: data.chart_data || null,
          function_calls: data.function_calls,
          map_type: data.map_type,  // Add map_type here!
          // Include all fields from the API response for flattened structure
          ...data
        };
      } else if (data.boundary || data.data || data.chart_data || data.map_type) {
       
        // Handle all other function types including individual and multi-district
        visualizations = {
          boundary: data.boundary || [],
          data: data.data || data,
          chart_data: data.chart_data || null,
          function_calls: data.function_calls,
          map_type: data.map_type,
          // Include all fields from the API response for flattened structure
          ...data
        };
      }

      // Create initial message with empty text
      const newAssistantMessage = {
        id: messageId,
        text: "", 
        type: "assistant",
        visualizations: null
      };

      setMessages(prev => [...prev, newAssistantMessage]);

      // Start typewriter effect
      if (data.response) {
        typewriterEffect(messageId, data.response, visualizations);
      }

    } catch (error) {
      console.error("Error:", error);
      const errorMessage = {
        id: Date.now() + 1,
        text: "Sorry, I encountered an error while processing your request. Please try again.",
        type: "assistant"
      };
      setMessages(prev => [...prev, errorMessage]);
    } finally {
      setIsLoading(false);
    }
  };

  const handleKeyPress = (e) => {
    if (e.key === "Enter" && !e.shiftKey) {
      e.preventDefault();
      handleQuery();
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
          Authorization: `Bearer ${process.env.OPENAI_API_KEY}`,
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
    if (messageIndex === -1) return;

    const assistantMessage = messages[messageIndex];
    const userMessage = messageIndex > 0 ? messages[messageIndex - 1] : null;

    // Update the message with the reaction
    setMessages(prev => prev.map(msg => 
      msg.id === messageId 
        ? { ...msg, reaction: reactionType }
        : msg
    ));

    // Send feedback to backend (optional)
    try {
      await fetch("https://ipi-poc.onrender.com/feedback/", {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({
          message_id: messageId,
          reaction: reactionType,
          user_query: userMessage?.text || "",
          assistant_response: assistantMessage.text,
          timestamp: new Date().toISOString()
        }),
      });
    } catch (error) {
      console.log("Feedback not sent:", error);
    }
  };

  const handleSaveVisualization = async () => {
    if (!modalData || !modalContentRef.current) return;
    
    setIsSaving(true);
    setSaveStatus(null);
    setSaveMessage('');
    
    try {
      // Generate filename based on visualization type and data
      const baseFilename = generateFilename(
        modalType === 'map' ? 'health-map' : 'health-chart',
        {
          center: modalData.center_point,
          radius: modalData.radius_km,
          timestamp: new Date().toISOString().slice(0, 19).replace(/:/g, '-')
        }
      );
      
      let result;
      
      if (modalType === 'map') {
        // For maps, try to save using map-specific functionality
        try {
          result = await saveMapVisualization('modal-map-content', baseFilename);
        } catch (mapError) {
          console.warn('Map-specific save failed, falling back to element capture:', mapError);
          result = await saveElementAsImage(modalContentRef.current, baseFilename);
        }
      } else {
        // For charts, save the entire modal content
        result = await saveElementAsImage(modalContentRef.current, baseFilename);
      }
      
      if (result && result.success) {
        setSaveStatus('success');
        setSaveMessage(`✅ ${modalType === 'map' ? 'Map' : 'Chart'} downloaded as ${result.filename}!`);
        
        // Auto-hide success message after 4 seconds
        setTimeout(() => {
          setSaveStatus(null);
          setSaveMessage('');
        }, 4000);
      } else {
        throw new Error('Save operation did not return success');
      }
      
    } catch (error) {
      console.error('Save error:', error);
      setSaveStatus('error');
      setSaveMessage(`❌ Failed to download ${modalType}: ${error.message}`);
      
      // Auto-hide error message after 6 seconds
      setTimeout(() => {
        setSaveStatus(null);
        setSaveMessage('');
      }, 6000);
    } finally {
      setIsSaving(false);
    }
  };

  // Check request types for the 4 supported functions
  const isIndividualDistrictRequest = (visualizations) => {
    if (!visualizations) return false;
    
    const { function_calls, data } = visualizations;
    
    if (function_calls) {
      if (function_calls.some(fc => fc.function === "get_district_health_data")) {
        // Check if it's single district (individual_district map_type)
        const flattened = visualizations.map_type === "individual_district";
        if (flattened) {
          return true;
        }
        
        if (data && Array.isArray(data) && data.length > 0 && data[0].result) {
          const result = data[0].result;
          return result.map_type === "individual_district";
        }
      }
    }
    
    if (data && Array.isArray(data) && data.length > 0 && data[0].result) {
      const result = data[0].result;
      if (result.map_type === "individual_district") {
        return true;
      }
    }
    
    return false;
  };

  const isMultiDistrictRequest = (visualizations) => {
    if (!visualizations) return false;
    
    const { function_calls, data } = visualizations;
    
    if (function_calls) {
      if (function_calls.some(fc => fc.function === "get_district_health_data")) {
        // Check if it's multi district (multi_district_comparison map_type)
        const flattened = visualizations.map_type === "multi_district_comparison";
        if (flattened) return true;
        
        if (data && Array.isArray(data) && data.length > 0 && data[0].result) {
          const result = data[0].result;
          return result.map_type === "multi_district_comparison";
        }
      }
    }
    
    if (data && Array.isArray(data) && data.length > 0 && data[0].result) {
      const result = data[0].result;
      if (result.map_type === "multi_district_comparison") {
        return true;
      }
    }
    
    return false;
  };

  const isStateWiseExtremesRequest = (visualizations) => {
    if (!visualizations) return false;
    
    const { function_calls, data } = visualizations;
    
    if (function_calls) {
      if (function_calls.some(fc => fc.function === "get_state_wise_indicator_extremes")) {
        return true;
      }
    }
    
    if (data && Array.isArray(data) && data.length > 0 && data[0].result) {
      const result = data[0].result;
      if (result.map_type === "state_wise_extremes") {
        return true;
      }
    }
    
    return false;
  };

  const isBorderDistrictsRequest = (visualizations) => {
    if (!visualizations) return false;
    
    const { function_calls, data } = visualizations;
    
    if (function_calls) {
      if (function_calls.some(fc => fc.function === "get_border_districts")) {
        return true;
      }
    }
    
    if (data && Array.isArray(data) && data.length > 0 && data[0].result) {
      const result = data[0].result;
      if (result.map_type === "border_districts") {
        return true;
      }
    }
    
    return false;
  };

  const isRadiusAnalysisRequest = (visualizations) => {
    if (!visualizations) return false;
    
    const { function_calls, data } = visualizations;
    
    if (function_calls) {
      if (function_calls.some(fc => fc.function === "get_districts_within_radius")) {
        return true;
      }
    }
    
    if (data && Array.isArray(data) && data.length > 0 && data[0].result) {
      const result = data[0].result;
      if (result.map_type === "radius_analysis") {
        return true;
      }
    }
    
    return false;
  };

  const isConstraintBasedRequest = (visualizations) => {
    if (!visualizations) return false;
    
    const { function_calls, data, map_type } = visualizations;
    
    // Check if map_type indicates constraint-based search
    if (map_type === "constraint_based_search") {
      return true;
    }
    
    // Check function calls for get_districts_by_constraints
    if (function_calls) {
      if (function_calls.some(fc => fc.function === "get_districts_by_constraints")) {
        return true;
      }
    }
    
    // Check for constraint-specific fields
    if (visualizations.constraints_applied && Array.isArray(visualizations.constraints_applied)) {
      return true;
    }
    
    if (data && Array.isArray(data) && data.length > 0 && data[0].result) {
      const result = data[0].result;
      if (result.map_type === "constraint_based_search") {
        return true;
      }
    }
    
    return false;
  };

  const isTopBottomDistrictsRequest = (visualizations) => {
    if (!visualizations) return false;
    
    const { function_calls, data, map_type } = visualizations;
    
    // Check if map_type indicates top/bottom districts analysis
    if (map_type === "top_bottom_districts") {
      return true;
    }
    
    // Primary detection: Check function calls for get_top_bottom_districts
    if (function_calls) {
      if (function_calls.some(fc => fc.function === "get_top_bottom_districts")) {
        return true;
      }
    }
    
    // More restrictive check: only if we have performance_type AND don't have state-specific indicators
    if (visualizations.performance_type && 
        (visualizations.performance_type === "top" || visualizations.performance_type === "bottom" || visualizations.performance_type === "both") &&
        // Ensure this is NOT a state multi-indicator request by checking for absence of state-specific data
        !visualizations.states && !visualizations.state_districts) {
      return true;
    }
    
    if (data && Array.isArray(data) && data.length > 0 && data[0].result) {
      const result = data[0].result;
      if (result.map_type === "top_bottom_districts") {
        return true;
      }
    }
    
    return false;
  };

  const isIndicatorChangeAnalysisRequest = (visualizations) => {
    if (!visualizations) return false;
    
    const { function_calls, data, map_type, response_type } = visualizations;
    
    // Check if map_type indicates indicator change analysis
    if (map_type === "indicator_change_analysis") {
      return true;
    }
    
    // Check function calls for get_indicator_change_analysis
    if (function_calls) {
      if (function_calls.some(fc => fc.function === "get_indicator_change_analysis")) {
        return true;
      }
    }
    
    // Check for change analysis specific fields
    if (response_type && (response_type === "country_change_analysis" || response_type === "state_change_analysis" || response_type === "district_change_analysis")) {
      return true;
    }
    
    // Check for analysis_level field
    if (visualizations.analysis_level && (visualizations.analysis_level === "country" || visualizations.analysis_level === "state" || visualizations.analysis_level === "district")) {
      return true;
    }
    
    if (data && Array.isArray(data) && data.length > 0 && data[0].result) {
      const result = data[0].result;
      if (result.map_type === "indicator_change_analysis" || result.response_type?.includes("change_analysis")) {
        return true;
      }
    }
    
    return false;
  };

  const isDistrictComparisonRequest = (visualizations) => {
    if (!visualizations) return false;
    
    const { function_calls, data, map_type, response_type } = visualizations;
    
    // Check if response_type indicates district performance comparison
    if (response_type === "district_performance_comparison") {
      return true;
    }
    
    // Check if map_type indicates district comparison
    if (map_type === "district_comparison") {
      return true;
    }
    
    // Check function calls for get_district_performance_comparison
    if (function_calls) {
      if (function_calls.some(fc => fc.function === "get_district_performance_comparison")) {
        return true;
      }
    }
    
    // Check nested data structure
    if (data && Array.isArray(data) && data.length > 0 && data[0].result) {
      const result = data[0].result;
      if (result.response_type === "district_performance_comparison" || result.map_type === "district_comparison") {
        return true;
      }
    }
    
    return false;
  };

  const isStateMultiIndicatorPerformanceRequest = (visualizations) => {
    if (!visualizations) return false;
    
    const { function_calls, data, map_type, response_type } = visualizations;
    
    // Check if response_type indicates state multi-indicator performance
    if (response_type === "state_multi_indicator_performance") {
      return true;
    }
    
    // Check if map_type indicates state multi-indicator performance
    if (map_type === "state_district_multi_indicator") {
      return true;
    }
    
    // Primary detection: Check function calls for get_state_multi_indicator_performance
    if (function_calls) {
      if (function_calls.some(fc => fc.function === "get_state_multi_indicator_performance")) {
        return true;
      }
    }
    
    // Check nested data structure for explicit state indicators
    if (data && Array.isArray(data) && data.length > 0 && data[0].result) {
      const result = data[0].result;
      if (result.response_type === "state_multi_indicator_performance" || 
          result.map_type === "state_district_multi_indicator") {
        return true;
      }
      
      // More specific check: must have both states AND state_districts data structure
      if (result.states && Array.isArray(result.states) && 
          result.state_districts && typeof result.state_districts === 'object') {
        return true;
      }
    }
    
    // Direct data structure check (but only if very specific)
    if (visualizations.states && Array.isArray(visualizations.states) && 
        visualizations.state_districts && typeof visualizations.state_districts === 'object' &&
        // Additional verification: check if we have multiple states or state-district structure
        (visualizations.states.length > 0 && Object.keys(visualizations.state_districts).length > 0)) {
      return true;
    }
    
    return false;
  };

  const isDistrictSimilarityAnalysisRequest = (visualizations) => {
    if (!visualizations) return false;
    
    const { function_calls, data, map_type, response_type } = visualizations;
    
    // Check if response_type indicates district similarity analysis
    if (response_type === "district_similarity_analysis") {
      return true;
    }
    
    // Check if map_type indicates district similarity analysis
    if (map_type === "district_similarity") {
      return true;
    }
    
    // Check function calls for get_district_similarity_analysis
    if (function_calls) {
      if (function_calls.some(fc => fc.function === "get_district_similarity_analysis")) {
        return true;
      }
    }
    
    // Check nested data structure
    if (data && Array.isArray(data) && data.length > 0 && data[0].result) {
      const result = data[0].result;
      if (result.response_type === "district_similarity_analysis" || result.map_type === "district_similarity") {
        return true;
      }
    }
    
    // Direct data structure check
    if (visualizations.analysis_type && 
        (visualizations.analysis_type === "similar" || visualizations.analysis_type === "different") &&
        visualizations.districts && Array.isArray(visualizations.districts)) {
      return true;
    }
    
    return false;
  };

  const isDistrictClassificationRequest = (visualizations) => {
    if (!visualizations) return false;
    
    const { function_calls, data, map_type, response_type } = visualizations;
    
    // Exclude change classification requests
    if (response_type === "district_classification_change_analysis" || 
        map_type === "district_classification_change") {
      return false;
    }
    
    // Check if response_type indicates district classification
    if (response_type === "district_classification_analysis") {
      return true;
    }
    
    // Check if map_type indicates district classification
    if (map_type === "district_classification") {
      return true;
    }
    
    // Check function calls for get_district_classification
    if (function_calls) {
      if (function_calls.some(fc => fc.function === "get_district_classification")) {
        return true;
      }
    }
    
    // Check nested data structure
    if (data && Array.isArray(data) && data.length > 0 && data[0].result) {
      const result = data[0].result;
      if (result.response_type === "district_classification_analysis" || result.map_type === "district_classification") {
        return true;
      }
    }
    
    // Direct data structure check (but exclude change requests)
    if (visualizations.classified_districts && Array.isArray(visualizations.classified_districts) &&
        visualizations.classification_legend && Array.isArray(visualizations.classification_legend) &&
        response_type !== "district_classification_change_analysis" && 
        map_type !== "district_classification_change") {
      return true;
    }
    
    return false;
  };

  const isDistrictClassificationChangeRequest = (visualizations) => {
    if (!visualizations) return false;
    
    const { function_calls, data, map_type, response_type } = visualizations;
    
    // Check if response_type indicates district classification change
    if (response_type === "district_classification_change_analysis") {
      return true;
    }
    
    // Check if map_type indicates district classification change
    if (map_type === "district_classification_change") {
      return true;
    }
    
    // Check function calls for get_district_classification_change
    if (function_calls) {
      if (function_calls.some(fc => fc.function === "get_district_classification_change")) {
        return true;
      }
    }
    
    // Check nested data structure
    if (data && Array.isArray(data) && data.length > 0 && data[0].result) {
      const result = data[0].result;
      if (result.response_type === "district_classification_change_analysis" || result.map_type === "district_classification_change") {
        return true;
      }
    }
    
    // Direct data structure check
    if (visualizations.classified_districts && Array.isArray(visualizations.classified_districts) &&
        visualizations.classification_legend && Array.isArray(visualizations.classification_legend) &&
        visualizations.analysis_type === "change") {
      return true;
    }
    
    return false;
  };

  const isMultiIndicatorPerformanceRequest = (visualizations) => {
    if (!visualizations) return false;
    
    const { function_calls, data, map_type, response_type } = visualizations;
    
    // Exclude change classification requests
    if (response_type === "district_classification_change_analysis" || 
        map_type === "district_classification_change") {
      return false;
    }
    
    // Check if response_type indicates multi-indicator performance
    if (response_type === "multi_indicator_performance") {
      return true;
    }
    
    // Check if map_type indicates multi-indicator performance
    if (map_type === "multi_indicator_performance") {
      return true;
    }
    
    // Check function calls for get_multi_indicator_performance
    if (function_calls) {
      if (function_calls.some(fc => fc.function === "get_multi_indicator_performance")) {
        return true;
      }
    }
    
    // Check nested data structure
    if (data && Array.isArray(data) && data.length > 0 && data[0].result) {
      const result = data[0].result;
      if (result.response_type === "multi_indicator_performance" || result.map_type === "multi_indicator_performance") {
        return true;
      }
    }
    
    return false;
  };

  const hasActualMapData = (visualizations) => {
    if (!visualizations) return false;
    
    const { boundary, boundary_data } = visualizations;
    
    // Check both boundary and boundary_data fields for different function types
    const hasBoundary = boundary && Array.isArray(boundary) && boundary.length > 0;
    const hasBoundaryData = boundary_data && Array.isArray(boundary_data) && boundary_data.length > 0;
    
    return hasBoundary || hasBoundaryData;
  };

  const formatMessage = (text) => {
    if (!text) return "";
    
    // Handle unicode sequences and markdown formatting
    const decodeUnicode = (str) => {
      return str.replace(/\\u[\dA-F]{4}/gi, function (match) {
        return String.fromCharCode(parseInt(match.replace(/\\u/g, ''), 16));
      });
    };
    
    let formatted = decodeUnicode(text);
    
    // Format bold text
    formatted = formatted.replace(/\*\*(.*?)\*\*/g, '<strong>$1</strong>');
    
    // Format italic text
    formatted = formatted.replace(/\*(.*?)\*/g, '<em>$1</em>');
    
    // Format line breaks
    formatted = formatted.replace(/\n/g, '<br/>');
    
    return formatted;
  };

  const openModal = (visualizations, type) => {
    setModalData(visualizations);
    setModalType(type);
    setModalOpen(true);
  };

  const renderVisualization = (visualizations) => {
    if (!visualizations) {
      return null;
    }



    return (
      <div className="visualization-container" style={{ margin: "0", padding: "0", marginBottom: "0", paddingBottom: "0", minHeight: "0", height: "auto" }}>
        {/* Individual District Health Analysis */}
        {isIndividualDistrictRequest(visualizations) && (
          <div className="visualization-section">
            <h4>🏥 Individual District Health Analysis</h4>
            <p>Comprehensive health indicators data for the specified district.</p>
            
            {/* Visual Analysis Box */}
            <div style={{
              background: "linear-gradient(135deg, #f8f9fa 0%, #e9ecef 100%)",
              border: "1px solid #dee2e6",
              borderRadius: "12px",
              padding: "16px",
              marginTop: "12px"
            }}>
              <h5 style={{
                margin: "0 0 12px 0",
                color: "#495057",
                fontSize: "16px",
                fontWeight: "600",
                display: "flex",
                alignItems: "center",
                gap: "8px"
              }}>
                📊 Visual Analysis
              </h5>
              <div className="visualization-buttons" style={{
                display: "flex",
                gap: "12px"
              }}>
                {hasActualMapData(visualizations) && (
                  <button 
                    className="visualization-button map-button"
                    onClick={() => openModal(visualizations, 'map')}
                    style={{
                      background: "linear-gradient(135deg, #2E7D32 0%, #1B5E20 100%)",
                      color: "white",
                      border: "none",
                      borderRadius: "8px",
                      padding: "12px 20px",
                      fontSize: "14px",
                      fontWeight: "600",
                      cursor: "pointer",
                      display: "flex",
                      alignItems: "center",
                      gap: "8px",
                      transition: "all 0.3s ease",
                      boxShadow: "0 2px 4px rgba(46, 125, 50, 0.2)",
                      margin: "0"
                    }}
                  >
                    🗺️ View Map
                  </button>
                )}
                {(
                  visualizations.data || 
                  visualizations.chart_data ||
                  (visualizations.boundary && visualizations.boundary.length > 0) ||
                  (isIndividualDistrictRequest(visualizations) && hasActualMapData(visualizations))
                ) && (
                  <button 
                    className="visualization-button chart-button"
                    onClick={() => openModal(visualizations, 'chart')}
                    style={{
                      background: "linear-gradient(135deg, #3498DB 0%, #2980B9 100%)",
                      color: "white",
                      border: "none",
                      borderRadius: "8px",
                      padding: "12px 20px",
                      fontSize: "14px",
                      fontWeight: "600",
                      cursor: "pointer",
                      display: "flex",
                      alignItems: "center",
                      gap: "8px",
                      transition: "all 0.3s ease",
                      boxShadow: "0 2px 4px rgba(52, 152, 219, 0.2)",
                      margin: "0"
                    }}
                  >
                    📊 View Charts
                  </button>
                )}
              </div>
            </div>
          </div>
        )}

        {/* Multi-District Health Analysis */}
        {isMultiDistrictRequest(visualizations) && (
          <div className="visualization-section">
            <h4>🏥 Multi-District Health Analysis</h4>
            <p>Comparative health indicators analysis across multiple districts.</p>
            
            {/* Visual Analysis Box */}
            <div style={{
              background: "linear-gradient(135deg, #f8f9fa 0%, #e9ecef 100%)",
              border: "1px solid #dee2e6",
              borderRadius: "12px",
              padding: "16px",
              marginTop: "12px"
            }}>
              <h5 style={{
                margin: "0 0 12px 0",
                color: "#495057",
                fontSize: "16px",
                fontWeight: "600",
                display: "flex",
                alignItems: "center",
                gap: "8px"
              }}>
                📊 Visual Analysis
              </h5>
              <div className="visualization-buttons" style={{
                display: "flex",
                gap: "12px"
              }}>
                {hasActualMapData(visualizations) && (
                  <button 
                    className="visualization-button map-button"
                    onClick={() => openModal(visualizations, 'map')}
                    style={{
                      background: "linear-gradient(135deg, #2E7D32 0%, #1B5E20 100%)",
                      color: "white",
                      border: "none",
                      borderRadius: "8px",
                      padding: "12px 20px",
                      fontSize: "14px",
                      fontWeight: "600",
                      cursor: "pointer",
                      display: "flex",
                      alignItems: "center",
                      gap: "8px",
                      transition: "all 0.3s ease",
                      boxShadow: "0 2px 4px rgba(46, 125, 50, 0.2)",
                      margin: "0"
                    }}
                  >
                    🗺️ View Map
                  </button>
                )}
                {(visualizations.chart_data || visualizations.districts) && (
                  <button 
                    className="visualization-button chart-button"
                    onClick={() => openModal(visualizations, 'chart')}
                    style={{
                      background: "linear-gradient(135deg, #3498DB 0%, #2980B9 100%)",
                      color: "white",
                      border: "none",
                      borderRadius: "8px",
                      padding: "12px 20px",
                      fontSize: "14px",
                      fontWeight: "600",
                      cursor: "pointer",
                      display: "flex",
                      alignItems: "center",
                      gap: "8px",
                      transition: "all 0.3s ease",
                      boxShadow: "0 2px 4px rgba(52, 152, 219, 0.2)",
                      margin: "0"
                    }}
                  >
                    📊 View Charts
                  </button>
                )}
              </div>
            </div>
          </div>
        )}

        {/* State-wise Extremes */}
        {isStateWiseExtremesRequest(visualizations) && (
          <div className="visualization-section">
           
            {/* Visual Analysis Box */}
            <div style={{
              background: "linear-gradient(135deg, #f8f9fa 0%, #e9ecef 100%)",
              border: "1px solid #dee2e6",
              borderRadius: "12px",
              padding: "16px",
              marginTop: "12px"
            }}>
              <h5 style={{
                margin: "0 0 12px 0",
                color: "#495057",
                fontSize: "16px",
                fontWeight: "600",
                display: "flex",
                alignItems: "center",
                gap: "8px"
              }}>
                📊 Visual Analysis
              </h5>
              <div className="visualization-buttons" style={{
                display: "flex",
                gap: "12px"
              }}>
                {hasActualMapData(visualizations) && (
                  <button 
                    className="visualization-button map-button"
                    onClick={() => openModal(visualizations, 'map')}
                    style={{
                      background: "linear-gradient(135deg, #2E7D32 0%, #1B5E20 100%)",
                      color: "white",
                      border: "none",
                      borderRadius: "8px",
                      padding: "12px 20px",
                      fontSize: "14px",
                      fontWeight: "600",
                      cursor: "pointer",
                      display: "flex",
                      alignItems: "center",
                      gap: "8px",
                      transition: "all 0.3s ease",
                      boxShadow: "0 2px 4px rgba(46, 125, 50, 0.2)",
                      margin: "0"
                    }}
                  >
                    🗺️ View Map
                  </button>
                )}
                {(visualizations.chart_data || visualizations.data) && (
                  <button 
                    className="visualization-button chart-button"
                    onClick={() => openModal(visualizations, 'chart')}
                    style={{
                      background: "linear-gradient(135deg, #3498DB 0%, #2980B9 100%)",
                      color: "white",
                      border: "none",
                      borderRadius: "8px",
                      padding: "12px 20px",
                      fontSize: "14px",
                      fontWeight: "600",
                      cursor: "pointer",
                      display: "flex",
                      alignItems: "center",
                      gap: "8px",
                      transition: "all 0.3s ease",
                      boxShadow: "0 2px 4px rgba(52, 152, 219, 0.2)",
                      margin: "0"
                    }}
                  >
                    📊 View Charts
                  </button>
                )}
              </div>
            </div>
          </div>
        )}

        {/* Border Districts Analysis */}
        {(() => {
          const isBorderDistrictsReq = isBorderDistrictsRequest(visualizations);
          return isBorderDistrictsReq;
        })() && (
          <div className="visualization-section">
            <h4>🌐 Border Districts Analysis</h4>
            <p>Health indicators analysis for districts at state boundaries.</p>
            
            {/* Visual Analysis Box */}
            <div style={{
              background: "linear-gradient(135deg, #f8f9fa 0%, #e9ecef 100%)",
              border: "1px solid #dee2e6",
              borderRadius: "12px",
              padding: "16px",
              marginTop: "12px"
            }}>
              <h5 style={{
                margin: "0 0 12px 0",
                color: "#495057",
                fontSize: "16px",
                fontWeight: "600",
                display: "flex",
                alignItems: "center",
                gap: "8px"
              }}>
                📊 Visual Analysis
              </h5>
              <div className="visualization-buttons" style={{
                display: "flex",
                gap: "12px"
              }}>
                {hasActualMapData(visualizations) && (
                  <button 
                    className="visualization-button map-button"
                    onClick={() => openModal(visualizations, 'map')}
                    style={{
                      background: "linear-gradient(135deg, #2E7D32 0%, #1B5E20 100%)",
                      color: "white",
                      border: "none",
                      borderRadius: "8px",
                      padding: "12px 20px",
                      fontSize: "14px",
                      fontWeight: "600",
                      cursor: "pointer",
                      display: "flex",
                      alignItems: "center",
                      gap: "8px",
                      transition: "all 0.3s ease",
                      boxShadow: "0 2px 4px rgba(46, 125, 50, 0.2)",
                      margin: "0"
                    }}
                  >
                    🗺️ View Map
                  </button>
                )}
                {(() => {
                  // For border districts analysis, show chart button if we have districts data
                  const shouldShowChartButton = (
                    visualizations.chart_data || 
                    (visualizations.data && visualizations.data.length > 0) ||
                    (visualizations.boundary && visualizations.boundary.length > 0) ||
                    (isBorderDistrictsRequest(visualizations) && hasActualMapData(visualizations))
                  );
                  
                  return shouldShowChartButton;
                })() && (
                  <button 
                    className="visualization-button chart-button"
                    onClick={() => openModal(visualizations, 'chart')}
                    style={{
                      background: "linear-gradient(135deg, #3498DB 0%, #2980B9 100%)",
                      color: "white",
                      border: "none",
                      borderRadius: "8px",
                      padding: "12px 20px",
                      fontSize: "14px",
                      fontWeight: "600",
                      cursor: "pointer",
                      display: "flex",
                      alignItems: "center",
                      gap: "8px",
                      transition: "all 0.3s ease",
                      boxShadow: "0 2px 4px rgba(52, 152, 219, 0.2)",
                      margin: "0"
                    }}
                  >
                    📊 View Charts
                  </button>
                )}
              </div>
            </div>
          </div>
        )}

        {/* Constraint-Based Analysis */}
        {(() => {
          
          
          return isConstraintBasedRequest(visualizations);
        })() && (
          <div className="visualization-section">
            <h4>🎯 Constraint-Based Analysis</h4>
            <p>Districts meeting specific health indicator thresholds and criteria.</p>
            
            {/* Visual Analysis Box */}
            <div style={{
              background: "linear-gradient(135deg, #f8f9fa 0%, #e9ecef 100%)",
              border: "1px solid #dee2e6",
              borderRadius: "12px",
              padding: "16px",
              marginTop: "12px"
            }}>
              <h5 style={{
                margin: "0 0 12px 0",
                color: "#495057",
                fontSize: "16px",
                fontWeight: "600",
                display: "flex",
                alignItems: "center",
                gap: "8px"
              }}>
                📊 Visual Analysis
              </h5>
              <div className="visualization-buttons" style={{
                display: "flex",
                gap: "12px"
              }}>
                {hasActualMapData(visualizations) && (
                  <button 
                    className="visualization-button map-button"
                    onClick={() => openModal(visualizations, 'map')}
                    style={{
                      background: "linear-gradient(135deg, #2E7D32 0%, #1B5E20 100%)",
                      color: "white",
                      border: "none",
                      borderRadius: "8px",
                      padding: "12px 20px",
                      fontSize: "14px",
                      fontWeight: "600",
                      cursor: "pointer",
                      display: "flex",
                      alignItems: "center",
                      gap: "8px",
                      transition: "all 0.3s ease",
                      boxShadow: "0 2px 4px rgba(46, 125, 50, 0.2)",
                      margin: "0"
                    }}
                  >
                    🗺️ View Map
                  </button>
                )}
                {(visualizations.chart_data || visualizations.districts) && (
                  <button 
                    className="visualization-button chart-button"
                    onClick={() => openModal(visualizations, 'chart')}
                    style={{
                      background: "linear-gradient(135deg, #3498DB 0%, #2980B9 100%)",
                      color: "white",
                      border: "none",
                      borderRadius: "8px",
                      padding: "12px 20px",
                      fontSize: "14px",
                      fontWeight: "600",
                      cursor: "pointer",
                      display: "flex",
                      alignItems: "center",
                      gap: "8px",
                      transition: "all 0.3s ease",
                      boxShadow: "0 2px 4px rgba(52, 152, 219, 0.2)",
                      margin: "0"
                    }}
                  >
                    📊 View Charts
                  </button>
                )}
              </div>
            </div>
          </div>
        )}

        {/* Top/Bottom Districts Analysis */}
        {isTopBottomDistrictsRequest(visualizations) && (
          <div className="visualization-section">
           
            
            {/* Visual Analysis Box */}
            <div style={{
              background: "linear-gradient(135deg, #f8f9fa 0%, #e9ecef 100%)",
              border: "1px solid #dee2e6",
              borderRadius: "12px",
              padding: "16px",
              marginTop: "12px"
            }}>
              <h5 style={{
                margin: "0 0 12px 0",
                color: "#495057",
                fontSize: "16px",
                fontWeight: "600",
                display: "flex",
                alignItems: "center",
                gap: "8px"
              }}>
                📊 Visual Analysis
              </h5>
              <div className="visualization-buttons" style={{
                display: "flex",
                gap: "12px"
              }}>
                {hasActualMapData(visualizations) && (
                  <button 
                    className="visualization-button map-button"
                    onClick={() => openModal(visualizations, 'map')}
                    style={{
                      background: "linear-gradient(135deg, #2E7D32 0%, #1B5E20 100%)",
                      color: "white",
                      border: "none",
                      borderRadius: "8px",
                      padding: "12px 20px",
                      fontSize: "14px",
                      fontWeight: "600",
                      cursor: "pointer",
                      display: "flex",
                      alignItems: "center",
                      gap: "8px",
                      transition: "all 0.3s ease",
                      boxShadow: "0 2px 4px rgba(46, 125, 50, 0.2)"
                    }}
                  >
                    🗺️ View Map
                  </button>
                )}
                {visualizations.chart_data && (
                  <button 
                    className="visualization-button chart-button"
                    onClick={() => openModal(visualizations, 'chart')}
                    style={{
                      background: "linear-gradient(135deg, #3498DB 0%, #2980B9 100%)",
                      color: "white",
                      border: "none",
                      borderRadius: "8px",
                      padding: "12px 20px",
                      fontSize: "14px",
                      fontWeight: "600",
                      cursor: "pointer",
                      display: "flex",
                      alignItems: "center",
                      gap: "8px",
                      transition: "all 0.3s ease",
                      boxShadow: "0 2px 4px rgba(52, 152, 219, 0.2)"
                    }}
                  >
                    📊 View Charts
                  </button>
                )}
              </div>
            </div>
          </div>
        )}

        {/* District Performance Comparison */}
        {isDistrictComparisonRequest(visualizations) && (
          <div className="visualization-section">
            
            
            {/* Visual Analysis Box */}
            <div style={{
              background: "linear-gradient(135deg, #f8f9fa 0%, #e9ecef 100%)",
              border: "1px solid #dee2e6",
              borderRadius: "12px",
              padding: "16px",
              marginTop: "12px"
            }}>
              <h5 style={{
                margin: "0 0 12px 0",
                color: "#495057",
                fontSize: "16px",
                fontWeight: "600",
                display: "flex",
                alignItems: "center",
                gap: "8px"
              }}>
                📊 Visual Analysis
              </h5>
              <div className="visualization-buttons" style={{
                display: "flex",
                gap: "12px"
              }}>
                {hasActualMapData(visualizations) && (
                  <button 
                    className="visualization-button map-button"
                    onClick={() => openModal(visualizations, 'map')}
                    style={{
                      background: "linear-gradient(135deg, #2E7D32 0%, #1B5E20 100%)",
                      color: "white",
                      border: "none",
                      borderRadius: "8px",
                      padding: "12px 20px",
                      fontSize: "14px",
                      fontWeight: "600",
                      cursor: "pointer",
                      display: "flex",
                      alignItems: "center",
                      gap: "8px",
                      transition: "all 0.3s ease",
                      boxShadow: "0 2px 4px rgba(46, 125, 50, 0.3)"
                    }}
                  >
                    🗺️ View Map
                  </button>
                )}
                {(visualizations.chart_data || (visualizations.data && visualizations.data[0]?.result?.chart_data)) && (
                  <button 
                    className="visualization-button chart-button"
                    onClick={() => openModal(visualizations, 'chart')}
                    style={{
                      background: "linear-gradient(135deg, #1976D2 0%, #0D47A1 100%)",
                      color: "white",
                      border: "none",
                      borderRadius: "8px",
                      padding: "12px 20px",
                      fontSize: "14px",
                      fontWeight: "600",
                      cursor: "pointer",
                      display: "flex",
                      alignItems: "center",
                      gap: "8px",
                      transition: "all 0.3s ease",
                      boxShadow: "0 2px 4px rgba(25, 118, 210, 0.3)"
                    }}
                  >
                    📊 View Charts
                  </button>
                )}
              </div>
              <div style={{
                fontSize: "13px",
                color: "#6c757d",
                marginTop: "12px",
                padding: "8px",
                background: "rgba(255, 255, 255, 0.5)",
                borderRadius: "6px"
              }}>
                
              </div>
            </div>
          </div>
        )}

        {/* Multi-Indicator Performance Analysis */}
        {isMultiIndicatorPerformanceRequest(visualizations) && (
          <div className="visualization-section">
            
            {/* Visual Analysis Box */}
            <div style={{
              background: "linear-gradient(135deg, #f8f9fa 0%, #e9ecef 100%)",
              border: "1px solid #dee2e6",
              borderRadius: "12px",
              padding: "16px",
              marginTop: "12px"
            }}>
              
              <div style={{
                display: "flex",
                gap: "12px",
                flexWrap: "wrap"
              }}>
                {hasActualMapData(visualizations) && (
                  <button
                    onClick={() => {
                      setModalData(visualizations);
                      setModalType('map');
                      setModalOpen(true);
                    }}
                    style={{
                      background: "linear-gradient(135deg, #28A745 0%, #1E7E34 100%)",
                      color: "white",
                      border: "none",
                      borderRadius: "8px",
                      padding: "12px 20px",
                      fontSize: "14px",
                      fontWeight: "600",
                      cursor: "pointer",
                      display: "flex",
                      alignItems: "center",
                      gap: "8px",
                      transition: "all 0.3s ease",
                      boxShadow: "0 2px 4px rgba(40, 167, 69, 0.3)"
                    }}
                  >
                    🗺️ View Map
                  </button>
                )}
                {visualizations.chart_data && visualizations.chart_data.length > 0 && (
                  <button
                    onClick={() => {
                      setModalData(visualizations);
                      setModalType('chart');
                      setModalOpen(true);
                    }}
                    style={{
                      background: "linear-gradient(135deg, #1976D2 0%, #0D47A1 100%)",
                      color: "white",
                      border: "none",
                      borderRadius: "8px",
                      padding: "12px 20px",
                      fontSize: "14px",
                      fontWeight: "600",
                      cursor: "pointer",
                      display: "flex",
                      alignItems: "center",
                      gap: "8px",
                      transition: "all 0.3s ease",
                      boxShadow: "0 2px 4px rgba(25, 118, 210, 0.3)"
                    }}
                  >
                    📊 View Charts
                  </button>
                )}
              </div>
              
            </div>
          </div>
        )}

        {/* Indicator Change Analysis */}
        {isIndicatorChangeAnalysisRequest(visualizations) && (
          <div className="visualization-section">
           
            
            {/* Visual Analysis Box */}
            <div style={{
              background: "linear-gradient(135deg, #f8f9fa 0%, #e9ecef 100%)",
              border: "1px solid #dee2e6",
              borderRadius: "12px",
              padding: "16px",
              marginTop: "12px"
            }}>
              <h5 style={{
                margin: "0 0 12px 0",
                color: "#495057",
                fontSize: "16px",
                fontWeight: "600",
                display: "flex",
                alignItems: "center",
                gap: "8px"
              }}>
                📊 Visual Analysis
              </h5>
              <div className="visualization-buttons" style={{
                display: "flex",
                gap: "12px"
              }}>
                {hasActualMapData(visualizations) && (
                  <button 
                    className="visualization-button map-button"
                    onClick={() => openModal(visualizations, 'map')}
                    style={{
                      background: "linear-gradient(135deg, #2E7D32 0%, #1B5E20 100%)",
                      color: "white",
                      border: "none",
                      borderRadius: "8px",
                      padding: "12px 20px",
                      fontSize: "14px",
                      fontWeight: "600",
                      cursor: "pointer",
                      display: "flex",
                      alignItems: "center",
                      gap: "8px",
                      transition: "all 0.3s ease",
                      boxShadow: "0 2px 4px rgba(46, 125, 50, 0.2)",
                      margin: "0"
                    }}
                  >
                    🗺️ View Map
                  </button>
                )}
                {visualizations.chart_data && (
                  <button 
                    className="visualization-button chart-button"
                    onClick={() => openModal(visualizations, 'chart')}
                    style={{
                      background: "linear-gradient(135deg, #3498DB 0%, #2980B9 100%)",
                      color: "white",
                      border: "none",
                      borderRadius: "8px",
                      padding: "12px 20px",
                      fontSize: "14px",
                      fontWeight: "600",
                      cursor: "pointer",
                      display: "flex",
                      alignItems: "center",
                      gap: "8px",
                      transition: "all 0.3s ease",
                      boxShadow: "0 2px 4px rgba(52, 152, 219, 0.2)",
                      margin: "0"
                    }}
                  >
                    📊 View Charts
                  </button>
                )}
              </div>
            </div>
          </div>
        )}

        {/* Radius Analysis */}
        {(() => {
          
          return isRadiusAnalysisRequest(visualizations);
        })() && (
          <div className="visualization-section" style={{ 
            marginBottom: "0px", 
            paddingBottom: "0px", 
            marginTop: "0px", 
            paddingTop: "0px",
            display: "block",
            overflow: "hidden"
          }}>
          
            <div className="visualization-buttons" style={{ 
              marginBottom: "0px", 
              marginTop: "0px", 
              paddingBottom: "0px", 
              paddingTop: "0px",
              height: "auto",
              minHeight: "0"
            }}>
              {hasActualMapData(visualizations) && (
                <button 
                  className="visualization-button map-button"
                  onClick={() => openModal(visualizations, 'map')}
                  style={{
                    background: "linear-gradient(135deg, #2E7D32 0%, #1B5E20 100%)",
                    color: "white",
                    border: "none",
                    borderRadius: "8px",
                    padding: "10px 16px",
                    fontSize: "14px",
                    fontWeight: "600",
                    cursor: "pointer",
                    display: "flex",
                    alignItems: "center",
                    gap: "8px",
                    transition: "all 0.3s ease",
                    boxShadow: "0 2px 4px rgba(46, 125, 50, 0.2)",
                    margin: "0",
                    marginBottom: "0",
                    marginTop: "0"
                  }}
                >
                  🗺️ View Map
                </button>
              )}
              {(() => {
                
                // For radius analysis, always show chart button if we have districts data
                // The RadiusAnalysis component has fallback chart generation
                const shouldShowChartButton = (
                  visualizations.chart_data || 
                  (visualizations.districts && visualizations.districts.length > 0) ||
                  (isRadiusAnalysisRequest(visualizations) && hasActualMapData(visualizations))
                );
                
                
                return shouldShowChartButton;
              })() && (
                <button 
                  className="visualization-button chart-button"
                  onClick={() => openModal(visualizations, 'chart')}
                  style={{
                    background: "linear-gradient(135deg, #3498DB 0%, #2980B9 100%)",
                    color: "white",
                    border: "none",
                    borderRadius: "8px",
                    padding: "10px 16px",
                    fontSize: "14px",
                    fontWeight: "600",
                    cursor: "pointer",
                    display: "flex",
                    alignItems: "center",
                    gap: "8px",
                    transition: "all 0.3s ease",
                    boxShadow: "0 2px 4px rgba(52, 152, 219, 0.2)",
                    marginLeft: "8px",
                    margin: "0 0 0 8px",
                    marginBottom: "0",
                    marginTop: "0"
                  }}
                >
                  📊 View Charts
                </button>
              )}
            </div>
          </div>
        )}

        {/* Multi-Indicator Performance Analysis */}
        {isMultiIndicatorPerformanceRequest(visualizations) && (
          <div className="visualization-section">
            <div className="visualization-buttons" style={{
              display: "flex",
              gap: "12px"
            }}>
              {hasActualMapData(visualizations) && (
                <button 
                  className="visualization-button map-button"
                  onClick={() => openModal(visualizations, 'map')}
                  style={{
                    background: "linear-gradient(135deg, #2E7D32 0%, #1B5E20 100%)",
                    color: "white",
                    border: "none",
                    borderRadius: "8px",
                    padding: "12px 20px",
                    fontSize: "14px",
                    fontWeight: "600",
                    cursor: "pointer",
                    display: "flex",
                    alignItems: "center",
                    gap: "8px",
                    transition: "all 0.3s ease",
                    boxShadow: "0 2px 4px rgba(46, 125, 50, 0.2)",
                    margin: "0"
                  }}
                >
                  🗺️ View Map
                </button>
              )}
              {visualizations.chart_data && (
                <button 
                  className="visualization-button chart-button"
                  onClick={() => openModal(visualizations, 'chart')}
                  style={{
                    background: "linear-gradient(135deg, #3498DB 0%, #2980B9 100%)",
                    color: "white",
                    border: "none",
                    borderRadius: "8px",
                    padding: "12px 20px",
                    fontSize: "14px",
                    fontWeight: "600",
                    cursor: "pointer",
                    display: "flex",
                    alignItems: "center",
                    gap: "8px",
                    transition: "all 0.3s ease",
                    boxShadow: "0 2px 4px rgba(52, 152, 219, 0.2)",
                    margin: "0"
                  }}
                >
                  📊 View Charts
                </button>
              )}
            </div>
          </div>
        )}

        {/* State Multi-Indicator Performance Analysis */}
        {isStateMultiIndicatorPerformanceRequest(visualizations) && (
          <div className="visualization-section">
            <div className="visualization-buttons" style={{
              display: "flex",
              gap: "12px"
            }}>
              {hasActualMapData(visualizations) && (
                <button 
                  className="visualization-button map-button"
                  onClick={() => openModal(visualizations, 'map')}
                  style={{
                    background: "linear-gradient(135deg, #2E7D32 0%, #1B5E20 100%)",
                    color: "white",
                    border: "none",
                    borderRadius: "8px",
                    padding: "12px 20px",
                    fontSize: "14px",
                    fontWeight: "600",
                    cursor: "pointer",
                    display: "flex",
                    alignItems: "center",
                    gap: "8px",
                    transition: "all 0.3s ease",
                    boxShadow: "0 2px 4px rgba(46, 125, 50, 0.2)",
                    margin: "0"
                  }}
                >
                  🏛️ View Map
                </button>
              )}
              {(visualizations.chart_data || visualizations.states) && (
                <button 
                  className="visualization-button chart-button"
                  onClick={() => openModal(visualizations, 'chart')}
                  style={{
                    background: "linear-gradient(135deg, #6A4C93 0%, #9A6FAC 100%)",
                    color: "white",
                    border: "none",
                    borderRadius: "8px",
                    padding: "12px 20px",
                    fontSize: "14px",
                    fontWeight: "600",
                    cursor: "pointer",
                    display: "flex",
                    alignItems: "center",
                    gap: "8px",
                    transition: "all 0.3s ease",
                    boxShadow: "0 2px 4px rgba(106, 76, 147, 0.2)",
                    margin: "0"
                  }}
                >
                  📊 View Charts
                </button>
              )}
            </div>
          </div>
        )}

        {/* District Similarity Analysis */}
        {isDistrictSimilarityAnalysisRequest(visualizations) && (
          <div className="visualization-section">
            <div className="visualization-buttons" style={{
              display: "flex",
              gap: "12px"
            }}>
              {hasActualMapData(visualizations) && (
                <button 
                  className="visualization-button map-button"
                  onClick={() => openModal(visualizations, 'map')}
                  style={{
                    background: "linear-gradient(135deg, #2E7D32 0%, #1B5E20 100%)",
                    color: "white",
                    border: "none",
                    borderRadius: "8px",
                    padding: "12px 20px",
                    fontSize: "14px",
                    fontWeight: "600",
                    cursor: "pointer",
                    boxShadow: "0 2px 8px rgba(46, 125, 50, 0.25)",
                    transition: "all 0.3s ease",
                    display: "flex",
                    alignItems: "center",
                    gap: "8px"
                  }}
                  onMouseEnter={(e) => {
                    e.target.style.transform = "translateY(-2px)";
                    e.target.style.boxShadow = "0 4px 12px rgba(46, 125, 50, 0.35)";
                  }}
                  onMouseLeave={(e) => {
                    e.target.style.transform = "translateY(0)";
                    e.target.style.boxShadow = "0 2px 8px rgba(46, 125, 50, 0.25)";
                  }}
                >
                  🗺️ View Map
                </button>
              )}
              {visualizations?.chart_data && Array.isArray(visualizations.chart_data) && visualizations.chart_data.length > 0 && (
                <button 
                  className="visualization-button chart-button"
                  onClick={() => openModal(visualizations, 'chart')}
                  style={{
                    background: "linear-gradient(135deg, #1976D2 0%, #0D47A1 100%)",
                    color: "white",
                    border: "none",
                    borderRadius: "8px",
                    padding: "12px 20px",
                    fontSize: "14px",
                    fontWeight: "600",
                    cursor: "pointer",
                    boxShadow: "0 2px 8px rgba(25, 118, 210, 0.25)",
                    transition: "all 0.3s ease",
                    display: "flex",
                    alignItems: "center",
                    gap: "8px"
                  }}
                  onMouseEnter={(e) => {
                    e.target.style.transform = "translateY(-2px)";
                    e.target.style.boxShadow = "0 4px 12px rgba(25, 118, 210, 0.35)";
                  }}
                  onMouseLeave={(e) => {
                    e.target.style.transform = "translateY(0)";
                    e.target.style.boxShadow = "0 2px 8px rgba(25, 118, 210, 0.25)";
                  }}
                >
                  📊 View Charts
                </button>
              )}
            </div>
          </div>
        )}

        {/* District Classification */}
        {isDistrictClassificationRequest(visualizations) && (
          <div className="visualization-section">
            <div className="visualization-buttons" style={{
              display: "flex",
              gap: "12px"
            }}>
              {hasActualMapData(visualizations) && (
                <button 
                  className="visualization-button map-button"
                  onClick={() => openModal(visualizations, 'map')}
                  style={{
                    background: "linear-gradient(135deg, #7B1FA2 0%, #4A148C 100%)",
                    color: "white",
                    border: "none",
                    borderRadius: "8px",
                    padding: "12px 20px",
                    fontSize: "14px",
                    fontWeight: "600",
                    cursor: "pointer",
                    boxShadow: "0 2px 8px rgba(123, 31, 162, 0.25)",
                    transition: "all 0.3s ease",
                    display: "flex",
                    alignItems: "center",
                    gap: "8px"
                  }}
                  onMouseEnter={(e) => {
                    e.target.style.transform = "translateY(-2px)";
                    e.target.style.boxShadow = "0 4px 12px rgba(123, 31, 162, 0.35)";
                  }}
                  onMouseLeave={(e) => {
                    e.target.style.transform = "translateY(0)";
                    e.target.style.boxShadow = "0 2px 8px rgba(123, 31, 162, 0.25)";
                  }}
                >
                  🗺️ View Classification Map
                </button>
              )}
              {visualizations?.chart_data && (
                <button 
                  className="visualization-button chart-button"
                  onClick={() => openModal(visualizations, 'chart')}
                  style={{
                    background: "linear-gradient(135deg, #E91E63 0%, #AD1457 100%)",
                    color: "white",
                    border: "none",
                    borderRadius: "8px",
                    padding: "12px 20px",
                    fontSize: "14px",
                    fontWeight: "600",
                    cursor: "pointer",
                    boxShadow: "0 2px 8px rgba(233, 30, 99, 0.25)",
                    transition: "all 0.3s ease",
                    display: "flex",
                    alignItems: "center",
                    gap: "8px"
                  }}
                  onMouseEnter={(e) => {
                    e.target.style.transform = "translateY(-2px)";
                    e.target.style.boxShadow = "0 4px 12px rgba(233, 30, 99, 0.35)";
                  }}
                  onMouseLeave={(e) => {
                    e.target.style.transform = "translateY(0)";
                    e.target.style.boxShadow = "0 2px 8px rgba(233, 30, 99, 0.25)";
                  }}
                >
                  📊 View Classification Charts
                </button>
              )}
            </div>
          </div>
        )}

        {/* District Classification Change */}
        {isDistrictClassificationChangeRequest(visualizations) && (
          <div className="visualization-section">
            <div className="visualization-buttons" style={{
              display: "flex",
              gap: "12px"
            }}>
              {hasActualMapData(visualizations) && (
                <button 
                  className="visualization-button map-button"
                  onClick={() => openModal(visualizations, 'map')}
                  style={{
                    background: "linear-gradient(135deg, #16A085 0%, #0E6B5E 100%)",
                    color: "white",
                    border: "none",
                    borderRadius: "8px",
                    padding: "12px 20px",
                    fontSize: "14px",
                    fontWeight: "600",
                    cursor: "pointer",
                    boxShadow: "0 2px 8px rgba(22, 160, 133, 0.25)",
                    transition: "all 0.3s ease",
                    display: "flex",
                    alignItems: "center",
                    gap: "8px"
                  }}
                  onMouseEnter={(e) => {
                    e.target.style.transform = "translateY(-2px)";
                    e.target.style.boxShadow = "0 4px 12px rgba(22, 160, 133, 0.35)";
                  }}
                  onMouseLeave={(e) => {
                    e.target.style.transform = "translateY(0)";
                    e.target.style.boxShadow = "0 2px 8px rgba(22, 160, 133, 0.25)";
                  }}
                >
                  📈 View Change Classification Map
                </button>
              )}
              {visualizations?.chart_data && (
                <button 
                  className="visualization-button chart-button"
                  onClick={() => openModal(visualizations, 'chart')}
                  style={{
                    background: "linear-gradient(135deg, #F39C12 0%, #D68910 100%)",
                    color: "white",
                    border: "none",
                    borderRadius: "8px",
                    padding: "12px 20px",
                    fontSize: "14px",
                    fontWeight: "600",
                    cursor: "pointer",
                    boxShadow: "0 2px 8px rgba(243, 156, 18, 0.25)",
                    transition: "all 0.3s ease",
                    display: "flex",
                    alignItems: "center",
                    gap: "8px"
                  }}
                  onMouseEnter={(e) => {
                    e.target.style.transform = "translateY(-2px)";
                    e.target.style.boxShadow = "0 4px 12px rgba(243, 156, 18, 0.35)";
                  }}
                  onMouseLeave={(e) => {
                    e.target.style.transform = "translateY(0)";
                    e.target.style.boxShadow = "0 2px 8px rgba(243, 156, 18, 0.25)";
                  }}
                >
                  📊 View Change Classification Charts
                </button>
              )}
            </div>
          </div>
        )}
      </div>
    );
  };

  return (
    <div className="app-container">
      {/* Enhanced Header */}
      <div style={{
        borderBottom: "1px solid #e8d5b7",
        padding: "28px 20px",
        background: "linear-gradient(135deg, #ffffff 0%, #faf9f7 100%)",
        textAlign: "center",
        position: "relative",
        boxShadow: "0 2px 8px rgba(139, 69, 19, 0.06)"
      }}>
        <h1 style={{
          margin: 0,
          fontSize: "32px",
          fontWeight: "700",
          background: "linear-gradient(135deg, #2E7D32 0%, #388E3C 100%)",
          WebkitBackgroundClip: "text",
          WebkitTextFillColor: "transparent",
          backgroundClip: "text"
        }}>
          🏥 Ask IPI
        </h1>
        <p style={{
          margin: "10px 0 0 0",
          fontSize: "18px",
          color: "#2E7D32",
          fontWeight: "400"
        }}>
          Conversational Analytics on Population Health and Social Determinants of Health Indicators for Districts in India
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
              background: "linear-gradient(135deg, #2E7D32 0%, #1B5E20 100%)",
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
              boxShadow: "0 4px 16px rgba(46, 125, 50, 0.3)",
              zIndex: 10
            }}
            onMouseEnter={(e) => {
              e.target.style.transform = "translateY(-50%) translateY(-2px)";
              e.target.style.boxShadow = "0 6px 20px rgba(46, 125, 50, 0.4)";
            }}
            onMouseLeave={(e) => {
              e.target.style.transform = "translateY(-50%)";
              e.target.style.boxShadow = "0 4px 16px rgba(46, 125, 50, 0.3)";
            }}
            title="Return to Home Page (Chat will be preserved)"
          >
            <span>🏠</span>
            <span>Home</span>
          </button>
        )}

        {/* Clear Chat Button */}
        {messages.length > 0 && (
          <button
            onClick={clearChat}
            style={{
              position: "absolute",
              top: "50%",
              left: "140px", // Position next to home button
              transform: "translateY(-50%)",
              background: "linear-gradient(135deg, #DC2626 0%, #B91C1C 100%)",
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
              boxShadow: "0 4px 16px rgba(220, 38, 38, 0.3)",
              zIndex: 10
            }}
            onMouseEnter={(e) => {
              e.target.style.transform = "translateY(-50%) translateY(-2px)";
              e.target.style.boxShadow = "0 6px 20px rgba(220, 38, 38, 0.4)";
            }}
            onMouseLeave={(e) => {
              e.target.style.transform = "translateY(-50%)";
              e.target.style.boxShadow = "0 4px 16px rgba(220, 38, 38, 0.3)";
            }}
            title="Clear all chat messages"
          >
            <span>🗑️</span>
            <span>Clear Chat</span>
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
          background: "linear-gradient(135deg, #2E7D32 0%, #388E3C 50%, #4CAF50 100%)",
          display: "flex",
          alignItems: "center",
          justifyContent: "center",
          color: "white",
          fontWeight: "700",
          fontSize: "20px",
          boxShadow: "0 4px 16px rgba(46, 125, 50, 0.3)",
          border: "2px solid rgba(255, 255, 255, 0.3)",
          transition: "all 0.3s ease"
        }}>
          U
        </div>
      </div>

      {/* Conversation Container */}
      <div 
        className="conversation-container" 
        ref={chatContainerRef}
        style={messages.length === 0 ? {
          minHeight: "auto", // Let content determine height
          display: "flex",
          flexDirection: "column",
          flex: "1"
        } : {}}
      >
        {/* Centered Welcome Section for Empty State */}
        {messages.length === 0 && (
          <div style={{
            display: "flex",
            flexDirection: "column",
            alignItems: "center",
            justifyContent: "flex-start",
            minHeight: "20vh",
            padding: "clamp(1rem, 3vw, 2rem) 1rem",
            textAlign: "center",
            paddingTop: "clamp(2rem, 5vh, 4rem)",
            paddingBottom: "clamp(1rem, 3vh, 2rem)"
          }}>
            <div style={{
              marginBottom: "clamp(1.5rem, 4vh, 3rem)",
              width: "100%",
              maxWidth: "800px"
            }}>
              <div style={{
                fontSize: "clamp(1.8rem, 4vw, 3rem)",
                fontWeight: "800",
                color: "#2E7D32",
                marginBottom: "clamp(0.5rem, 2vh, 1rem)",
                background: "linear-gradient(135deg, #4CAF50 0%, #2E7D32 100%)",
                backgroundClip: "text",
                WebkitBackgroundClip: "text",
                WebkitTextFillColor: "transparent",
                lineHeight: "1.2"
              }}>
                IPI Assistant
              </div>
              <div style={{
                fontSize: "clamp(0.9rem, 2.5vw, 1.25rem)",
                color: "#666",
                fontWeight: "500",
                maxWidth: "100%",
                lineHeight: "1.4",
                margin: "0 auto"
              }}>
                Ask me anything about IPI indicators and district data
              </div>
            </div>
          </div>
        )}

        {messages.map((message) => (
          <div key={message.id} className={`conversation-message ${message.type}`} style={message.visualizations ? { paddingBottom: "0px" } : {}}>
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
                  <div dangerouslySetInnerHTML={{ __html: formatMessage(message.text) }} />
                  
                  {/* Show typewriter cursor for messages that are currently typing */}
                  {isTyping && currentTypingMessageId === message.id && (
                    <span style={{
                      display: "inline-block",
                      width: "2px",
                      height: "20px",
                      backgroundColor: "#2E7D32",
                      marginLeft: "2px",
                      animation: "blink 1s infinite"
                    }}>
                    </span>
                  )}
                  
                  {/* Reaction buttons for assistant messages - only show when not typing */}
                  {message.type === "assistant" && !isTyping && (
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
                             border: "1px solid #2E7D32",
                             borderRadius: "20px",
                             padding: "6px 12px",
                             cursor: "pointer",
                             fontSize: "14px",
                             fontWeight: "500",
                             color: "#2E7D32",
                             display: "flex",
                             alignItems: "center",
                             gap: "6px",
                             transition: "all 0.3s ease",
                             boxShadow: "0 2px 8px rgba(46, 125, 50, 0.1)",
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
                             e.target.style.boxShadow = "0 4px 12px rgba(46, 125, 50, 0.15)";
                           }}
                           onMouseLeave={(e) => {
                             e.target.style.background = "linear-gradient(135deg, #f5f3f0 0%, #e8d5b7 100%)";
                             e.target.style.transform = "translateY(0)";
                             e.target.style.boxShadow = "0 2px 8px rgba(46, 125, 50, 0.1)";
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
                             boxShadow: "0 12px 40px rgba(46, 125, 50, 0.25)",
                             padding: "8px 4px",
                             border: "2px solid #2E7D32",
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
                          border: "1px solid #2E7D32",
                          boxShadow: "0 2px 8px rgba(46, 125, 50, 0.1)",
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
                            color: "#2E7D32",
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
                {message.visualizations && !isTyping && (
                  (() => {
                   
                    return true;
                  })() && 
                  <div className="visualization-controls" style={{ 
                    margin: "0", 
                    padding: "0"
                  }}>
                    {/* Call the renderVisualization function */}
                    {renderVisualization(message.visualizations)}
                    
                    {/* Fallback basic map button if renderVisualization doesn't show anything */}
                    {(() => {
                      const hasMapResult = hasActualMapData(message.visualizations);
                      const isRadius = isRadiusAnalysisRequest(message.visualizations);
                      const isBorder = isBorderDistrictsRequest(message.visualizations);
                      const isState = isStateWiseExtremesRequest(message.visualizations);
                      const isIndividual = isIndividualDistrictRequest(message.visualizations);
                      const isMulti = isMultiDistrictRequest(message.visualizations);
                      const isConstraint = isConstraintBasedRequest(message.visualizations);
                      const isTopBottom = isTopBottomDistrictsRequest(message.visualizations);
                      const isIndicatorChange = false; // isIndicatorChangeAnalysisRequest(message.visualizations); // Temporarily disabled
                      const isDistrictComparison = isDistrictComparisonRequest(message.visualizations);
                      const isMultiIndicatorPerformance = isMultiIndicatorPerformanceRequest(message.visualizations);
                      const isStateMultiIndicatorPerformance = isStateMultiIndicatorPerformanceRequest(message.visualizations);
                      const isDistrictSimilarityAnalysis = isDistrictSimilarityAnalysisRequest(message.visualizations);
                      const isDistrictClassification = isDistrictClassificationRequest(message.visualizations);
                      const isDistrictClassificationChange = isDistrictClassificationChangeRequest(message.visualizations);
                      
                      // Only show fallback if none of the specialized functions match
                      if (hasMapResult && !isRadius && !isBorder && !isState && !isIndividual && !isMulti && !isConstraint && !isTopBottom && !isIndicatorChange && !isDistrictComparison && !isMultiIndicatorPerformance && !isStateMultiIndicatorPerformance && !isDistrictSimilarityAnalysis && !isDistrictClassification && !isDistrictClassificationChange) {
                        return (
                          <div style={{ marginBottom: '10px' }}>
                            <button
                              className="expand-button"
                              onClick={() => openModal(message.visualizations, 'map')}
                              style={{
                                background: "linear-gradient(135deg, #2E7D32 0%, #1B5E20 100%)",
                                color: "white",
                                border: "none",
                                borderRadius: "8px",
                                padding: "10px 16px",
                                fontSize: "14px",
                                fontWeight: "600",
                                cursor: "pointer",
                                display: "flex",
                                alignItems: "center",
                                gap: "8px",
                                transition: "all 0.3s ease",
                                boxShadow: "0 2px 4px rgba(46, 125, 50, 0.2)"
                              }}
                            >
                              🗺️ View Map
                            </button>
                          </div>
                        );
                      }
                      return null;
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
              <div className="message-avatar assistant">🏥</div>
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
                  Analyzing data...
                </div>
              </div>
            </div>
          </div>
        )}
      </div>

      {/* Input Section - Always visible below Ask IPI when no messages */}
      <div 
        className={messages.length === 0 ? "chat-input-section-centered" : "chat-input-section"}
        style={messages.length === 0 ? {
          position: "relative !important",
          display: "block !important",
          marginTop: "clamp(1rem, 3vh, 2rem)",
          marginBottom: "40px", // Space for bottom warning box
          zIndex: 10,
          width: "100%",
          visibility: "visible"
        } : {}}
      >
        <div className="chat-input-wrapper">
          <div className="chat-input-container">
            <textarea
              value={query}
              onChange={(e) => setQuery(e.target.value)}
              placeholder={isRecording ? "🎤 Listening... Speak your question" : "Message your assistant... (Press Enter to send, Shift+Enter for new line, 🎤 for voice)"}
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
                  background: "linear-gradient(135deg, #2E7D32 0%, #1B5E20 100%)",
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
                  background: "linear-gradient(135deg, #4CAF50 0%, #2E7D32 100%)",
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
              style={{
                background: !isLoading && query.trim() 
                  ? "linear-gradient(135deg, #2E7D32 0%, #1B5E20 100%)"
                  : "#ccc"
              }}
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
            color: "#2E7D32",
            marginTop: "16px",
            opacity: 0.8,
            fontWeight: "500",
            letterSpacing: "0.5px"
          }}>
            🏥 AI-powered analysis • 📊 Interactive maps • 🔍 District insights
          </div>
        </div>

      </div>

      {/* Fixed Warning and Example Queries Box at Bottom - Only show when no messages */}
      {messages.length === 0 && (
        <div style={{
          position: "fixed",
          bottom: "0",
          left: "0",
          right: "0",
          background: "linear-gradient(135deg, #fff8e1 0%, #f3e5ab 100%)",
          border: "1px solid #ffa000",
          borderBottom: "none",
          borderRadius: "12px 12px 0 0",
          padding: "clamp(16px, 3vw, 24px)",
          fontSize: "clamp(12px, 2.5vw, 14px)",
          lineHeight: "1.6",
          zIndex: 100,
          maxHeight: "40vh",
          overflowY: "auto",
          boxShadow: "0 -4px 20px rgba(255, 160, 0, 0.2)"
        }}>
          {/* Warning Section */}
          <div style={{
            display: "flex",
            alignItems: "center",
            gap: "8px",
            marginBottom: "16px",
            color: "#e65100",
            fontWeight: "600",
            flexWrap: "wrap"
          }}>
            <span style={{ fontSize: "clamp(16px, 3vw, 18px)", flexShrink: 0 }}>⚠️</span>
            <span style={{ fontSize: "clamp(16px, 2.5vw, 14px)" }}>
              Please note: The system may take some time to load and process your first query as the system is built on free resources.
            </span>
          </div>

          {/* Example Queries Section */}
          <div>
            <div style={{
              color: "#2E7D32",
              fontWeight: "600",
              marginBottom: "12px",
              fontSize: "clamp(13px, 2.8vw, 15px)"
            }}>
              💡 Here are some example questions you can ask our chatbot:
            </div>
            
            <div style={{
              display: "grid",
              gridTemplateColumns: "repeat(auto-fit, minmax(280px, 1fr))",
              gap: "12px"
            }}>
              {/* Geographical Constraints based Analysis */}
              <div style={{
                background: "rgba(255, 255, 255, 0.7)",
                borderRadius: "8px",
                padding: "clamp(8px, 2vw, 12px)",
                border: "1px solid rgba(46, 125, 50, 0.2)",
                cursor: "pointer",
                transition: "all 0.2s ease",
                minHeight: "fit-content"
              }}
              onClick={() => setQuery("Show me districts within 50 km of Delhi and their Anemia prevalence")}
              onMouseEnter={(e) => {
                e.target.style.background = "rgba(255, 255, 255, 0.9)";
                e.target.style.transform = "translateY(-2px)";
                e.target.style.boxShadow = "0 4px 12px rgba(25, 118, 210, 0.15)";
              }}
              onMouseLeave={(e) => {
                e.target.style.background = "rgba(255, 255, 255, 0.7)";
                e.target.style.transform = "translateY(0)";
                e.target.style.boxShadow = "none";
              }}
              >
                <div style={{
                  display: "flex",
                  alignItems: "center",
                  gap: "6px",
                  marginBottom: "6px",
                  color: "#1976D2",
                  fontWeight: "600",
                  fontSize: "clamp(11px, 2.2vw, 13px)",
                  flexWrap: "wrap"
                }}>
                  <span style={{ fontSize: "clamp(12px, 2.5vw, 14px)" }}>🗺️</span>
                  <span style={{ lineHeight: "1.3" }}>Geographical Constraints based Analysis</span>
                </div>
                <div style={{
                  fontStyle: "italic",
                  color: "#555",
                  fontSize: "clamp(11px, 2.2vw, 13px)",
                  lineHeight: "1.4",
                  wordBreak: "break-word"
                }}>
                  "Show me districts within 50 km of Delhi and their Anemia prevalence"
                </div>
              </div>

              {/* Border Districts Analysis */}
              <div style={{
                background: "rgba(255, 255, 255, 0.7)",
                borderRadius: "8px",
                padding: "clamp(8px, 2vw, 12px)",
                border: "1px solid rgba(46, 125, 50, 0.2)",
                cursor: "pointer",
                transition: "all 0.2s ease",
                minHeight: "fit-content"
              }}
              onClick={() => setQuery("Tell me the districts bordering West Bengal and their performance on Institutional Childbirth.")}
              onMouseEnter={(e) => {
                e.target.style.background = "rgba(255, 255, 255, 0.9)";
                e.target.style.transform = "translateY(-2px)";
                e.target.style.boxShadow = "0 4px 12px rgba(25, 118, 210, 0.15)";
              }}
              onMouseLeave={(e) => {
                e.target.style.background = "rgba(255, 255, 255, 0.7)";
                e.target.style.transform = "translateY(0)";
                e.target.style.boxShadow = "none";
              }}
              >
                <div style={{
                  display: "flex",
                  alignItems: "center",
                  gap: "6px",
                  marginBottom: "6px",
                  color: "#1976D2",
                  fontWeight: "600",
                  fontSize: "clamp(11px, 2.2vw, 13px)",
                  flexWrap: "wrap"
                }}>
                  <span style={{ fontSize: "clamp(12px, 2.5vw, 14px)" }}>🗺️</span>
                  <span style={{ lineHeight: "1.3" }}>Border Districts Analysis</span>
                </div>
                <div style={{
                  fontStyle: "italic",
                  color: "#555",
                  fontSize: "clamp(11px, 2.2vw, 13px)",
                  lineHeight: "1.4",
                  wordBreak: "break-word"
                }}>
                  "Tell me the districts bordering West Bengal and their performance on Institutional Childbirth."
                </div>
              </div>

              {/* Constraints based Analysis */}
              <div style={{
                background: "rgba(255, 255, 255, 0.7)",
                borderRadius: "8px",
                padding: "clamp(8px, 2vw, 12px)",
                border: "1px solid rgba(46, 125, 50, 0.2)",
                cursor: "pointer",
                transition: "all 0.2s ease",
                minHeight: "fit-content"
              }}
              onClick={() => setQuery("Tell me the districts with malnutrition greater than 30 and vaccination less than 50.")}
              onMouseEnter={(e) => {
                e.target.style.background = "rgba(255, 255, 255, 0.9)";
                e.target.style.transform = "translateY(-2px)";
                e.target.style.boxShadow = "0 4px 12px rgba(156, 39, 176, 0.15)";
              }}
              onMouseLeave={(e) => {
                e.target.style.background = "rgba(255, 255, 255, 0.7)";
                e.target.style.transform = "translateY(0)";
                e.target.style.boxShadow = "none";
              }}
              >
                <div style={{
                  display: "flex",
                  alignItems: "center",
                  gap: "6px",
                  marginBottom: "6px",
                  color: "#9C27B0",
                  fontWeight: "600",
                  fontSize: "clamp(11px, 2.2vw, 13px)",
                  flexWrap: "wrap"
                }}>
                  <span style={{ fontSize: "clamp(12px, 2.5vw, 14px)" }}>🔍</span>
                  <span style={{ lineHeight: "1.3" }}>Constraints based Analysis</span>
                </div>
                <div style={{
                  fontStyle: "italic",
                  color: "#555",
                  fontSize: "clamp(11px, 2.2vw, 13px)",
                  lineHeight: "1.4",
                  wordBreak: "break-word"
                }}>
                  "Tell me the districts with malnutrition greater than 30 and vaccination less than 50."
                </div>
              </div>

              {/* Nutrition & Wellness */}
              <div style={{
                background: "rgba(255, 255, 255, 0.7)",
                borderRadius: "8px",
                padding: "clamp(8px, 2vw, 12px)",
                border: "1px solid rgba(46, 125, 50, 0.2)",
                cursor: "pointer",
                transition: "all 0.2s ease",
                minHeight: "fit-content"
              }}
              onClick={() => setQuery("Show malnutrition trends in Uttar Pradesh from 2016 to 2021")}
              onMouseEnter={(e) => {
                e.target.style.background = "rgba(255, 255, 255, 0.9)";
                e.target.style.transform = "translateY(-2px)";
                e.target.style.boxShadow = "0 4px 12px rgba(233, 30, 99, 0.15)";
              }}
              onMouseLeave={(e) => {
                e.target.style.background = "rgba(255, 255, 255, 0.7)";
                e.target.style.transform = "translateY(0)";
                e.target.style.boxShadow = "none";
              }}
              >
                <div style={{
                  display: "flex",
                  alignItems: "center",
                  gap: "6px",
                  marginBottom: "6px",
                  color: "#E91E63",
                  fontWeight: "600",
                  fontSize: "clamp(11px, 2.2vw, 13px)",
                  flexWrap: "wrap"
                }}>
                  <span style={{ fontSize: "clamp(12px, 2.5vw, 14px)" }}>🍎</span>
                  <span style={{ lineHeight: "1.3" }}>Nutrition & Wellness</span>
                </div>
                <div style={{
                  fontStyle: "italic",
                  color: "#555",
                  fontSize: "clamp(11px, 2.2vw, 13px)",
                  lineHeight: "1.4",
                  wordBreak: "break-word"
                }}>
                  "Show malnutrition trends in Uttar Pradesh from 2016 to 2021"
                </div>
              </div>

              {/* Health Trends */}
              <div style={{
                background: "rgba(255, 255, 255, 0.7)",
                borderRadius: "8px",
                padding: "clamp(8px, 2vw, 12px)",
                border: "1px solid rgba(46, 125, 50, 0.2)",
                cursor: "pointer",
                transition: "all 0.2s ease",
                minHeight: "fit-content"
              }}
              onClick={() => setQuery("What are the top 10 districts with improving diabetes prevalence?")}
              onMouseEnter={(e) => {
                e.target.style.background = "rgba(255, 255, 255, 0.9)";
                e.target.style.transform = "translateY(-2px)";
                e.target.style.boxShadow = "0 4px 12px rgba(255, 152, 0, 0.15)";
              }}
              onMouseLeave={(e) => {
                e.target.style.background = "rgba(255, 255, 255, 0.7)";
                e.target.style.transform = "translateY(0)";
                e.target.style.boxShadow = "none";
              }}
              >
                <div style={{
                  display: "flex",
                  alignItems: "center",
                  gap: "6px",
                  marginBottom: "6px",
                  color: "#FF9800",
                  fontWeight: "600",
                  fontSize: "clamp(11px, 2.2vw, 13px)",
                  flexWrap: "wrap"
                }}>
                  <span style={{ fontSize: "clamp(12px, 2.5vw, 14px)" }}>📈</span>
                  <span style={{ lineHeight: "1.3" }}>Health Trends</span>
                </div>
                <div style={{
                  fontStyle: "italic",
                  color: "#555",
                  fontSize: "clamp(11px, 2.2vw, 13px)",
                  lineHeight: "1.4",
                  wordBreak: "break-word"
                }}>
                  "What are the top 10 districts with improving diabetes prevalence?"
                </div>
              </div>

              {/* Geographic Classification */}
              <div style={{
                background: "rgba(255, 255, 255, 0.7)",
                borderRadius: "8px",
                padding: "clamp(8px, 2vw, 12px)",
                border: "1px solid rgba(46, 125, 50, 0.2)",
                cursor: "pointer",
                transition: "all 0.2s ease",
                minHeight: "fit-content"
              }}
              onClick={() => setQuery("Create a map showing diabetes prevalence in Uttar Pradesh.")}
              onMouseEnter={(e) => {
                e.target.style.background = "rgba(255, 255, 255, 0.9)";
                e.target.style.transform = "translateY(-2px)";
                e.target.style.boxShadow = "0 4px 12px rgba(76, 175, 80, 0.15)";
              }}
              onMouseLeave={(e) => {
                e.target.style.background = "rgba(255, 255, 255, 0.7)";
                e.target.style.transform = "translateY(0)";
                e.target.style.boxShadow = "none";
              }}
              >
                <div style={{
                  display: "flex",
                  alignItems: "center",
                  gap: "6px",
                  marginBottom: "6px",
                  color: "#4CAF50",
                  fontWeight: "600",
                  fontSize: "clamp(11px, 2.2vw, 13px)",
                  flexWrap: "wrap"
                }}>
                  <span style={{ fontSize: "clamp(12px, 2.5vw, 14px)" }}>🗺️</span>
                  <span style={{ lineHeight: "1.3" }}>Geographic Classification</span>
                </div>
                <div style={{
                  fontStyle: "italic",
                  color: "#555",
                  fontSize: "clamp(11px, 2.2vw, 13px)",
                  lineHeight: "1.4",
                  wordBreak: "break-word"
                }}>
                  "Create a map showing diabetes prevalence in Uttar Pradesh."
                </div>
              </div>
            </div>
          </div>
        </div>
      )}

      {/* Modal Popup */}
      {modalOpen && modalData && (
        <div className="modal-overlay" onClick={() => setModalOpen(false)}>
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
                    background: isSaving ? '#6c757d' : 'linear-gradient(135deg, #2E7D32 0%, #4CAF50 100%)',
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
                    boxShadow: '0 2px 4px rgba(46, 125, 50, 0.2)',
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
                <button className="modal-close-button" onClick={() => setModalOpen(false)}>
                  ✕ Close
                </button>
              </div>
            </div>
            <div className="modal-content" ref={modalContentRef}>
              {modalType === 'map' && (
                <div className="modal-map-content">
                  <div className="visualization-container">
                    {isRadiusAnalysisRequest(modalData) ? (
                      (() => {
                        // Extract dynamic center point and radius from function calls or use intelligent defaults
                        const extractCenterAndRadius = (modalData) => {
                          // Check if we have explicit values
                          if (modalData.center_point && modalData.radius_km) {
                            return {
                              center_point: modalData.center_point,
                              radius_km: modalData.radius_km,
                              center_type: modalData.center_type || "district"
                            };
                          }
                          
                          // Try to extract from function calls
                          const functionCall = modalData.function_calls?.[0];
                          if (functionCall && functionCall.function === 'get_districts_within_radius') {
                            const args = functionCall.arguments || {};
                            return {
                              center_point: args.center_point || "Delhi",
                              radius_km: args.radius_km || 100,
                              center_type: args.center_point && args.center_point.includes(',') ? "coordinates" : "district"
                            };
                          }
                          
                          // Intelligent defaults based on available data
                          const districts = modalData.districts || modalData.boundary || [];
                          const boundaries = modalData.boundary_data || modalData.boundary || [];
                          
                          // If we have districts/boundaries, try to determine center and radius
                          if (boundaries.length > 0) {
                            // Check if most districts are Delhi-related
                            const delhiCount = boundaries.filter(b => 
                              (b.state_name && b.state_name.toLowerCase().includes('delhi')) ||
                              (b.district_name && b.district_name.toLowerCase().includes('delhi'))
                            ).length;
                            
                            if (delhiCount > boundaries.length * 0.3) { // If >30% are Delhi districts
                              return {
                                center_point: "Delhi",
                                radius_km: 100,
                                center_type: "district"
                              };
                            }
                          }
                          
                          // Final fallback
                          return {
                            center_point: "Delhi",
                            radius_km: 100,
                            center_type: "district"
                          };
                        };
                        
                        const centerInfo = extractCenterAndRadius(modalData);
                        
                        const radiusData = {
                          ...centerInfo,
                          query_type: modalData.query_type || "radius_analysis",
                          // Handle both new structure (districts) and old structure (boundary as districts)
                          districts: modalData.districts || (Array.isArray(modalData.boundary) ? modalData.boundary : []),
                          // Handle both new structure (boundary_data) and old structure (boundary)
                          boundary_data: modalData.boundary_data || modalData.boundary || [],
                          // Include chart data if available
                          chart_data: modalData.chart_data
                        };
                        
                        
                        
                        if (radiusData.districts && radiusData.districts.length > 0) {
                          console.log('');
                        }
                        
                        return (
                          <RadiusAnalysis 
                            radiusData={radiusData}
                            mapOnly={true}
                          />
                        );
                      })()
                    ) : isMultiDistrictRequest(modalData) ? (
                      <MultiDistrictAnalysis 
                        data={modalData}
                        mapOnly={true}
                      />
                    ) : isIndividualDistrictRequest(modalData) ? (
                      <MultiDistrictAnalysis 
                        data={modalData}
                        mapOnly={true}
                      />
                    ) : isBorderDistrictsRequest(modalData) ? (
                      <BorderDistrictsAnalysis 
                        borderData={modalData}
                        mapOnly={true}
                      />
                    ) : isStateWiseExtremesRequest(modalData) ? (
                      <StateWiseExtremes 
                        extremesData={modalData}
                        mapOnly={true}
                      />
                    ) : isConstraintBasedRequest(modalData) ? (
                      <ConstraintBasedAnalysis 
                        constraintData={modalData}
                        mapOnly={true}
                      />
                    ) : isTopBottomDistrictsRequest(modalData) ? (
                      <TopBottomAnalysis 
                        topBottomData={modalData}
                        mapOnly={true}
                      />
                    ) : false && isIndicatorChangeAnalysisRequest(modalData) ? (
                      // IndicatorChangeAnalysis temporarily disabled
                      <div>IndicatorChangeAnalysis disabled</div>
                    ) : isDistrictComparisonRequest(modalData) ? (
                      <DistrictComparisonAnalysis 
                        data={modalData}
                        mapOnly={true}
                      />
                    ) : isMultiIndicatorPerformanceRequest(modalData) ? (
                      <MultiIndicatorPerformance 
                        data={modalData}
                        mapOnly={true}
                      />
                    ) : isStateMultiIndicatorPerformanceRequest(modalData) ? (
                      <StateMultiIndicatorPerformance 
                        data={modalData}
                        mapOnly={true}
                      />
                    ) : isDistrictSimilarityAnalysisRequest(modalData) ? (
                      <DistrictSimilarityAnalysis 
                        data={modalData}
                        mapOnly={true}
                      />
                    ) : isDistrictClassificationRequest(modalData) ? (
                      <DistrictClassification 
                        data={modalData}
                        mapOnly={true}
                      />
                    ) : isDistrictClassificationChangeRequest(modalData) ? (
                      <DistrictClassificationChange 
                        data={modalData}
                        mapOnly={true}
                      />
                    ) : (
                      <p>Map visualization would be rendered here for other analysis types.</p>
                    )}
                  </div>
                </div>
              )}
              
              {modalType === 'chart' && (
                <div className="modal-chart-content">
                  <div className="visualization-container">
                    {isRadiusAnalysisRequest(modalData) ? (
                      (() => {
                        // Extract dynamic center point and radius from function calls or use intelligent defaults
                        const extractCenterAndRadius = (modalData) => {
                          // Check if we have explicit values
                          if (modalData.center_point && modalData.radius_km) {
                            return {
                              center_point: modalData.center_point,
                              radius_km: modalData.radius_km,
                              center_type: modalData.center_type || "district"
                            };
                          }
                          
                          // Try to extract from function calls
                          const functionCall = modalData.function_calls?.[0];
                          if (functionCall && functionCall.function === 'get_districts_within_radius') {
                            const args = functionCall.arguments || {};
                            return {
                              center_point: args.center_point || "Delhi",
                              radius_km: args.radius_km || 100,
                              center_type: args.center_point && args.center_point.includes(',') ? "coordinates" : "district"
                            };
                          }
                          
                          // Intelligent defaults
                          return {
                            center_point: "Delhi",
                            radius_km: 100,
                            center_type: "district"
                          };
                        };
                        
                        const centerInfo = extractCenterAndRadius(modalData);
                        
                        const radiusData = {
                          ...centerInfo,
                          query_type: modalData.query_type || "radius_analysis",
                          districts: modalData.districts || (Array.isArray(modalData.boundary) ? modalData.boundary : []),
                          boundary_data: modalData.boundary_data || modalData.boundary || [],
                          chart_data: modalData.chart_data
                        };
                        
                        
                        if (radiusData.chart_data) {
                          console.log('');
                        }
                        
                        return (
                          <RadiusAnalysis 
                            radiusData={radiusData}
                            chartOnly={true}
                          />
                        );
                      })()
                    ) : isMultiDistrictRequest(modalData) ? (
                      <MultiDistrictAnalysis 
                        data={modalData}
                        chartOnly={true}
                      />
                    ) : isIndividualDistrictRequest(modalData) ? (
                      <MultiDistrictAnalysis 
                        data={modalData}
                        chartOnly={true}
                      />
                    ) : isBorderDistrictsRequest(modalData) ? (
                      <BorderDistrictsAnalysis 
                        borderData={modalData}
                        chartOnly={true}
                      />
                    ) : isStateWiseExtremesRequest(modalData) ? (
                      <StateWiseExtremes 
                        extremesData={modalData}
                        chartOnly={true}
                      />
                    ) : isConstraintBasedRequest(modalData) ? (
                      <ConstraintBasedAnalysis 
                        constraintData={modalData}
                        chartOnly={true}
                      />
                    ) : isTopBottomDistrictsRequest(modalData) ? (
                      <TopBottomAnalysis 
                        topBottomData={modalData}
                        chartOnly={true}
                      />
                    ) : false && isIndicatorChangeAnalysisRequest(modalData) ? (
                      // IndicatorChangeAnalysis temporarily disabled
                      <div>IndicatorChangeAnalysis disabled</div>
                    ) : isDistrictComparisonRequest(modalData) ? (
                      <DistrictComparisonAnalysis 
                        data={modalData}
                        chartOnly={true}
                      />
                    ) : isMultiIndicatorPerformanceRequest(modalData) ? (
                      <MultiIndicatorPerformance 
                        data={modalData}
                        chartOnly={true}
                      />
                    ) : isStateMultiIndicatorPerformanceRequest(modalData) ? (
                      <StateMultiIndicatorPerformance 
                        data={modalData}
                        chartOnly={true}
                      />
                    ) : isDistrictSimilarityAnalysisRequest(modalData) ? (
                      <DistrictSimilarityAnalysis 
                        data={modalData}
                        chartOnly={true}
                      />
                    ) : isDistrictClassificationRequest(modalData) ? (
                      <DistrictClassification 
                        data={modalData}
                        chartOnly={true}
                      />
                    ) : isDistrictClassificationChangeRequest(modalData) ? (
                      <DistrictClassificationChange 
                        data={modalData}
                        chartOnly={true}
                      />
                    ) : (
                      <p>Chart visualization would be rendered here for other analysis types.</p>
                    )}
                  </div>
                </div>
              )}
            </div>
          </div>
        </div>
      )}
      
      {/* Global Save Status Display */}
      {(saveStatus || saveMessage) && !modalOpen && (
        <div className={`global-save-status ${saveStatus}`}>
          <span className="save-status-icon">
            {saveStatus === 'success' ? '✅' : saveStatus === 'error' ? '❌' : '⏳'}
          </span>
          <span className="save-status-message">{saveMessage}</span>
        </div>
      )}
    </div>
  );
}






