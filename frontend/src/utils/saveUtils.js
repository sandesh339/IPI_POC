import html2canvas from 'html2canvas';
import mapboxgl from 'mapbox-gl';

// Global map registry for save functionality
window.mapInstances = window.mapInstances || new Map();
window.mapReadyPromises = window.mapReadyPromises || new Map();

// Set up Mapbox GL to preserve drawing buffers globally
if (typeof mapboxgl !== 'undefined') {
  // Override the default Map constructor to enable preserveDrawingBuffer
  const OriginalMap = mapboxgl.Map;
  mapboxgl.Map = function(options) {
    const enhancedOptions = {
      ...options,
      preserveDrawingBuffer: true,
      antialias: true,
      failIfMajorPerformanceCaveat: false
    };
    return new OriginalMap(enhancedOptions);
  };
  
  // Copy all static properties and methods
  Object.keys(OriginalMap).forEach(key => {
    mapboxgl.Map[key] = OriginalMap[key];
  });
  
  // Set up prototype
  mapboxgl.Map.prototype = OriginalMap.prototype;
}

// Enhanced map initialization helper for consistent setup across components
export const initializeMapForCapture = (mapInstance, mapId) => {
  if (!mapInstance) return false;
  
  try {
    console.log(`Initializing map for capture: ${mapId}`);
    
    // Register the map instance
    registerMapInstance(mapId, mapInstance);
    window.mapboxMap = mapInstance;
    
    // Create a promise that resolves when map is fully ready
    const readyPromise = new Promise((resolve) => {
      let resolved = false;
      
      const checkReady = () => {
        if (resolved) return;
        
        if (mapInstance.isStyleLoaded && mapInstance.areTilesLoaded) {
          if (mapInstance.isStyleLoaded() && mapInstance.areTilesLoaded()) {
            resolved = true;
            resolve();
          }
        }
      };
      
      // Check immediately
      checkReady();
      
      // Listen for events
      mapInstance.on('idle', checkReady);
      mapInstance.on('render', checkReady);
      
      // Fallback timeout
      setTimeout(() => {
        if (!resolved) {
          resolved = true;
          resolve();
        }
      }, 5000);
    });
    
    window.mapReadyPromises.set(mapId, readyPromise);
    
    // Verify WebGL canvas has preserveDrawingBuffer
    const canvas = mapInstance.getCanvas();
    if (canvas) {
      const gl = canvas.getContext('webgl') || canvas.getContext('webgl2') || canvas.getContext('experimental-webgl');
      if (gl) {
        const attrs = gl.getContextAttributes();
        if (attrs.preserveDrawingBuffer) {
          console.log(`✅ WebGL preserveDrawingBuffer confirmed for ${mapId}`);
        } else {
          console.warn(`⚠️  WebGL preserveDrawingBuffer not enabled for ${mapId}`);
        }
      }
    }
    
    return true;
  } catch (error) {
    console.error('Error initializing map for capture:', error);
    return false;
  }
};

// Enhanced React Map GL initialization
export const initializeReactMapGLForCapture = (mapRef, mapId) => {
  if (!mapRef || !mapRef.current) return false;
  
  try {
    console.log(`Initializing React Map GL for capture: ${mapId}`);
    
    // Get the underlying mapbox map instance
    const mapInstance = mapRef.current.getMap();
    
    if (!mapInstance) {
      console.warn('Could not get map instance from React Map GL ref');
      return false;
    }
    
    // Initialize using the standard method
    return initializeMapForCapture(mapInstance, mapId);
  } catch (error) {
    console.error('Error initializing React Map GL for capture:', error);
    return false;
  }
};

// Main save function with robust error handling
export const saveMapVisualization = async (elementId, filename = 'map-visualization') => {
  try {
    console.log(`🎯 Starting save for: ${elementId}`);
    
    const element = document.getElementById(elementId) || 
                   document.querySelector(`[data-map-id="${elementId}"]`) || 
                   document.body;
    
    if (!element) {
      throw new Error(`Element with id "${elementId}" not found`);
    }
    
    // Wait for map to be ready if we have a ready promise
    if (window.mapReadyPromises.has(elementId)) {
      console.log('⏳ Waiting for map to be ready...');
      await window.mapReadyPromises.get(elementId);
    }
    
    // Find the map instance
    let mapInstance = getMapInstance(elementId) || window.mapboxMap;
    
    // Try to find react-map-gl instance
    if (!mapInstance) {
      const mapContainer = element.querySelector('.mapboxgl-map');
      if (mapContainer) {
        mapInstance = mapContainer._map;
      }
    }
    
    if (mapInstance && mapInstance.getCanvas) {
      console.log('🗺️  Found map instance, attempting direct canvas capture');
      
      // Get the canvas
      const canvas = mapInstance.getCanvas();
      if (!canvas) {
        throw new Error('Map canvas not found');
      }
      
      // Verify canvas has content
      const canvasWidth = canvas.width;
      const canvasHeight = canvas.height;
      
      console.log('Canvas dimensions:', { width: canvasWidth, height: canvasHeight });
      
      if (canvasWidth === 0 || canvasHeight === 0) {
        console.warn('Canvas has zero dimensions, forcing resize...');
        mapInstance.resize();
        await new Promise(resolve => setTimeout(resolve, 1000));
      }
      
      // Force a repaint to ensure canvas content is fresh
      mapInstance.triggerRepaint();
      await new Promise(resolve => setTimeout(resolve, 500));
      
      // Verify WebGL context
      const gl = canvas.getContext('webgl') || canvas.getContext('webgl2') || canvas.getContext('experimental-webgl');
      if (gl) {
        const attrs = gl.getContextAttributes();
        console.log('WebGL context attributes:', attrs);
        
        if (!attrs.preserveDrawingBuffer) {
          console.warn('⚠️  WebGL context does not preserve drawing buffer - attempting fallback');
          
          // Try to read pixels directly from WebGL
          const pixels = new Uint8Array(canvasWidth * canvasHeight * 4);
          gl.readPixels(0, 0, canvasWidth, canvasHeight, gl.RGBA, gl.UNSIGNED_BYTE, pixels);
          
          // Check if pixels are all zeros (blank)
          const hasContent = pixels.some(pixel => pixel !== 0);
          
          if (!hasContent) {
            console.warn('Canvas appears blank, using html2canvas fallback');
            return await saveWithHtml2Canvas(element, filename);
          }
        }
      }
      
      // Try direct canvas save
      try {
        return await saveCanvasDirectly(canvas, filename);
      } catch (canvasError) {
        console.warn('Direct canvas save failed:', canvasError);
        return await saveWithHtml2Canvas(element, filename);
      }
    }
    
    // Fallback to html2canvas
    console.log('📸 No map instance found, using html2canvas fallback');
    return await saveWithHtml2Canvas(element, filename);
    
  } catch (error) {
    console.error('❌ Error saving map visualization:', error);
    throw error;
  }
};

// Enhanced html2canvas save with map-specific options
const saveWithHtml2Canvas = async (element, filename) => {
  try {
    console.log('Using html2canvas for capture...');
    
    // Wait a bit to ensure rendering is complete
    await new Promise(resolve => setTimeout(resolve, 1000));
    
    const canvas = await html2canvas(element, {
      useCORS: true,
      allowTaint: true,
      scale: 1,
      backgroundColor: '#ffffff',
      logging: false,
      ignoreElements: (element) => {
        // Ignore attribution and other overlay elements that might cause issues
        return element.classList.contains('mapboxgl-ctrl-attrib') ||
               element.classList.contains('mapboxgl-ctrl-bottom-right') ||
               element.classList.contains('mapboxgl-ctrl');
      },
      onclone: (clonedDoc) => {
        // Ensure map containers are visible in the clone
        const mapElements = clonedDoc.querySelectorAll('.mapboxgl-map, [class*="map"]');
        mapElements.forEach(mapEl => {
          mapEl.style.visibility = 'visible';
          mapEl.style.opacity = '1';
        });
      }
    });
    
    return await saveCanvasDirectly(canvas, filename);
  } catch (error) {
    console.error('html2canvas failed:', error);
    throw error;
  }
};

// Enhanced save function specifically for React Map GL components
export const saveReactMapGLVisualization = async (elementId, filename = 'react-map-visualization') => {
  try {
    console.log(`🎯 Starting React Map GL save for: ${elementId}`);
    
    const element = document.getElementById(elementId) || 
                   document.querySelector(`[data-map-id="${elementId}"]`) || 
                   document.querySelector('.modal-map-content') ||
                   document.body;
    
    if (!element) {
      throw new Error(`Element with id "${elementId}" not found`);
    }
    
    // Wait for neighboring districts map to be ready if applicable
    if (window.neighboringDistrictsMapReady !== true) {
      console.log('⏳ Waiting for neighboring districts map to be fully ready...');
      
      // Wait up to 10 seconds for the map to be ready
      for (let i = 0; i < 20; i++) {
        if (window.neighboringDistrictsMapReady === true) {
          console.log('✅ Neighboring districts map is ready!');
          break;
        }
        await new Promise(resolve => setTimeout(resolve, 500));
      }
    }
    
    // Additional wait to ensure all components are rendered
    await new Promise(resolve => setTimeout(resolve, 2000));
    
    console.log('📸 Capturing React Map GL visualization with html2canvas...');
    
    const canvas = await html2canvas(element, {
      useCORS: true,
      allowTaint: true,
      scale: 1,
      backgroundColor: '#ffffff',
      logging: true, // Enable logging for debugging
      width: element.scrollWidth,
      height: element.scrollHeight,
      ignoreElements: (element) => {
        // Don't ignore popups or overlays that are part of the visualization
        return element.classList.contains('mapboxgl-ctrl-attrib') ||
               element.classList.contains('mapboxgl-ctrl-bottom-right') ||
               (element.classList.contains('mapboxgl-ctrl') && !element.classList.contains('legend'));
      },
      onclone: (clonedDoc) => {
        // Ensure all map and visualization elements are visible
        const mapElements = clonedDoc.querySelectorAll('.mapboxgl-map, [class*="map"], .legend, [class*="legend"], [class*="popup"]');
        mapElements.forEach(mapEl => {
          mapEl.style.visibility = 'visible';
          mapEl.style.opacity = '1';
          mapEl.style.display = 'block';
        });
        
        // Ensure React Map GL canvas is visible
        const canvasElements = clonedDoc.querySelectorAll('canvas');
        canvasElements.forEach(canvas => {
          canvas.style.visibility = 'visible';
          canvas.style.opacity = '1';
        });
      }
    });
    
    return await saveCanvasDirectly(canvas, filename);
    
  } catch (error) {
    console.error('❌ Error saving React Map GL visualization:', error);
    throw error;
  }
};

// Helper function to create Mapbox map with proper capture settings
export const createMapboxMapForCapture = (container, options = {}) => {
  const defaultOptions = {
    preserveDrawingBuffer: true,
    antialias: true,
    failIfMajorPerformanceCaveat: false,
    ...options
  };
  
  if (typeof mapboxgl !== 'undefined') {
    return new mapboxgl.Map({
      container,
      ...defaultOptions
    });
  }
  
  console.warn('Mapbox GL JS not available for direct initialization');
  return null;
};

// Map registry functions
export const registerMapInstance = (id, mapInstance) => {
  if (window.mapInstances) {
    window.mapInstances.set(id, mapInstance);
    console.log(`Map instance registered: ${id}`);
  }
};

export const getMapInstance = (id) => {
  return window.mapInstances ? window.mapInstances.get(id) : null;
};

export const getLatestMapInstance = () => {
  if (!window.mapInstances || window.mapInstances.size === 0) {
    return null;
  }
  
  const entries = Array.from(window.mapInstances.entries());
  return entries[entries.length - 1][1];
};

// Chart saving functions (keep existing functionality)
export const saveChartAsImage = async (chartRef, filename = 'chart', format = 'png') => {
  try {
    if (!chartRef || !chartRef.current) {
      throw new Error('Chart reference not found');
    }

    const canvas = chartRef.current.canvas;
    if (!canvas) {
      throw new Error('Chart canvas not found');
    }

    return await saveCanvasDirectly(canvas, filename, format);
  } catch (error) {
    console.error('Error saving chart:', error);
    throw error;
  }
};

export const saveElementAsImage = async (element, filename = 'visualization', format = 'png', options = {}) => {
  try {
    if (!element) {
      throw new Error('Element not found');
    }

    // Check for multiple charts or single chart (but not for maps)
    const chartCanvases = element.querySelectorAll('canvas[role="img"], canvas');
    const hasMap = element.querySelector('.mapboxgl-map, .leaflet-container, [class*="map"]');
    
    if (chartCanvases.length > 0 && !hasMap) {
      if (chartCanvases.length === 1 && chartCanvases[0].width > 0 && chartCanvases[0].height > 0) {
        console.log('Using direct canvas approach for single Chart.js visualization');
        return await saveCanvasDirectly(chartCanvases[0], filename, format);
      } else if (chartCanvases.length > 1) {
        console.log(`Found ${chartCanvases.length} charts, using html2canvas for combined capture`);
        // Continue with html2canvas to capture multiple charts together
      }
    }
    
    // For chart containers, try to exclude modal UI elements by focusing on chart content
    if (!hasMap && element.classList.contains('modal-chart-content')) {
      console.log('Detected modal chart content, attempting to exclude UI elements');
      
      // Try to find the actual chart content within the modal
      const chartContent = 
        element.querySelector('.charts-container') ||
        element.querySelector('.chart-grid') ||
        element.querySelector('.visualization-area') ||
        element.querySelector('.recharts-wrapper') ||
        element.querySelector('.chart-container') ||
        element.querySelector('[class*="chart"]:not([class*="modal"])') ||
        element.querySelector('div > div:first-child'); // First child div, likely chart content
      
      if (chartContent && chartContent !== element) {
        console.log('Found chart content within modal, using that instead');
        element = chartContent;
      }
    }

    // Use html2canvas for everything else
    const defaultOptions = {
      useCORS: true,
      allowTaint: true,
      scale: 1,
      backgroundColor: '#ffffff',
      logging: false,
      ...options
    };

    console.log('Using html2canvas for element capture');
    
    const canvas = await html2canvas(element, defaultOptions);
    
    if (!canvas || canvas.width === 0 || canvas.height === 0) {
      throw new Error('html2canvas produced empty canvas');
    }

    return await saveCanvasDirectly(canvas, filename, format);
  } catch (error) {
    console.error('Error saving element as image:', error);
    throw error;
  }
};

export const saveCanvasDirectly = async (canvas, filename = 'chart', format = 'png') => {
  try {
    if (!canvas) {
      throw new Error('Canvas not found');
    }

    console.log('Saving canvas directly:', { 
      width: canvas.width, 
      height: canvas.height, 
      hasWebGL: !!canvas.getContext('webgl') 
    });

    if (canvas.width === 0 || canvas.height === 0) {
      throw new Error('Canvas has zero dimensions');
    }

    // For WebGL canvases, log context information
    const gl = canvas.getContext('webgl') || canvas.getContext('webgl2') || canvas.getContext('experimental-webgl');
    if (gl) {
      const attrs = gl.getContextAttributes();
      console.log('WebGL context attributes:', attrs);
      
      if (!attrs.preserveDrawingBuffer) {
        console.warn('WebGL context does not preserve drawing buffer - image may be blank');
      }
    }

    // Convert canvas to blob
    return new Promise((resolve, reject) => {
      canvas.toBlob((blob) => {
        if (blob) {
          const url = URL.createObjectURL(blob);
          const link = document.createElement('a');
          link.href = url;
          link.download = `${filename}.${format}`;
          document.body.appendChild(link);
          link.click();
          document.body.removeChild(link);
          URL.revokeObjectURL(url);
          console.log(`✅ Canvas saved successfully as ${filename}.${format}`);
          resolve({ success: true, filename: `${filename}.${format}` });
        } else {
          const error = new Error('Failed to convert canvas to blob');
          console.error('Canvas to blob conversion failed');
          reject(error);
        }
      }, `image/${format}`, 0.95);
    });
  } catch (error) {
    console.error('Error saving canvas directly:', error);
    throw error;
  }
};

// Save map with legend as composite image
export const saveMapWithLegend = async (mapCanvas, legendElement, filename = 'map', format = 'png') => {
  try {
    if (!mapCanvas || !legendElement) {
      throw new Error('Map canvas or legend element not found');
    }

    // Create a new canvas to combine map and legend
    const combinedCanvas = document.createElement('canvas');
    const ctx = combinedCanvas.getContext('2d');
    
    // Get legend canvas using html2canvas
    const legendCanvas = await html2canvas(legendElement, {
      useCORS: true,
      allowTaint: true,
      backgroundColor: null
    });
    
    // Set combined canvas size
    combinedCanvas.width = mapCanvas.width;
    combinedCanvas.height = mapCanvas.height;
    
    // Draw map first
    ctx.drawImage(mapCanvas, 0, 0);
    
    // Draw legend in bottom right corner
    const legendX = mapCanvas.width - legendCanvas.width - 10;
    const legendY = mapCanvas.height - legendCanvas.height - 10;
    ctx.drawImage(legendCanvas, legendX, legendY);
    
    return await saveCanvasDirectly(combinedCanvas, filename, format);
  } catch (error) {
    console.error('Error saving map with legend:', error);
    throw error;
  }
};

// Backend visualization saving (keep existing)
export const saveVisualizationToBackend = async (visualizationData, metadata = {}) => {
  try {
    const response = await fetch('/api/save_visualization/', {
      method: 'POST',
      headers: {
        'Content-Type': 'application/json',
      },
      body: JSON.stringify({
        visualization_data: visualizationData,
        metadata: metadata
      })
    });

    if (!response.ok) {
      throw new Error(`HTTP error! status: ${response.status}`);
    }

    const result = await response.json();
    return result;
  } catch (error) {
    console.error('Error saving visualization to backend:', error);
    throw error;
  }
};

// Get saved visualizations from backend
export const getSavedVisualizations = async () => {
  try {
    const response = await fetch('/api/get_saved_visualizations/');
    
    if (!response.ok) {
      throw new Error(`HTTP error! status: ${response.status}`);
    }

    const result = await response.json();
    return result;
  } catch (error) {
    console.error('Error getting saved visualizations:', error);
    throw error;
  }
};

// Generate filename based on visualization type
export const generateFilename = (visualizationType, data = {}) => {
  const timestamp = new Date().toISOString().slice(0, 19).replace(/:/g, '-');
  const sdgGoal = data.sdg_goal ? `_SDG${data.sdg_goal}` : '';
  const district = data.district ? `_${data.district.replace(/\s+/g, '-')}` : '';
  
  return `${visualizationType}${sdgGoal}${district}_${timestamp}`;
};

// Save multiple charts in a single element
export const saveMultipleCharts = async (element, filename) => {
  try {
    // Use html2canvas to capture the entire element containing multiple charts
    const canvas = await html2canvas(element, {
      useCORS: true,
      allowTaint: true,
      scale: 1,
      backgroundColor: '#ffffff'
    });
    
    return await saveCanvasDirectly(canvas, filename);
  } catch (error) {
    console.error('Error saving multiple charts:', error);
    throw error;
  }
};

// Save visualization in multiple formats
export const saveVisualizationMultiFormat = async (element, chartRef, filename, visualizationData) => {
  try {
    const results = [];
    
    // Save as PNG
    const pngResult = await saveElementAsImage(element, `${filename}_png`, 'png');
    results.push(pngResult);
    
    // Save as JPEG
    const jpegResult = await saveElementAsImage(element, `${filename}_jpeg`, 'jpeg');
    results.push(jpegResult);
    
    // Save data to backend if provided
    if (visualizationData) {
      const backendResult = await saveVisualizationToBackend(visualizationData, {
        filename: filename,
        timestamp: new Date().toISOString()
      });
      results.push(backendResult);
    }
    
    return results;
  } catch (error) {
    console.error('Error saving visualization in multiple formats:', error);
    throw error;
  }
};

// Debug function to check map readiness
export const debugMapSaveReadiness = (mapInstance) => {
  if (!mapInstance) {
    console.log('❌ No map instance provided');
    return false;
  }
  
  const canvas = mapInstance.getCanvas();
  if (!canvas) {
    console.log('❌ No canvas found');
    return false;
  }
  
  const gl = canvas.getContext('webgl') || canvas.getContext('webgl2') || canvas.getContext('experimental-webgl');
  if (gl) {
    const attrs = gl.getContextAttributes();
    console.log('🔍 WebGL context attributes:', attrs);
    
    if (!attrs.preserveDrawingBuffer) {
      console.log('⚠️  preserveDrawingBuffer is false - maps may save as blank');
    } else {
      console.log('✅ preserveDrawingBuffer is true - maps should save properly');
    }
  }
  
  console.log('Canvas info:', {
    width: canvas.width,
    height: canvas.height,
    style: {
      width: canvas.style.width,
      height: canvas.style.height
    }
  });
  
  return true;
}; 