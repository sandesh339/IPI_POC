import React, { useState, useMemo, useEffect, useRef } from 'react';
import { Bar, Line, Scatter, Pie } from 'react-chartjs-2';
import mapboxgl from 'mapbox-gl';
import 'mapbox-gl/dist/mapbox-gl.css';
import { registerMapInstance, initializeMapForCapture } from '../utils/saveUtils';

import {
  Chart as ChartJS,
  CategoryScale,
  LinearScale,
  BarElement,
  LineElement,
  PointElement,
  ArcElement,
  Title,
  Tooltip,
  Legend
} from 'chart.js';

// Register Chart.js components
ChartJS.register(
  CategoryScale,
  LinearScale,
  BarElement,
  LineElement,
  PointElement,
  ArcElement,
  Title,
  Tooltip,
  Legend
);

// Set Mapbox access token
mapboxgl.accessToken = process.env.REACT_APP_MAPBOX_TOKEN;

// Color schemes for radius analysis
const RADIUS_COLORS = {
  center: '#e74c3c',          // Red for center point
  radius_circle: '#3498db',   // Blue for radius circle
  district_good: '#27ae60',   // Green for good performance
  district_poor: '#e67e22',   // Orange for poor performance
  district_avg: '#95a5a6',    // Gray for average performance
  improvement: '#2ecc71',     // Green for improvement
  decline: '#e74c3c'          // Red for decline
};

// Enhanced state colors for state-wise visualization
const STATE_COLORS = [
  "#E74C3C", "#3498DB", "#2ECC71", "#F39C12", "#9B59B6", 
  "#1ABC9C", "#E67E22", "#34495E", "#E91E63", "#00BCD4",
  "#F1C40F", "#8E44AD", "#E67E22", "#2C3E50", "#D35400"
];

// Map Component for Radius Analysis
const RadiusAnalysisMap = ({ data, fullScreen = false }) => {
  const mapContainer = useRef(null);
  const map = useRef(null);

  useEffect(() => {
    if (!data || !data.boundary_data || map.current) return;

    console.log('Initializing radius analysis map with data:', data);
    console.log('Sample boundary_data:', data.boundary_data[0]);
    console.log('Districts data:', data.districts);
    console.log('Districts count:', data.districts?.length);
    if (data.districts && data.districts.length > 0) {
      console.log('Sample district:', data.districts[0]);
      console.log('Sample district indicators:', data.districts[0].indicators);
    }

    // Determine center coordinates dynamically
    let centerCoords;
    
    // Method 1: If we have explicit center coordinates object from backend (PREFERRED)
    if (data.center_coordinates) {
      const lng = parseFloat(data.center_coordinates.lng);
      const lat = parseFloat(data.center_coordinates.lat);
      if (!isNaN(lng) && !isNaN(lat)) {
        centerCoords = [lng, lat];
        console.log('Using center coordinates from backend:', centerCoords);
      }
    }
    
    // Method 2: If coordinates are provided directly as string
    if (!centerCoords && data.center_type === 'coordinates' && data.center_point) {
      const coords = data.center_point.split(',');
      if (coords.length === 2) {
        const lng = parseFloat(coords[1]);
        const lat = parseFloat(coords[0]);
        if (!isNaN(lng) && !isNaN(lat)) {
          centerCoords = [lng, lat]; // [lng, lat]
          console.log('Using coordinates from center_point string:', centerCoords);
        }
      }
    }
    
    // Method 3: Try to find center from boundary data for the specified center point
    if (!centerCoords && data.center_point && data.boundary_data && data.center_type === "district") {
      console.log('Looking for center district:', data.center_point, 'in boundary data');
      
      // Clean the center point name for better matching
      const cleanCenterPoint = data.center_point.toLowerCase().trim();
      
      const centerDistrict = data.boundary_data.find(b => {
        const district1 = (b.district || '').toLowerCase().trim();
        const district2 = (b.district_name || '').toLowerCase().trim();
        
        // Exact match first
        if (district1 === cleanCenterPoint || district2 === cleanCenterPoint) {
          return true;
        }
        
        // Partial match (either direction)
        return district1.includes(cleanCenterPoint) || 
               district2.includes(cleanCenterPoint) ||
               cleanCenterPoint.includes(district1) ||
               cleanCenterPoint.includes(district2);
      });
      
      console.log('Found center district boundary data:', centerDistrict?.district_name || centerDistrict?.district);
      
      if (centerDistrict && centerDistrict.geometry) {
        try {
          const geometry = typeof centerDistrict.geometry === 'string' 
            ? JSON.parse(centerDistrict.geometry) 
            : centerDistrict.geometry;
          
          if (geometry.coordinates && geometry.coordinates[0] && Array.isArray(geometry.coordinates[0])) {
            // Calculate centroid of the polygon
            const coords = geometry.coordinates[0];
            if (coords.length > 0) {
              const centerLng = coords.reduce((sum, coord) => {
                return sum + (Array.isArray(coord) && coord.length >= 2 ? coord[0] : 0);
              }, 0) / coords.length;
              const centerLat = coords.reduce((sum, coord) => {
                return sum + (Array.isArray(coord) && coord.length >= 2 ? coord[1] : 0);
              }, 0) / coords.length;
              
              if (!isNaN(centerLng) && !isNaN(centerLat) && isFinite(centerLng) && isFinite(centerLat)) {
                centerCoords = [centerLng, centerLat];
                console.log('Calculated center coordinates from district geometry:', centerCoords);
              }
            }
          }
        } catch (e) {
          console.warn('Could not parse geometry for center district:', e);
        }
      }
    }
    
    // Method 4: Major Indian cities lookup
    if (!centerCoords && data.center_point) {
      const cityCoordinates = {
        'delhi': [77.2090, 28.6139],
        'new delhi': [77.2090, 28.6139],
        'mumbai': [72.8777, 19.0760],
        'bangalore': [77.5946, 12.9716],
        'bengaluru': [77.5946, 12.9716],
        'chennai': [80.2707, 13.0827],
        'kolkata': [88.3639, 22.5726],
        'hyderabad': [78.4867, 17.3850],
        'pune': [73.8567, 18.5204],
        'jaipur': [75.7873, 26.9124],
        'lucknow': [80.9462, 26.8467],
        'chandigarh': [76.7794, 30.7333],
        'bhopal': [77.4126, 23.2599],
        'indore': [75.8577, 22.7196],
        'ahmedabad': [72.5714, 23.0225],
        'surat': [72.8311, 21.1702],
        'gurgaon': [77.0266, 28.4595],
        'gurugram': [77.0266, 28.4595],
        'noida': [77.3910, 28.5355],
        'faridabad': [77.3178, 28.4089],
        'ghaziabad': [77.4538, 28.6692],
        'nagpur': [79.0882, 21.1458],
        'patna': [85.1376, 25.5941],
        'bhubaneswar': [85.8245, 20.2961],
        'raipur': [81.6296, 21.2514]
      };
      
      const centerPointLower = data.center_point.toLowerCase();
      for (const [city, coords] of Object.entries(cityCoordinates)) {
        if (centerPointLower.includes(city) || city.includes(centerPointLower)) {
          centerCoords = coords;
          break;
        }
      }
    }
    
    // Method 5: Calculate geographic center from all boundary data
    if (!centerCoords && data.boundary_data && data.boundary_data.length > 0) {
      let totalLng = 0, totalLat = 0, count = 0;
      
      data.boundary_data.forEach(boundary => {
        try {
          const geometry = typeof boundary.geometry === 'string' 
            ? JSON.parse(boundary.geometry) 
            : boundary.geometry;
            
          if (geometry && geometry.coordinates && geometry.coordinates[0] && Array.isArray(geometry.coordinates[0])) {
            const coords = geometry.coordinates[0];
            if (coords.length > 0) {
              const boundaryLng = coords.reduce((sum, coord) => sum + (Array.isArray(coord) && coord.length >= 2 ? coord[0] : 0), 0) / coords.length;
              const boundaryLat = coords.reduce((sum, coord) => sum + (Array.isArray(coord) && coord.length >= 2 ? coord[1] : 0), 0) / coords.length;
              
              // Validate calculated coordinates
              if (!isNaN(boundaryLng) && !isNaN(boundaryLat) && isFinite(boundaryLng) && isFinite(boundaryLat)) {
                totalLng += boundaryLng;
                totalLat += boundaryLat;
                count++;
              }
            }
          }
        } catch (e) {
          // Skip invalid geometries
          console.warn('Error processing boundary geometry for center calculation:', e);
        }
      });
      
      if (count > 0) {
        const avgLng = totalLng / count;
        const avgLat = totalLat / count;
        
        // Final validation before assignment
        if (!isNaN(avgLng) && !isNaN(avgLat) && isFinite(avgLng) && isFinite(avgLat)) {
          centerCoords = [avgLng, avgLat];
        }
      }
    }
    
    // Final fallback: Center of India
    if (!centerCoords || !Array.isArray(centerCoords) || centerCoords.length !== 2 || 
        isNaN(centerCoords[0]) || isNaN(centerCoords[1]) || 
        !isFinite(centerCoords[0]) || !isFinite(centerCoords[1])) {
      console.warn('Invalid center coordinates detected, using fallback:', centerCoords);
      centerCoords = [78.9629, 20.5937];
    }
    
    console.log('Dynamic center coordinates determined:', centerCoords, 'for center_point:', data.center_point);

    // Calculate appropriate zoom level based on radius
    const calculateZoomLevel = (radiusKm) => {
      // Dynamic zoom calculation based on radius
      if (radiusKm <= 25) return 10;      // Very close view for small radius
      if (radiusKm <= 50) return 9;       // Close view
      if (radiusKm <= 100) return 8;      // Medium view
      if (radiusKm <= 200) return 7;      // Default view
      if (radiusKm <= 400) return 6;      // Wide view
      if (radiusKm <= 800) return 5;      // Very wide view
      return 4;                           // Country level view
    };

    const initialZoom = calculateZoomLevel(data.radius_km || 100);

    // Initialize map
    map.current = new mapboxgl.Map({
      container: mapContainer.current,
      style: 'mapbox://styles/mapbox/light-v11',
      center: centerCoords,
      zoom: initialZoom,
      preserveDrawingBuffer: true // Essential for WebGL canvas capture
    });
    
    console.log('Map initialized with zoom level:', initialZoom, 'for radius:', data.radius_km || 100, 'km');

    // Register the map instance for save functionality with multiple IDs for better capture
    registerMapInstance('radius-analysis-map', map.current);
    registerMapInstance('modal-map-content', map.current);
    registerMapInstance('modal-map-0', map.current);
    
    // Also attach to window for global access
    window.mapboxMap = map.current;
    
    // Initialize for capture with proper timing
    setTimeout(() => {
      initializeMapForCapture(map.current, 'radius-analysis-map');
    }, 1000);

    map.current.on('load', () => {
      console.log('Radius analysis map loaded, initializing for capture...');
      
      // Initialize for capture once map is fully loaded
      setTimeout(() => {
        initializeMapForCapture(map.current, 'radius-analysis-map');
        // Trigger a repaint to ensure canvas is ready
        map.current.triggerRepaint();
      }, 500);
      
      // Calculate dynamic radius
      let effectiveRadius = 100; // Default fallback
      
      // Method 1: Use explicit radius if provided
      if (data.radius_km && data.radius_km > 0) {
        effectiveRadius = data.radius_km;
      }
      // Method 2: Estimate radius from district spread if no explicit radius
      else if (data.boundary_data && data.boundary_data.length > 0 && centerCoords) {
        let maxDistance = 0;
        let validDistances = 0;
        
        data.boundary_data.forEach(boundary => {
          try {
            const geometry = typeof boundary.geometry === 'string' 
              ? JSON.parse(boundary.geometry) 
              : boundary.geometry;
              
            if (geometry && geometry.coordinates && geometry.coordinates[0]) {
              const coords = geometry.coordinates[0];
              // Calculate centroid of this boundary
              const boundaryLng = coords.reduce((sum, coord) => sum + coord[0], 0) / coords.length;
              const boundaryLat = coords.reduce((sum, coord) => sum + coord[1], 0) / coords.length;
              
              // Calculate distance from center to this boundary centroid
              // Validate coordinates before calculation
              if (isNaN(boundaryLng) || isNaN(boundaryLat) || isNaN(centerCoords[0]) || isNaN(centerCoords[1])) {
                console.warn('Invalid coordinates for distance calculation:', {
                  boundaryLng, boundaryLat, centerCoords
                });
                return 0; // Skip this boundary
              }
              
              const distance = Math.sqrt(
                Math.pow((boundaryLng - centerCoords[0]) * 111 * Math.cos(centerCoords[1] * Math.PI / 180), 2) + 
                Math.pow((boundaryLat - centerCoords[1]) * 111, 2)
              );
              
              maxDistance = Math.max(maxDistance, distance);
              validDistances++;
            }
          } catch (e) {
            // Skip invalid geometries
          }
        });
        
        if (validDistances > 0 && maxDistance > 0) {
          // Add 20% buffer and round to nearest 25km for clean values
          const estimatedRadius = Math.ceil((maxDistance * 1.2) / 25) * 25;
          effectiveRadius = Math.max(estimatedRadius, 50); // Minimum 50km radius
          console.log('Estimated radius from district spread:', effectiveRadius, 'km (max distance:', maxDistance.toFixed(1), 'km)');
        }
      }
      
      console.log('Dynamic radius determined:', effectiveRadius, 'km');
      
      // Add radius circle using a simple approach
      // Validate centerCoords before creating radius circle
      const safeCenterCoords = (centerCoords && Array.isArray(centerCoords) && 
                               centerCoords.length === 2 && 
                               !isNaN(centerCoords[0]) && !isNaN(centerCoords[1]) && 
                               isFinite(centerCoords[0]) && isFinite(centerCoords[1])) 
                               ? centerCoords 
                               : [78.9629, 20.5937];
      
      if (centerCoords !== safeCenterCoords) {
        console.warn('Using fallback center coordinates for radius circle');
      }
      
      const radiusInDegrees = effectiveRadius / 111; // Rough conversion km to degrees
      const circleCoords = [];
      const steps = 64;
      
      for (let i = 0; i <= steps; i++) {
        const angle = (i * 360) / steps;
        const x = safeCenterCoords[0] + radiusInDegrees * Math.cos(angle * Math.PI / 180);
        const y = safeCenterCoords[1] + radiusInDegrees * Math.sin(angle * Math.PI / 180);
        
        // Validate calculated coordinates
        if (!isNaN(x) && !isNaN(y) && isFinite(x) && isFinite(y)) {
          circleCoords.push([x, y]);
        }
      }

      const circle = {
        type: 'Feature',
        geometry: {
          type: 'Polygon',
          coordinates: [circleCoords]
        }
      };

      map.current.addSource('radius-circle', {
        type: 'geojson',
        data: circle
      });

      map.current.addLayer({
        id: 'radius-circle-fill',
        type: 'fill',
        source: 'radius-circle',
        paint: {
          'fill-color': RADIUS_COLORS.radius_circle,
          'fill-opacity': 0.1
        }
      });

      map.current.addLayer({
        id: 'radius-circle-stroke',
        type: 'line',
        source: 'radius-circle',
        paint: {
          'line-color': RADIUS_COLORS.radius_circle,
          'line-width': 3,
          'line-dasharray': [2, 2]
        }
      });

      // Add center point marker
      // Validate centerCoords before using for GeoJSON
      const validCenterCoords = (centerCoords && Array.isArray(centerCoords) && 
                                centerCoords.length === 2 && 
                                !isNaN(centerCoords[0]) && !isNaN(centerCoords[1]) && 
                                isFinite(centerCoords[0]) && isFinite(centerCoords[1])) 
                                ? centerCoords 
                                : [78.9629, 20.5937];
      
      if (centerCoords !== validCenterCoords) {
        console.warn('Using fallback center coordinates for center point marker');
      }
      
      map.current.addSource('center-point', {
        type: 'geojson',
        data: {
          type: 'Feature',
          geometry: {
            type: 'Point',
            coordinates: validCenterCoords
          },
          properties: {
            title: data.center_point,
            type: 'center'
          }
        }
      });

      map.current.addLayer({
        id: 'center-point',
        type: 'circle',
        source: 'center-point',
        paint: {
          'circle-color': RADIUS_COLORS.center,
          'circle-radius': 8,
          'circle-stroke-color': '#ffffff',
          'circle-stroke-width': 2
        }
      });

      // Create GeoJSON for districts within radius
      const allStates = [...new Set(data.boundary_data?.map(b => b.state || b.state_name) || [])];
      const stateColors = {};
      allStates.forEach((state, index) => {
        stateColors[state] = STATE_COLORS[index % STATE_COLORS.length];
      });

      // Parse geometry data - handle both GeoJSON objects and WKB hex strings
      const parseGeometry = (geom) => {
        if (!geom) return null;
        
        // If it's already a GeoJSON geometry object
        if (typeof geom === 'object' && geom.type && geom.coordinates) {
          // Validate coordinates exist and are not empty
          if (Array.isArray(geom.coordinates) && geom.coordinates.length > 0) {
            return geom;
          }
        }
        
        // If it's a WKB hex string, we need to convert it
        if (typeof geom === 'string' && geom.length > 0) {
          try {
            // Try to parse as JSON first (in case it's a stringified GeoJSON)
            const parsed = JSON.parse(geom);
            if (parsed.type && parsed.coordinates && Array.isArray(parsed.coordinates)) {
              return parsed;
            }
          } catch (e) {
            // Not JSON, might be WKB hex - for now, skip this boundary
            console.warn('Cannot parse geometry for radius analysis:', geom.substring(0, 50) + '...');
            return null;
          }
        }
        
        return null;
      };

      // Only include boundaries with valid geometry
      const districtsGeoJSON = {
        type: 'FeatureCollection',
        features: data.boundary_data
          ?.map((boundary) => {
            try {
              // Parse geometry first
              const parsedGeometry = parseGeometry(boundary.geometry);
              if (!parsedGeometry) {
                return null;
              }
              return {
                ...boundary,
                geometry: parsedGeometry
              };
            } catch (error) {
              console.warn('Error processing boundary in radius analysis:', boundary.district_name || 'unknown', error);
              return null;
            }
          })
          ?.filter(boundary => {
            if (!boundary) return false;
            const hasValidGeometry = boundary.geometry && 
                                   boundary.geometry.type && 
                                   boundary.geometry.coordinates &&
                                   Array.isArray(boundary.geometry.coordinates) &&
                                   boundary.geometry.coordinates.length > 0;
            return hasValidGeometry;
          })
          ?.map((boundary) => {
            const districtName = boundary.district_name || boundary.district;
            const districtData = data.districts?.find(d => d.district_name === districtName);

            // Calculate a simple performance score based on indicators
            let performanceScore = 50;
            if (districtData && districtData.indicators && districtData.indicators.length > 0) {
              // Use the first indicator's 2021 value as a rough performance metric
              const firstIndicator = districtData.indicators[0];
              if (firstIndicator.prevalence_2021 !== undefined) {
                // Normalize to 0-100 scale (this is a simple approximation)
                performanceScore = Math.min(100, Math.max(0, firstIndicator.prevalence_2021 * 10));
              }
            }

            return {
              type: 'Feature',
              properties: {
                district_name: districtName,
                state_name: boundary.state_name || boundary.state || 'Unknown State',
                distance_km: districtData?.distance_km || 0,
                performance_score: performanceScore,
                state_color: stateColors[boundary.state_name || boundary.state] || '#cccccc',
                has_data: !!districtData,
                // Add visual distinction for districts without health data
                opacity: districtData ? 0.8 : 0.4,
                strokeWidth: districtData ? 2 : 1
              },
              geometry: boundary.geometry
            };
          }) || []
      };

      console.log('districtsGeoJSON created with features:', districtsGeoJSON.features.length);

      // Add districts source
      map.current.addSource('radius-districts', {
        type: 'geojson',
        data: districtsGeoJSON
      });

      // Update the fill layer to use state colors directly
      map.current.addLayer({
        id: 'radius-districts-fill',
        type: 'fill',
        source: 'radius-districts',
        paint: {
          'fill-color': [
            'match',
            ['get', 'state_name'],
            ...allStates.reduce((acc, state) => [...acc, state, stateColors[state]], []),
            '#cccccc' // Default color for unknown states
          ],
          'fill-opacity': [
            'case',
            ['get', 'has_data'],
            0.7, // Full opacity for districts with health data
            0.4  // Reduced opacity for districts without health data
          ]
        }
      });

      // Add district stroke layer
      map.current.addLayer({
        id: 'radius-districts-stroke',
        type: 'line',
        source: 'radius-districts',
        paint: {
          'line-color': '#ffffff',
          'line-width': [
            'case',
            ['get', 'has_data'],
            2, // Thicker stroke for districts with health data
            1  // Thinner stroke for districts without health data
          ]
        }
      });

      // Add hover effect
      map.current.addLayer({
        id: 'radius-districts-hover',
        type: 'fill',
        source: 'radius-districts',
        paint: {
          'fill-color': '#000000',
          'fill-opacity': [
            'case',
            ['boolean', ['feature-state', 'hover'], false],
            0.3,
            0
          ]
        }
      });

      let hoveredStateId = null;

      // Handle hover states
      map.current.on('mousemove', 'radius-districts-fill', (e) => {
        if (e.features.length > 0) {
          if (hoveredStateId !== null) {
            map.current.setFeatureState(
              { source: 'radius-districts', id: hoveredStateId },
              { hover: false }
            );
          }
          hoveredStateId = e.features[0].id;
          map.current.setFeatureState(
            { source: 'radius-districts', id: hoveredStateId },
            { hover: true }
          );
        }
      });

      map.current.on('mouseleave', 'radius-districts-fill', () => {
        if (hoveredStateId !== null) {
          map.current.setFeatureState(
            { source: 'radius-districts', id: hoveredStateId },
            { hover: false }
          );
        }
        hoveredStateId = null;
      });

      // Add popup on click
      map.current.on('click', 'radius-districts-fill', (e) => {
        // Validate event and coordinates
        if (!e.features || !e.features[0] || !e.lngLat) {
          console.warn('Invalid click event data');
          return;
        }

        const lng = e.lngLat.lng;
        const lat = e.lngLat.lat;
        
        if (isNaN(lng) || isNaN(lat)) {
          console.warn('Invalid coordinates for radius popup:', lng, lat);
          return;
        }

        const properties = e.features[0].properties;
        const districtName = properties.district_name;
        
        // Find the district data from the original data array
        const districtData = data.districts?.find(d => d.district_name === districtName);
        
        console.log('=== POPUP CLICK DEBUG ===');
        console.log('Clicked district:', districtName);
        console.log('Found district data:', districtData);
        console.log('Raw district data JSON:', JSON.stringify(districtData, null, 2));
        console.log('District data indicators:', districtData?.indicators);
        console.log('Indicators length:', districtData?.indicators?.length);
        if (districtData?.indicators && districtData.indicators.length > 0) {
          console.log('First indicator:', districtData.indicators[0]);
        }
        console.log('Available districts:', data.districts?.map(d => d.district_name));
        console.log('Raw data.districts sample:', JSON.stringify(data.districts?.[0], null, 2));

        // Calculate responsive sizing
        const screenWidth = window.innerWidth;
        const screenHeight = window.innerHeight;
        const popupMaxWidth = Math.min(400, screenWidth * 0.35);
        const popupMaxHeight = Math.min(500, screenHeight * 0.7);

        let popupContent = `
          <div style="
            font-family: 'Segoe UI', -apple-system, BlinkMacSystemFont, sans-serif; 
            width: ${popupMaxWidth}px; 
            max-height: ${popupMaxHeight}px; 
            background: white;
            border-radius: 12px;
            box-shadow: 0 10px 30px rgba(0,0,0,0.2);
            overflow: hidden; 
            display: flex; 
            flex-direction: column;
            position: relative;
            border: 1px solid rgba(0,0,0,0.1);
          ">
            <!-- Header Section -->
            <div style="
              background: linear-gradient(135deg, #4f46e5 0%, #7c3aed 100%); 
              color: white; 
              padding: 16px 20px; 
              position: relative;
            ">
              <h3 style="margin: 0; font-size: 18px; font-weight: 700; line-height: 1.3; color: white;">${properties.district_name}</h3>
              <p style="margin: 6px 0 0 0; opacity: 0.9; font-size: 14px; font-weight: 500;">📍 ${properties.state_name}</p>
              
              <!-- Distance Badge -->
              <div style="
                display: inline-flex; 
                align-items: center; 
                background: rgba(255,255,255,0.2); 
                backdrop-filter: blur(10px);
                padding: 6px 12px; 
                border-radius: 20px; 
                margin-top: 12px;
                border: 1px solid rgba(255,255,255,0.3);
              ">
                <span style="margin-right: 6px; font-size: 14px;">🎯</span>
                <span style="font-weight: 600; font-size: 14px;">${properties.distance_km.toFixed(1)} km from center</span>
              </div>
            </div>
            
            <!-- Content Section -->
            <div style="padding: 16px 20px; flex: 1; overflow: hidden; display: flex; flex-direction: column;">
        `;

        // More robust indicator checking
        const hasIndicators = districtData && 
                             districtData.indicators && 
                             Array.isArray(districtData.indicators) && 
                             districtData.indicators.length > 0;
        
        console.log('Has indicators check:', hasIndicators);
        console.log('District data type:', typeof districtData);
        console.log('Indicators type:', typeof districtData?.indicators);
        console.log('Is indicators array?', Array.isArray(districtData?.indicators));
        
        if (hasIndicators) {
          // Show indicator values in a scrollable container
          const indicatorHeight = Math.min(350, popupMaxHeight - 180);
          popupContent += `
              <div style="margin-bottom: 16px;">
                <h4 style="margin: 0 0 12px 0; color: #374151; font-size: 16px; font-weight: 700; display: flex; align-items: center;">
                  <span style="margin-right: 8px; font-size: 18px;">📊</span>
                  Health Indicators (${districtData.indicators.length})
                </h4>
              </div>
              
              <div style="
                overflow-y: auto; 
                max-height: ${indicatorHeight}px; 
                padding-right: 8px; 
                flex: 1;
                scrollbar-width: thin; 
                scrollbar-color: #cbd5e0 #f7fafc;
              ">
          `;

          // Show all indicators in detailed format
          districtData.indicators.forEach((indicator, index) => {
            const changeColor = indicator.change_interpretation?.status === 'improving' ? '#28a745' :
                               indicator.change_interpretation?.status === 'declining' ? '#dc3545' : '#6c757d';
            const changeIcon = indicator.change_interpretation?.status === 'improving' ? '↗️' :
                              indicator.change_interpretation?.status === 'declining' ? '↘️' : '→';

            popupContent += `
              <div style="
                background: linear-gradient(145deg, #ffffff 0%, #f8fafc 100%); 
                border: 1px solid #e2e8f0; 
                border-radius: 12px; 
                padding: 16px; 
                margin-bottom: 12px;
                box-shadow: 0 2px 8px rgba(0,0,0,0.04);
                transition: all 0.2s ease;
              ">
                <!-- Indicator Name -->
                <div style="
                  font-weight: 700; 
                  color: #1e293b; 
                  margin-bottom: 12px; 
                  font-size: 14px; 
                  line-height: 1.4;
                  border-bottom: 2px solid #e2e8f0;
                  padding-bottom: 8px;
                ">${indicator.indicator_name}</div>
                
                <!-- Values Row -->
                <div style="display: flex; justify-content: space-between; align-items: center; gap: 12px; margin-bottom: 12px;">
                  <div style="flex: 1;">
                    <div style="display: flex; align-items: center; gap: 12px; font-size: 12px;">
                      <div style="
                        background: #f1f5f9; 
                        padding: 6px 10px; 
                        border-radius: 8px;
                        border-left: 3px solid #64748b;
                      ">
                        <span style="color: #64748b; font-weight: 500;">2016:</span> 
                        <strong style="color: #1e293b; margin-left: 4px;">${indicator.prevalence_2016?.toFixed(2) ?? 'N/A'}</strong>
                      </div>
                      <div style="color: #64748b; font-size: 14px;">→</div>
                      <div style="
                        background: #eff6ff; 
                        padding: 6px 10px; 
                        border-radius: 8px;
                        border-left: 3px solid #3b82f6;
                      ">
                        <span style="color: #3b82f6; font-weight: 500;">2021:</span> 
                        <strong style="color: #1e293b; margin-left: 4px;">${indicator.prevalence_2021?.toFixed(2) ?? 'N/A'}</strong>
                      </div>
                    </div>
                  </div>
                  
                  ${indicator.prevalence_change !== undefined ? `
                    <div style="
                      background: ${changeColor}; 
                      color: white; 
                      padding: 8px 12px; 
                      border-radius: 20px; 
                      font-size: 11px; 
                      font-weight: 600; 
                      white-space: nowrap;
                      box-shadow: 0 2px 4px rgba(0,0,0,0.1);
                      display: flex;
                      align-items: center;
                      gap: 4px;
                    ">
                      <span>${changeIcon}</span>
                      <span>${indicator.prevalence_change > 0 ? '+' : ''}${indicator.prevalence_change.toFixed(2)}</span>
                    </div>
                  ` : ''}
                </div>
                
                
                
                ${indicator.headcount_2021 !== undefined ? `
                  <div style="
                    background: #f8fafc; 
                    padding: 8px 12px; 
                    border-radius: 8px; 
                    font-size: 11px; 
                    color: #475569;
                    border: 1px solid #e2e8f0;
                  ">
                    <span style="color: #64748b; font-weight: 500;">👥 Headcount:</span> 
                    <strong style="color: #1e293b; margin-left: 4px;">${indicator.headcount_2021.toLocaleString()}</strong>
                  </div>
                ` : ''}
              </div>
            `;
          });

          popupContent += `
              </div>
            </div>
          </div>`;
        } else {
          popupContent += `
              <div style="
                color: #64748b; 
                text-align: center; 
                padding: 40px 20px;
                display: flex;
                flex-direction: column;
                align-items: center;
                gap: 12px;
              ">
                <div style="font-size: 48px; opacity: 0.5;">📊</div>
                <div style="font-size: 16px; font-weight: 600; color: #374151;">No Health Data Available</div>
                <div style="font-size: 14px; color: #64748b; max-width: 250px; line-height: 1.5;">
                  No health indicator data is currently available for this district in our database.
                </div>
              </div>
            </div>
          </div>`;
        }

        // Create popup with improved styling
        const popup = new mapboxgl.Popup({
          closeButton: true,
          closeOnClick: false,
          maxWidth: `${popupMaxWidth}px`,
          className: 'radius-analysis-popup',
          anchor: 'top-left',
          offset: [15, 15]
        })
          .setLngLat([lng, lat])
          .setHTML(popupContent)
          .addTo(map.current);
          
        // Add custom CSS for better close button visibility
        const popupElement = popup.getElement();
        if (popupElement) {
          const closeButton = popupElement.querySelector('.mapboxgl-popup-close-button');
          if (closeButton) {
            closeButton.style.cssText = `
              background: rgba(0,0,0,0.8) !important;
              color: white !important;
              border-radius: 50% !important;
              width: 24px !important;
              height: 24px !important;
              font-size: 16px !important;
              font-weight: bold !important;
              border: 2px solid white !important;
              box-shadow: 0 2px 8px rgba(0,0,0,0.3) !important;
              right: 8px !important;
              top: 8px !important;
              display: flex !important;
              align-items: center !important;
              justify-content: center !important;
              z-index: 1000 !important;
            `;
          }
        }
      });

      // Change cursor on hover
      map.current.on('mouseenter', 'radius-districts-fill', () => {
        map.current.getCanvas().style.cursor = 'pointer';
      });

      map.current.on('mouseleave', 'radius-districts-fill', () => {
        map.current.getCanvas().style.cursor = '';
      });
      
      // Final initialization for capture after all layers are added
      setTimeout(() => {
        console.log('All radius analysis layers loaded, final capture initialization...');
        initializeMapForCapture(map.current, 'radius-analysis-map');
        map.current.triggerRepaint();
        
        // Set up canvas preservation
        const canvas = map.current.getCanvas();
        if (canvas) {
          const gl = canvas.getContext('webgl') || canvas.getContext('webgl2');
          if (gl) {
            console.log('WebGL context found, preserveDrawingBuffer:', gl.getContextAttributes().preserveDrawingBuffer);
          }
        }
      }, 1000);

      // Add legend
      const legendEl = document.createElement('div');
      legendEl.className = 'mapboxgl-ctrl mapboxgl-ctrl-group';
      legendEl.style.cssText = `
        background: white;
        padding: 10px;
        border-radius: 4px;
        box-shadow: 0 0 10px rgba(0,0,0,0.1);
        font-family: Arial, sans-serif;
        font-size: 12px;
        line-height: 18px;
        max-width: 200px;
      `;

      let legendContent = `
        <div style="font-weight: bold; margin-bottom: 8px; color: #2c3e50;">Legend</div>
        <div style="display: flex; align-items: center; margin-bottom: 4px;">
          <div style="width: 15px; height: 15px; background: ${RADIUS_COLORS.center}; margin-right: 8px; border-radius: 50%;"></div>
          <span>Center Point</span>
        </div>
        <div style="display: flex; align-items: center; margin-bottom: 8px;">
          <div style="width: 15px; height: 3px; background: ${RADIUS_COLORS.radius_circle}; margin-right: 8px; border: 1px dashed ${RADIUS_COLORS.radius_circle};"></div>
          <span>${effectiveRadius} km Radius</span>
        </div>
        <div style="font-weight: bold; margin: 8px 0 4px 0; color: #2c3e50;">States:</div>
      `;

      // Add state colors to legend
      allStates.forEach(state => {
        legendContent += `
          <div style="display: flex; align-items: center; margin-bottom: 4px;">
            <div style="width: 15px; height: 15px; background: ${stateColors[state]}; margin-right: 8px;"></div>
            <span>${state}</span>
          </div>
        `;
      });

      legendEl.innerHTML = legendContent;

      // Add legend to map
      map.current.addControl({
        onAdd: () => legendEl,
        onRemove: () => {}
      }, 'top-right');
      
    }); // Close map load callback

    return () => {
      if (map.current) {
        map.current.remove();
        map.current = null;
      }
    };
  }, [data]);

  if (!data || !data.boundary_data) {
    return (
      <div className="flex items-center justify-center h-96 bg-gray-100 rounded-lg">
        <p className="text-gray-500">Map data not available</p>
      </div>
    );
  }

  return (
    <div 
      ref={mapContainer} 
      className={fullScreen ? "w-full h-full" : "w-full h-96 rounded-lg border border-gray-300"}
      style={fullScreen ? { 
        position: 'relative', 
        width: '100%', 
        height: '70vh', 
        minHeight: '600px',
        flex: 1
      } : { minHeight: '400px' }}
    />
  );
};

// Distance vs Change Chart (Updated to use backend chart data)
const DistanceChangeChart = ({ data }) => {
  const chartData = useMemo(() => {
    // First try to use backend-generated chart data
    if (data?.chart_data?.distance_vs_prevalence) {
      return data.chart_data.distance_vs_prevalence;
    }

    // Fallback to frontend generation if backend data not available
    if (!data || !data.districts) return null;

    const validDistricts = data.districts.filter(d =>
      d.indicators && d.indicators.some(i => i.prevalence_change !== undefined)
    );

    if (validDistricts.length === 0) return null;

    // Use STATE_COLORS directly for state coloring
    const states = [...new Set(validDistricts.map(d => d.state_name))];
    const stateColors = {};
    states.forEach((state, index) => {
      stateColors[state] = STATE_COLORS[index % STATE_COLORS.length];
    });

    const datasets = states.map(state => {
      const stateDistricts = validDistricts.filter(d => d.state_name === state);
      return {
        label: state,
        data: stateDistricts.map(d => {
          // Use first indicator's prevalence change as representative
          const indicator = d.indicators.find(i => i.prevalence_change !== undefined);
          return {
            x: d.distance_km,
            y: indicator?.prevalence_change || 0,
            district: d.district_name
          };
        }),
        backgroundColor: stateColors[state],
        borderColor: stateColors[state],
        pointRadius: 6,
        pointHoverRadius: 8
      };
    });

    return {
      type: 'scatter',
      title: 'Distance vs Health Indicator Change',
      datasets: datasets,
      options: {
        scales: {
          x: {
            type: 'linear',
            position: 'bottom',
            title: {
              display: true,
              text: 'Distance from Center (km)',
              font: { size: 14, weight: 'bold' }
            }
          },
          y: {
            title: {
              display: true,
              text: 'Health Indicator Change',
              font: { size: 14, weight: 'bold' }
            },
            grid: {
              color: (context) => context.tick.value === 0 ? '#000' : '#e0e0e0'
            }
          }
        },
        plugins: {
          legend: {
            position: 'top',
            labels: {
              usePointStyle: true,
              pointStyle: 'circle'
            }
          },
          tooltip: {
            callbacks: {
              label: (context) => {
                const point = context.parsed;
                const district = context.raw.district;
                return `${district}: ${point.x} km, Change: ${point.y?.toFixed(2)}`;
              }
            }
          }
        }
      }
    };
  }, [data]);

  if (!chartData) {
    return (
      <div className="flex items-center justify-center h-64 bg-gray-100 rounded-lg">
        <p className="text-gray-500">No health indicator change data available</p>
      </div>
    );
  }

  return <EnhancedChart chartConfig={chartData} />;
};

// Enhanced chart component that can render multiple chart types
const EnhancedChart = ({ chartConfig, title = null }) => {
  if (!chartConfig) return null;

  const { type, datasets, labels, options: chartOptions } = chartConfig;
  
  // Select appropriate Chart.js component
  let ChartComponent;
  switch (type) {
    case 'line':
      ChartComponent = Line;
      break;
    case 'scatter':
      ChartComponent = Scatter;
      break;
    case 'pie':
      ChartComponent = Pie;
      break;
    case 'bar':
    default:
      ChartComponent = Bar;
      break;
  }

  const defaultOptions = {
    responsive: true,
    plugins: {
      legend: {
        position: 'top'
      },
      title: {
        display: true,
        text: title || chartConfig.title || 'Chart',
        font: { size: 16, weight: 'bold' }
      }
    },
    ...chartOptions
  };

  const chartData = {
    labels,
    datasets
  };

  return (
    <div className="bg-white p-4 rounded-lg shadow">
      <ChartComponent data={chartData} options={defaultOptions} />
    </div>
  );
};

// Health Indicator Comparison Chart (Updated to use backend chart data)
const HealthIndicatorChart = ({ data }) => {
  const chartData = useMemo(() => {
    // First try to use backend-generated chart data
    if (data?.chart_data?.indicator_comparison) {
      return data.chart_data.indicator_comparison;
    }

    // Fallback to frontend generation if backend data not available
    if (!data || !data.districts) return null;

    const validDistricts = data.districts.filter(d =>
      d.indicators && d.indicators.some(i => i.prevalence_change !== undefined)
    ).slice(0, 10); // Top 10 districts

    if (validDistricts.length === 0) return null;

    // Sort by distance for better visualization
    validDistricts.sort((a, b) => a.distance_km - b.distance_km);

    return {
      type: 'bar',
      title: 'Health Indicator Change Trends',
      labels: validDistricts.map(d => `${d.district_name.substring(0, 15)}${d.district_name.length > 15 ? '...' : ''}`),
      datasets: [
        {
          label: 'Health Indicator Change (2016-2021)',
          data: validDistricts.map(d => {
            // Use first indicator's prevalence change as representative
            const indicator = d.indicators.find(i => i.prevalence_change !== undefined);
            return indicator?.prevalence_change || 0;
          }),
          backgroundColor: validDistricts.map(d => {
            const indicator = d.indicators.find(i => i.prevalence_change !== undefined);
            const change = indicator?.prevalence_change || 0;
            return change > 0 ? RADIUS_COLORS.decline : RADIUS_COLORS.improvement;
          }),
          borderColor: validDistricts.map(d => {
            const indicator = d.indicators.find(i => i.prevalence_change !== undefined);
            const change = indicator?.prevalence_change || 0;
            return change > 0 ? RADIUS_COLORS.decline : RADIUS_COLORS.improvement;
          }),
          borderWidth: 1
        }
      ],
      options: {
        scales: {
          y: {
            title: {
              display: true,
              text: 'Health Indicator Change',
              font: { size: 14, weight: 'bold' }
            },
            grid: {
              color: (context) => context.tick.value === 0 ? '#000' : '#e0e0e0'
            }
          },
          x: {
            title: {
              display: true,
              text: 'Districts (sorted by distance)',
              font: { size: 14, weight: 'bold' }
            }
          }
        },
        plugins: {
          tooltip: {
            callbacks: {
              label: (context) => {
                const district = data.districts.find(d =>
                  d.district_name.startsWith(context.label.replace('...', ''))
                );
                const change = context.formattedValue;
                return `${district?.district_name}: ${change > 0 ? '+' : ''}${change} (${district?.distance_km?.toFixed(1)} km)`;
              }
            }
          }
        }
      }
    };
  }, [data]);

  if (!chartData) {
    return (
      <div className="flex items-center justify-center h-64 bg-gray-100 rounded-lg">
        <p className="text-gray-500">No health indicator data available</p>
      </div>
    );
  }

  return <EnhancedChart chartConfig={chartData} />;
};

// Main RadiusAnalysis Component
const RadiusAnalysis = ({ radiusData, mapOnly = false, chartOnly = false }) => {
  const [activeTab, setActiveTab] = useState('overview');

  console.log('RadiusAnalysis component received data:', radiusData);

  if (!radiusData || !radiusData.districts) {
    console.log('RadiusAnalysis: No data or districts available');
    return (
      <div className="flex items-center justify-center h-64 bg-gray-100 rounded-lg">
        <div className="text-center">
          <p className="text-gray-500 text-lg">No radius analysis data available</p>
          <p className="text-gray-400 text-sm mt-2">Please try a different search query</p>
        </div>
      </div>
    );
  }

  if (mapOnly) {
    return (
      <div className="w-full h-full">
        <RadiusAnalysisMap data={radiusData} fullScreen={true} />
      </div>
    );
  }

  if (chartOnly) {
    return (
      <div className="space-y-6">
        <DistanceChangeChart data={radiusData} />
        <HealthIndicatorChart data={radiusData} />
      </div>
    );
  }

  const generateSummaryStats = () => {
    const { districts, radius_km, center_point, center_type } = radiusData;
    
    const totalDistricts = districts.length;
    const statesCount = new Set(districts.map(d => d.state_name)).size;
    
    const avgDistance = districts.reduce((sum, d) => sum + (d.distance_km || 0), 0) / totalDistricts;
    
    // Calculate average prevalence for 2021
    const districtsWithIndicators = districts.filter(d => d.indicators && d.indicators.length > 0);
    let avgPrevalence2021 = 0;
    if (districtsWithIndicators.length > 0) {
      const totalPrevalence = districtsWithIndicators.reduce((sum, d) => {
        const avgDistrictPrevalence = d.indicators.reduce((indSum, ind) => 
          indSum + (ind.prevalence_2021 || 0), 0) / d.indicators.length;
        return sum + avgDistrictPrevalence;
      }, 0);
      avgPrevalence2021 = totalPrevalence / districtsWithIndicators.length;
    }
    
    // Calculate average change
    const districtsWithChange = districts.filter(d => 
      d.indicators && d.indicators.some(i => i.prevalence_change !== undefined)
    );
    let avgChange = 0;
    if (districtsWithChange.length > 0) {
      const totalChange = districtsWithChange.reduce((sum, d) => {
        const changes = d.indicators.map(i => i.prevalence_change).filter(c => c !== undefined);
        const avgDistrictChange = changes.reduce((a, b) => a + b, 0) / changes.length;
        return sum + avgDistrictChange;
      }, 0);
      avgChange = totalChange / districtsWithChange.length;
    }

    return {
      totalDistricts,
      statesCount,
      avgDistance: avgDistance.toFixed(1),
      avgPrevalence2021: avgPrevalence2021.toFixed(2),
      avgChange: avgChange.toFixed(2),
      radius_km,
      center_point,
      center_type
    };
  };

  const stats = generateSummaryStats();

  // If overview tab is active, render just the map with minimal UI
  if (activeTab === 'overview') {
    return (
      <div className="relative w-full h-screen overflow-hidden">
        {/* Floating tab navigation for overview */}
        <div className="absolute top-4 left-4 z-10 bg-white rounded-lg shadow-lg">
          <nav className="flex">
            {[
              { id: 'overview', label: 'Map Overview' },
              { id: 'analysis', label: 'Health Analysis' },
              { id: 'trends', label: 'Indicator Trends' }
            ].map((tab) => (
              <button
                key={tab.id}
                onClick={() => setActiveTab(tab.id)}
                className={`py-2 px-4 font-medium text-sm first:rounded-l-lg last:rounded-r-lg ${
                  activeTab === tab.id
                    ? 'bg-blue-500 text-white'
                    : 'bg-white text-gray-700 hover:bg-gray-50'
                }`}
              >
                {tab.label}
              </button>
            ))}
          </nav>
        </div>
        
        {/* Floating stats summary */}
        <div className="absolute top-4 right-4 z-10 bg-white rounded-lg shadow-lg p-4 max-w-sm">
          <h3 className="text-lg font-bold text-gray-800 mb-2">
            {stats.radius_km} km Radius
          </h3>
          <div className="grid grid-cols-2 gap-2 text-xs">
            <div className="text-center">
              <div className="text-lg font-bold text-blue-600">{stats.totalDistricts}</div>
              <div className="text-gray-600">Districts</div>
            </div>
            <div className="text-center">
              <div className="text-lg font-bold text-green-600">{stats.statesCount}</div>
              <div className="text-gray-600">States</div>
            </div>
          </div>
          <p className="text-xs text-gray-600 mt-2">
            <strong>Center:</strong> {stats.center_point}
          </p>
        </div>
        
        <RadiusAnalysisMap data={radiusData} fullScreen={true} />
      </div>
    );
  }

  return (
    <div className="space-y-6">
      {/* Summary Header */}
      <div className="bg-white rounded-lg shadow p-6">
        <h2 className="text-2xl font-bold text-gray-800 mb-4">
          Health Districts Within {stats.radius_km} km Radius Analysis
        </h2>
        <p className="text-gray-600 mb-4">
          <strong>Center:</strong> {stats.center_point} ({stats.center_type})
        </p>
        
        <div className="grid grid-cols-2 md:grid-cols-3 lg:grid-cols-5 gap-4">
          <div className="bg-blue-50 p-4 rounded-lg text-center">
            <div className="text-2xl font-bold text-blue-600">{stats.totalDistricts}</div>
            <div className="text-sm text-gray-600">Total Districts</div>
          </div>
          <div className="bg-green-50 p-4 rounded-lg text-center">
            <div className="text-2xl font-bold text-green-600">{stats.statesCount}</div>
            <div className="text-sm text-gray-600">States Covered</div>
          </div>
          <div className="bg-purple-50 p-4 rounded-lg text-center">
            <div className="text-2xl font-bold text-purple-600">{stats.avgDistance}</div>
            <div className="text-sm text-gray-600">Avg Distance (km)</div>
          </div>
          <div className="bg-orange-50 p-4 rounded-lg text-center">
            <div className="text-2xl font-bold text-orange-600">{stats.avgPrevalence2021}</div>
            <div className="text-sm text-gray-600">Avg Prevalence</div>
          </div>
          <div className="bg-red-50 p-4 rounded-lg text-center">
            <div className="text-2xl font-bold text-red-600">
              {stats.avgChange > 0 ? '+' : ''}{stats.avgChange}
            </div>
            <div className="text-sm text-gray-600">Avg Change</div>
          </div>
        </div>
      </div>

      {/* Navigation Tabs */}
      <div className="bg-white rounded-lg shadow">
        <div className="border-b border-gray-200">
          <nav className="-mb-px flex space-x-8 px-6">
            {[
              { id: 'overview', label: 'Map Overview' },
              { id: 'analysis', label: 'Health Analysis' },
              { id: 'trends', label: 'Indicator Trends' }
            ].map((tab) => (
              <button
                key={tab.id}
                onClick={() => setActiveTab(tab.id)}
                className={`py-4 px-1 border-b-2 font-medium text-sm ${
                  activeTab === tab.id
                    ? 'border-blue-500 text-blue-600'
                    : 'border-transparent text-gray-500 hover:text-gray-700 hover:border-gray-300'
                }`}
              >
                {tab.label}
              </button>
            ))}
          </nav>
        </div>

        <div className="p-6">
          {activeTab === 'analysis' && (
            <div className="space-y-6">
              <HealthIndicatorChart data={radiusData} />
              <DistanceChangeChart data={radiusData} />
              {radiusData?.chart_data?.state_wise_distribution && (
                <EnhancedChart 
                  chartConfig={radiusData.chart_data.state_wise_distribution} 
                  title="Districts Distribution by State"
                />
              )}
              {radiusData?.chart_data?.summary_stats && (
                <EnhancedChart 
                  chartConfig={radiusData.chart_data.summary_stats} 
                  title="Health Indicators Summary"
                />
              )}
            </div>
          )}

          {activeTab === 'trends' && (
            <div className="space-y-6">
              {radiusData?.chart_data?.trend_analysis && (
                <EnhancedChart 
                  chartConfig={radiusData.chart_data.trend_analysis} 
                  title="Health Indicator Trends by Distance"
                />
              )}
              <DistanceChangeChart data={radiusData} />
              <div className="bg-white p-4 rounded-lg border border-gray-200">
                <h3 className="text-lg font-semibold text-gray-800 mb-3">Key Insights</h3>
                <div className="space-y-2 text-sm text-gray-600">
                  <p>Analysis shows health indicator patterns across {stats.totalDistricts} districts within {stats.radius_km} km of {stats.center_point}.</p>
                  <p>Average distance from center: {stats.avgDistance} km</p>
                  <p>Average health indicator change: {stats.avgChange > 0 ? '+' : ''}{stats.avgChange}</p>
                  {radiusData?.chart_data && (
                    <p>Enhanced charts generated from backend analysis include: {Object.keys(radiusData.chart_data).filter(key => radiusData.chart_data[key]).join(', ')}</p>
                  )}
                </div>
              </div>
            </div>
          )}
        </div>
      </div>
    </div>
  );
};

export default RadiusAnalysis;