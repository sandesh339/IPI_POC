import React, { useState, useMemo, useRef, useEffect } from 'react';
import { Bar } from 'react-chartjs-2';
import Map, { Source, Layer } from 'react-map-gl';
import 'mapbox-gl/dist/mapbox-gl.css';
import {
  Chart as ChartJS,
  CategoryScale,
  LinearScale,
  BarElement,
  Title,
  Tooltip,
  Legend,
} from 'chart.js';
import { registerMapInstance, initializeReactMapGLForCapture } from '../utils/saveUtils';

ChartJS.register(
  CategoryScale,
  LinearScale,
  BarElement,
  Title,
  Tooltip,
  Legend
);

const MAPBOX_TOKEN = process.env.REACT_APP_MAPBOX_TOKEN;

export default function MultiDistrictAnalysis({ data = {}, chartOnly = false, mapOnly = false }) {
  console.log('MultiDistrictAnalysis received data:', data);

  const [viewState, setViewState] = useState({
    longitude: 78.96,
    latitude: 20.59,
    zoom: 5
  });
  const mapRef = useRef(null);

  // Set up WebGL context preservation when map loads
  useEffect(() => {
    const initializeMap = async () => {
      if (mapRef.current) {
        try {
          await initializeReactMapGLForCapture(mapRef, 'multi-district-map');
          console.log('Multi-district map initialized for capture');
        } catch (error) {
          console.error('Error initializing multi-district map:', error);
        }
      }
    };
    
    initializeMap();
    const timer = setTimeout(initializeMap, 100);
    return () => clearTimeout(timer);
  }, []);

  // Extract the actual data from nested structure
  const actualData = useMemo(() => {
    // Handle flattened data structure from health_main.py for both individual and multi-district
    if (data?.map_type === "multi_district_comparison" && data?.chart_data) {
      return data;
    }
    
    if (data?.map_type === "individual_district" && data?.chart_data) {
      return data;
    }
    
    // Handle nested data structure
    if (data?.data && Array.isArray(data.data) && data.data.length > 0 && data.data[0]?.result) {
      const result = data.data[0].result;
      if (result.map_type === "multi_district_comparison" || result.map_type === "individual_district") {
        return result;
      }
    }
    
    // If data is the direct result
    if (data && typeof data === 'object' && (data.map_type === "multi_district_comparison" || data.map_type === "individual_district")) {
      return data;
    }
    
    return data;
  }, [data]);

  // Extract district information
  const districtInfo = useMemo(() => {
    if (!actualData) return null;
    
    // Handle individual district data
    if (actualData.map_type === "individual_district") {
      return {
        districts: [{
          district_name: actualData.district_name,
          state_name: actualData.state_name,
          indicators: actualData.data || []
        }],
        totalDistricts: 1,
        totalIndicators: actualData.total_indicators || 0,
        indicators: actualData.data ? actualData.data.map(d => d.indicator_name) : [],
        year: actualData.year || 2021,
        chartData: actualData.chart_data || null,
        analysis: actualData.analysis || ""
      };
    }
    
    // Handle multi-district data
    return {
      districts: actualData.districts || [],
      totalDistricts: actualData.total_districts || 0,
      totalIndicators: actualData.total_indicators || 0,
      indicators: actualData.indicators || [],
      year: actualData.year || 2021,
      chartData: actualData.chart_data || null,
      analysis: actualData.analysis || ""
    };
  }, [actualData]);

  // Process boundary data for map visualization
  const mapFeatures = useMemo(() => {
    if (!districtInfo || !actualData.boundary || actualData.boundary.length === 0) return [];

    const colors = ['#3498db', '#e74c3c', '#2ecc71', '#f39c12', '#9b59b6', '#1abc9c', '#e67e22', '#34495e'];
    
    return actualData.boundary.map((boundaryItem, index) => ({
      type: "Feature",
      geometry: boundaryItem.geometry || boundaryItem,
      properties: {
        district_name: boundaryItem.district_name,
        state_name: boundaryItem.state_name,
        fill_color: colors[index % colors.length],
        stroke_color: colors[index % colors.length]
      }
    }));
  }, [districtInfo, actualData]);

  // Set initial map view to show all districts
  useEffect(() => {
    if (mapFeatures.length > 0) {
      let minLng = Infinity, maxLng = -Infinity;
      let minLat = Infinity, maxLat = -Infinity;
      
      mapFeatures.forEach(feature => {
        const geometry = feature.geometry;
        
        const extractCoordinates = (coords) => {
          if (Array.isArray(coords[0])) {
            coords.forEach(coord => extractCoordinates(coord));
          } else {
            const [lng, lat] = coords;
            minLng = Math.min(minLng, lng);
            maxLng = Math.max(maxLng, lng);
            minLat = Math.min(minLat, lat);
            maxLat = Math.max(maxLat, lat);
          }
        };

        if (geometry && geometry.coordinates) {
          extractCoordinates(geometry.coordinates);
        }
      });
      
      if (minLng !== Infinity) {
        const centerLng = (minLng + maxLng) / 2;
        const centerLat = (minLat + maxLat) / 2;
        
        // Calculate appropriate zoom level
        const latDiff = maxLat - minLat;
        const lngDiff = maxLng - minLng;
        const maxDiff = Math.max(latDiff, lngDiff);
        
        let zoom = 6;
        if (maxDiff < 1) zoom = 9;
        else if (maxDiff < 2) zoom = 8;
        else if (maxDiff < 5) zoom = 7;
        else if (maxDiff < 10) zoom = 6;
        else zoom = 5;
        
        setViewState(prev => ({
          ...prev,
          longitude: centerLng,
          latitude: centerLat,
          zoom: zoom
        }));
      }
    }
  }, [mapFeatures]);

  // Chart options with proper Chart.js configuration
  const chartOptions = useMemo(() => ({
    responsive: true,
    maintainAspectRatio: false,
    plugins: {
      legend: {
        position: 'top',
        labels: {
          font: { size: 12 },
          color: '#2C3E50'
        }
      },
      title: {
        display: true,
        text: districtInfo?.chartData?.title || 'District Health Indicator Comparison',
        font: { size: 16, weight: 'bold' },
        color: '#2C3E50'
      },
      tooltip: {
        mode: 'index',
        intersect: false,
        callbacks: {
          label: function(context) {
            return `${context.dataset.label}: ${context.parsed.y.toFixed(2)}%`;
          }
        }
      }
    },
    scales: {
      y: {
        beginAtZero: true,
        title: {
          display: true,
          text: 'Prevalence (%)',
          color: '#2C3E50'
        },
        ticks: { color: '#2C3E50' },
        grid: { color: '#ECF0F1' }
      },
      x: {
        title: {
          display: true,
          text: 'Districts',
          color: '#2C3E50'
        },
        ticks: { 
          color: '#2C3E50',
          maxRotation: 45,
          minRotation: 0
        },
        grid: { color: '#ECF0F1' }
      }
    }
  }), [districtInfo]);

  if (!districtInfo || districtInfo.totalDistricts === 0) {
    return (
      <div style={{ 
        padding: '20px', 
        textAlign: 'center',
        color: '#7F8C8D'
      }}>
        <p>No multi-district data available for visualization.</p>
      </div>
    );
  }

  return (
    <div style={{ 
      padding: chartOnly || mapOnly ? '0' : '20px',
      fontFamily: 'system-ui, -apple-system, sans-serif'
    }}>
      {/* District Summary Header */}
      {!chartOnly && !mapOnly && (
        <div style={{
          background: 'linear-gradient(135deg, #ffffff 0%, #f8f9fa 100%)',
          padding: '24px',
          borderRadius: '12px',
          border: '2px solid #3498DB',
          marginBottom: '20px',
          boxShadow: '0 4px 12px rgba(52, 152, 219, 0.1)'
        }}>
          <h2 style={{
            color: '#2C3E50',
            margin: '0 0 12px 0',
            fontSize: '24px',
            fontWeight: 'bold'
          }}>
            📊 Multi-District Health Comparison
          </h2>
          <div style={{
            display: 'grid',
            gridTemplateColumns: 'repeat(auto-fit, minmax(150px, 1fr))',
            gap: '16px',
            marginTop: '16px'
          }}>
            <div style={{ textAlign: 'center', padding: '12px', background: '#E8F6F3', borderRadius: '8px' }}>
              <div style={{ fontSize: '24px', fontWeight: 'bold', color: '#138D75' }}>
                {districtInfo.totalDistricts}
              </div>
              <div style={{ fontSize: '12px', color: '#5D6D7E' }}>Districts</div>
            </div>
            <div style={{ textAlign: 'center', padding: '12px', background: '#EBF5FB', borderRadius: '8px' }}>
              <div style={{ fontSize: '24px', fontWeight: 'bold', color: '#2980B9' }}>
                {districtInfo.totalIndicators}
              </div>
              <div style={{ fontSize: '12px', color: '#5D6D7E' }}>Indicators</div>
            </div>
            <div style={{ textAlign: 'center', padding: '12px', background: '#FDF2E9', borderRadius: '8px' }}>
              <div style={{ fontSize: '24px', fontWeight: 'bold', color: '#D68910' }}>
                {districtInfo.year}
              </div>
              <div style={{ fontSize: '12px', color: '#5D6D7E' }}>Data Year</div>
            </div>
          </div>
        </div>
      )}

      {/* Chart Section */}
      {(chartOnly || !mapOnly) && districtInfo.chartData && (
        <div style={{
          background: '#ffffff',
          padding: '20px',
          borderRadius: '12px',
          border: '1px solid #E8F4FD',
          boxShadow: '0 2px 8px rgba(52, 152, 219, 0.1)',
          marginBottom: mapOnly ? '0' : '20px',
          height: chartOnly ? '500px' : '400px'
        }}>
          <Bar data={districtInfo.chartData} options={chartOptions} />
        </div>
      )}

      {/* Map Section */}
      {(mapOnly || !chartOnly) && mapFeatures.length > 0 && (
        <div style={{
          background: '#ffffff',
          padding: '20px',
          borderRadius: '12px',
          border: '1px solid #E8F4FD',
          boxShadow: '0 2px 8px rgba(52, 152, 219, 0.1)',
          height: mapOnly ? '500px' : '400px',
          marginBottom: '20px'
        }}>
          <h3 style={{ 
            color: '#2C3E50', 
            marginBottom: '16px',
            fontSize: '16px'
          }}>
            🗺️ District Locations Map
          </h3>
          <div style={{ height: mapOnly ? '440px' : '320px', borderRadius: '8px', overflow: 'hidden' }}>
            <Map
              ref={mapRef}
              {...viewState}
              onMove={evt => setViewState(evt.viewState)}
              style={{ width: '100%', height: '100%' }}
              mapStyle="mapbox://styles/mapbox/light-v10"
              mapboxAccessToken={MAPBOX_TOKEN}
            >
              <Source
                id="district-boundaries"
                type="geojson"
                data={{
                  type: "FeatureCollection",
                  features: mapFeatures
                }}
              >
                <Layer
                  id="district-fill"
                  type="fill"
                  paint={{
                    'fill-color': ['get', 'fill_color'],
                    'fill-opacity': 0.6
                  }}
                />
                <Layer
                  id="district-stroke"
                  type="line"
                  paint={{
                    'line-color': ['get', 'stroke_color'],
                    'line-width': 2
                  }}
                />
              </Source>
            </Map>
          </div>
          
          {/* Map Legend */}
          <div style={{
            marginTop: '12px',
            padding: '12px',
            background: '#F8F9FA',
            borderRadius: '8px',
            fontSize: '12px'
          }}>
            <strong>Districts: </strong>
            {districtInfo.districts.map((district, index) => (
              <span key={index} style={{ marginRight: '12px' }}>
                <span style={{ 
                  display: 'inline-block',
                  width: '12px',
                  height: '12px',
                  backgroundColor: ['#3498db', '#e74c3c', '#2ecc71', '#f39c12', '#9b59b6', '#1abc9c', '#e67e22', '#34495e'][index % 8],
                  marginRight: '4px',
                  borderRadius: '2px'
                }}></span>
                {district.district_name}
              </span>
            ))}
          </div>
        </div>
      )}

      {/* Districts Details Table */}
      {!chartOnly && !mapOnly && (
        <div style={{
          background: '#ffffff',
          padding: '20px',
          borderRadius: '12px',
          border: '1px solid #E8F4FD',
          boxShadow: '0 2px 8px rgba(52, 152, 219, 0.1)'
        }}>
          <h3 style={{ 
            color: '#2C3E50', 
            marginBottom: '16px',
            fontSize: '16px'
          }}>
            📋 District Details Summary
          </h3>
          
          <div style={{
            overflowX: 'auto',
            borderRadius: '8px',
            border: '1px solid #E8F4FD'
          }}>
            <table style={{
              width: '100%',
              borderCollapse: 'collapse',
              fontSize: '13px'
            }}>
              <thead>
                <tr style={{ background: '#F8F9FA' }}>
                  <th style={{ padding: '12px 8px', textAlign: 'left', color: '#2C3E50' }}>District</th>
                  <th style={{ padding: '12px 8px', textAlign: 'left', color: '#2C3E50' }}>State</th>
                  <th style={{ padding: '12px 8px', textAlign: 'center', color: '#2C3E50' }}>Indicators</th>
                  <th style={{ padding: '12px 8px', textAlign: 'center', color: '#2C3E50' }}>Avg Prevalence</th>
                </tr>
              </thead>
              <tbody>
                {districtInfo.districts.map((district, index) => {
                  const validIndicators = district.indicators.filter(ind => ind.prevalence_2021 !== null);
                  const avgPrevalence = validIndicators.length > 0 
                    ? validIndicators.reduce((sum, ind) => sum + ind.prevalence_2021, 0) / validIndicators.length
                    : 0;
                  
                  return (
                    <tr key={index} style={{
                      borderBottom: '1px solid #F8F9FA',
                      background: index % 2 === 0 ? '#ffffff' : '#FAFBFC'
                    }}>
                      <td style={{ 
                        padding: '12px 8px',
                        fontWeight: 'bold',
                        color: '#2C3E50'
                      }}>
                        {district.district_name}
                      </td>
                      <td style={{ 
                        padding: '12px 8px',
                        color: '#34495E'
                      }}>
                        {district.state_name}
                      </td>
                      <td style={{ 
                        padding: '12px 8px', 
                        textAlign: 'center',
                        color: '#2980B9',
                        fontWeight: 'bold'
                      }}>
                        {district.indicators.length}
                      </td>
                      <td style={{ 
                        padding: '12px 8px', 
                        textAlign: 'center',
                        color: '#E67E22',
                        fontWeight: 'bold'
                      }}>
                        {avgPrevalence.toFixed(2)}%
                      </td>
                    </tr>
                  );
                })}
              </tbody>
            </table>
          </div>
        </div>
      )}
    </div>
  );
}
