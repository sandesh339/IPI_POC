import React, { useEffect, useRef, useState } from 'react';
import mapboxgl from 'mapbox-gl';
import Chart from 'chart.js/auto';
import 'mapbox-gl/dist/mapbox-gl.css';

// Set your Mapbox access token
mapboxgl.accessToken = process.env.REACT_APP_MAPBOX_TOKEN;

const DistrictClassificationChange = ({ data, mapOnly = false, chartOnly = false }) => {
  const mapContainer = useRef(null);
  const map = useRef(null);
  const chartRef = useRef(null);
  const chartInstance = useRef(null);
  const [isMapLoaded, setIsMapLoaded] = useState(false);

  // Debug logging (can be removed after confirming fix)
  console.log('✅ DistrictClassificationChange - Component loaded successfully with', data?.classified_districts?.length, 'districts');

  useEffect(() => {
    if (!map.current && mapContainer.current && (!chartOnly)) {
      map.current = new mapboxgl.Map({
        container: mapContainer.current,
        style: 'mapbox://styles/mapbox/light-v11',
        center: [78.9629, 20.5937],
        zoom: mapOnly ? 4.5 : 4.5
      });

      map.current.on('load', () => {
        setIsMapLoaded(true);
      });
    }

    return () => {
      if (map.current) {
        map.current.remove();
        map.current = null;
      }
    };
  }, [mapOnly, chartOnly]);

  useEffect(() => {
    if (isMapLoaded && data && data.classified_districts && data.boundary_data) {
      renderClassificationMap();
    }
  }, [isMapLoaded, data]);

  useEffect(() => {
    if (data && data.chart_data && (!mapOnly)) {
      renderChart();
    }
    return () => {
      if (chartInstance.current) {
        chartInstance.current.destroy();
        chartInstance.current = null;
      }
    };
  }, [data, mapOnly]);

  const addMapLegend = () => {
    if (!map.current || !data.classification_legend) return;

    // Remove existing legend if it exists
    const existingLegend = document.querySelector('.mapbox-change-legend-control');
    if (existingLegend) {
      existingLegend.remove();
    }

    // Create legend element
    const legendEl = document.createElement('div');
    legendEl.className = 'mapboxgl-ctrl mapboxgl-ctrl-group mapbox-change-legend-control';
    legendEl.style.cssText = `
      background: white;
      padding: 12px;
      border-radius: 6px;
      box-shadow: 0 2px 10px rgba(0,0,0,0.1);
      font-family: -apple-system, BlinkMacSystemFont, 'Segoe UI', Roboto, sans-serif;
      font-size: 12px;
      line-height: 1.4;
      max-width: 220px;
      min-width: 180px;
    `;

    // Build legend content
    let legendContent = `
      <div style="font-weight: 600; margin-bottom: 8px; color: #1f2937; font-size: 13px;">
        Change Classification
      </div>
    `;

    // Add each classification class
    data.classification_legend.forEach(item => {
      legendContent += `
        <div style="display: flex; align-items: center; margin-bottom: 6px; padding: 2px 0;">
          <div style="
            width: 16px; 
            height: 16px; 
            background: ${item.color}; 
            margin-right: 8px; 
            border-radius: 3px;
            border: 1px solid #e5e7eb;
            flex-shrink: 0;
          "></div>
          <div style="flex: 1; min-width: 0;">
            <div style="font-weight: 500; color: #374151; font-size: 11px;">
              ${item.class_name}
            </div>
            <div style="color: #6b7280; font-size: 10px; line-height: 1.2;">
              ${item.range_text}
            </div>
          </div>
        </div>
      `;
    });

    // Add indicator direction info
    if (data.indicator_info) {
      const direction = data.indicator_info.change_higher_is_better ? 'Positive change is better' : 'Negative change is better';
      legendContent += `
        <div style="
          margin-top: 8px; 
          padding-top: 6px; 
          border-top: 1px solid #e5e7eb; 
          font-size: 10px; 
          color: #6b7280;
          text-align: center;
        ">
          ${direction}
        </div>
      `;
    }

    legendEl.innerHTML = legendContent;

    // Add legend to map as a control
    map.current.addControl({
      onAdd: () => legendEl,
      onRemove: () => {
        if (legendEl.parentNode) {
          legendEl.parentNode.removeChild(legendEl);
        }
      }
    }, 'bottom-right');
  };

  const renderClassificationMap = () => {
    if (!map.current || !data) return;

    // Create classification data lookup
    const classificationLookup = {};
    data.classified_districts.forEach(district => {
      classificationLookup[district.district_name] = {
        class_color: district.class_color,
        class_name: district.class_name,
        class_description: district.class_description,
        change_value: district.indicator_value, // Use indicator_value as it contains the actual change data
        state_name: district.state_name
      };
    });

    // Create GeoJSON for classified districts
    const classifiedGeoJSON = {
      type: 'FeatureCollection',
      features: data.boundary_data
        .filter(boundary => classificationLookup[boundary.district_name])
        .map(boundary => {
          const classification = classificationLookup[boundary.district_name];
          return {
            type: 'Feature',
            properties: {
              district_name: boundary.district_name,
              state_name: classification.state_name,
              class_color: classification.class_color,
              class_name: classification.class_name,
              class_description: classification.class_description,
              change_value: classification.change_value,
              indicator_name: data.indicator_info.indicator_name
            },
            geometry: boundary.geometry
          };
        })
    };

    // Remove existing layers and sources
    const layerId = 'classification-change-layer';
    const sourceId = 'classification-change-source';
    
    if (map.current.getLayer(layerId)) {
      map.current.removeLayer(layerId);
    }
    if (map.current.getSource(sourceId)) {
      map.current.removeSource(sourceId);
    }

    // Add source and layer
    map.current.addSource(sourceId, {
      type: 'geojson',
      data: classifiedGeoJSON
    });

    map.current.addLayer({
      id: layerId,
      type: 'fill',
      source: sourceId,
      paint: {
        'fill-color': [
          'case',
          ['has', 'class_color'],
          ['get', 'class_color'],
          '#cccccc'
        ],
        'fill-opacity': 0.8,
        'fill-outline-color': '#ffffff'
      }
    });

    // Add border layer
    const borderLayerId = 'classification-change-border';
    if (map.current.getLayer(borderLayerId)) {
      map.current.removeLayer(borderLayerId);
    }

    map.current.addLayer({
      id: borderLayerId,
      type: 'line',
      source: sourceId,
      paint: {
        'line-color': '#ffffff',
        'line-width': 1,
        'line-opacity': 0.8
      }
    });

    // Create popup
    const popup = new mapboxgl.Popup({
      closeButton: false,
      closeOnClick: false
    });

    // Mouse events
    map.current.on('mouseenter', layerId, (e) => {
      map.current.getCanvas().style.cursor = 'pointer';
      
      const feature = e.features[0];
      const props = feature.properties;
      
      const changeValue = parseFloat(props.change_value);
      const changeText = changeValue > 0 ? 
        `+${changeValue.toFixed(1)} pp` : 
        `${changeValue.toFixed(1)} pp`;
      
      const popupContent = `
        <div class="bg-white p-3 rounded-lg shadow-lg max-w-xs">
          <h3 class="font-bold text-lg text-gray-800 mb-2">${props.district_name}</h3>
          <p class="text-sm text-gray-600 mb-1"><strong>State:</strong> ${props.state_name}</p>
          <p class="text-sm text-gray-600 mb-1"><strong>${props.indicator_name} Change:</strong> ${changeText}</p>
          <div class="flex items-center mt-2">
            <div class="w-4 h-4 rounded mr-2" style="background-color: ${props.class_color}"></div>
            <span class="text-sm font-medium text-gray-800">${props.class_name}</span>
          </div>
          <p class="text-xs text-gray-500 mt-1">${props.class_description}</p>
        </div>
      `;

      popup.setLngLat(e.lngLat).setHTML(popupContent).addTo(map.current);
    });

    map.current.on('mouseleave', layerId, () => {
      map.current.getCanvas().style.cursor = '';
      popup.remove();
    });

    // Add map legend
    addMapLegend();

    // Fit bounds to show all classified districts
    if (classifiedGeoJSON.features.length > 0) {
      const bounds = new mapboxgl.LngLatBounds();
      classifiedGeoJSON.features.forEach(feature => {
        if (feature.geometry.type === 'Polygon') {
          feature.geometry.coordinates[0].forEach(coord => bounds.extend(coord));
        } else if (feature.geometry.type === 'MultiPolygon') {
          feature.geometry.coordinates.forEach(polygon => {
            polygon[0].forEach(coord => bounds.extend(coord));
          });
        }
      });
      map.current.fitBounds(bounds, { padding: 50 });
    }
  };

  const renderChart = () => {
    if (!chartRef.current || !data.chart_data) return;

    // Destroy existing chart
    if (chartInstance.current) {
      chartInstance.current.destroy();
    }

    const ctx = chartRef.current.getContext('2d');
    
    chartInstance.current = new Chart(ctx, {
      type: 'bar',
      data: data.chart_data.data,
      options: {
        responsive: true,
        maintainAspectRatio: false, // Allow height control by container
        layout: {
          padding: {
            top: 5,
            bottom: 5,
            left: 10,
            right: 10
          }
        },
        plugins: {
          title: {
            display: false // Hide title since we have our own styled title
          },
          legend: {
            display: false
          },
          tooltip: {
            backgroundColor: 'rgba(0, 0, 0, 0.8)',
            titleFont: {
              size: 14,
              weight: 'bold'
            },
            bodyFont: {
              size: 13
            },
            padding: 12,
            cornerRadius: 8,
            callbacks: {
              title: function(context) {
                return `${context[0].label} Change`;
              },
              label: function(context) {
                const percentage = ((context.parsed.y / data.statistics.total_districts) * 100).toFixed(1);
                return `Districts: ${context.parsed.y} (${percentage}%)`;
              }
            }
          }
        },
        scales: {
          y: {
            beginAtZero: true,
            max: Math.max(...Object.values(data.statistics.class_counts)) * 1.02, // Set max to only 102% of highest value
            title: {
              display: true,
              text: 'Districts',
              font: {
                size: 11,
                weight: 'bold'
              }
            },
            ticks: {
              stepSize: Math.ceil(Math.max(...Object.values(data.statistics.class_counts)) / 3), // Only 3 steps total
              font: {
                size: 10
              }
            },
            grid: {
              color: 'rgba(0, 0, 0, 0.08)'
            }
          },
          x: {
            title: {
              display: true,
              text: 'Change Performance Class',
              font: {
                size: 11,
                weight: 'bold'
              }
            },
            ticks: {
              font: {
                size: 10,
                weight: 'bold'
              }
            },
            grid: {
              display: false
            }
          }
        }
      }
    });

    // Force chart resize after creation
    setTimeout(() => {
      if (chartInstance.current) {
        chartInstance.current.resize();
      }
    }, 100);
  };

  const renderLegend = () => {
    if (!data.classification_legend) return null;

    return (
      <div className="bg-gradient-to-br from-green-50 to-emerald-50 rounded-xl p-6 shadow-xl border border-green-100">
        <h3 className="font-bold text-xl mb-5 text-gray-800 flex items-center">
          <span className="mr-2">📈</span>
          Change Classes
        </h3>
        <div className="space-y-4">
          {data.classification_legend.map((item, index) => (
            <div key={index} className="flex items-center p-3 bg-white rounded-lg shadow-sm hover:shadow-md transition-shadow duration-200">
              <div 
                className="w-8 h-8 rounded-lg mr-4 border-2 border-white shadow-md"
                style={{ backgroundColor: item.color }}
              ></div>
              <div className="flex-1">
                <div className="font-semibold text-base text-gray-800">{item.class_name}</div>
                <div className="text-sm text-green-600 font-medium">{item.range_text}</div>
                <div className="text-sm text-gray-600">{item.description}</div>
              </div>
            </div>
          ))}
        </div>
        
        {data.indicator_info && (
          <div className="mt-6 pt-4 border-t border-green-200">
            <div className="bg-white rounded-lg p-3 shadow-sm">
              <p className="text-sm text-gray-700 mb-2">
                <strong className="text-green-600">Change Direction:</strong>{' '}
                <span className="font-medium">
              {data.indicator_info.change_higher_is_better ? 'Positive change is better' : 'Negative change is better'}
                </span>
              </p>
              <p className="text-sm text-gray-600">
                <strong className="text-green-600">Classification method:</strong> Jenks Natural Breaks
              </p>
            </div>
          </div>
        )}
      </div>
    );
  };

  const renderStatistics = () => {
    if (!data.statistics) return null;

    const stats = data.statistics;
    const total = stats.total_districts;

    return (
      <div className="bg-gradient-to-br from-orange-50 to-red-50 rounded-xl p-6 shadow-xl border border-orange-100">
        <h3 className="font-bold text-xl mb-5 text-gray-800 flex items-center">
          <span className="mr-2">📊</span>
          Change Statistics
        </h3>
        
        {/* Key Statistics Cards */}
        <div className="grid grid-cols-2 gap-4 mb-6">
          <div className="bg-white rounded-lg p-4 shadow-sm border border-orange-100">
            <p className="text-sm text-orange-600 font-medium">Total Districts</p>
            <p className="font-bold text-2xl text-gray-800">{total}</p>
          </div>
          <div className="bg-white rounded-lg p-4 shadow-sm border border-orange-100">
            <p className="text-sm text-orange-600 font-medium">Mean Change</p>
            <p className="font-bold text-2xl text-gray-800">{stats.mean_value.toFixed(1)} pp</p>
          </div>
          <div className="bg-white rounded-lg p-4 shadow-sm border border-orange-100">
            <p className="text-sm text-orange-600 font-medium">Change Range</p>
            <p className="font-bold text-lg text-gray-800">{stats.min_value.toFixed(1)} to {stats.max_value.toFixed(1)} pp</p>
          </div>
          <div className="bg-white rounded-lg p-4 shadow-sm border border-orange-100">
            <p className="text-sm text-orange-600 font-medium">Std. Deviation</p>
            <p className="font-bold text-2xl text-gray-800">{stats.std_value.toFixed(1)} pp</p>
          </div>
        </div>
        
        {/* Districts per Class */}
        <div className="bg-white rounded-lg p-4 shadow-sm border border-orange-100">
          <h4 className="font-semibold text-base text-gray-800 mb-3 flex items-center">
            <span className="mr-2">🏛️</span>
            Districts per Change Class
          </h4>
          <div className="space-y-3">
          {Object.entries(stats.class_counts).map(([classNum, count]) => {
            const percentage = ((count / total) * 100).toFixed(1);
            const legend = data.classification_legend.find(l => l.class_number === parseInt(classNum));
            return (
                <div key={classNum} className="flex justify-between items-center p-2 bg-gray-50 rounded-lg hover:bg-gray-100 transition-colors duration-200">
                <span className="flex items-center">
                  <div 
                      className="w-4 h-4 rounded mr-3 border border-white shadow-sm"
                    style={{ backgroundColor: legend?.color || '#cccccc' }}
                  ></div>
                    <span className="font-medium text-gray-800">{legend?.class_name || `Class ${classNum}`}</span>
                </span>
                  <span className="font-semibold text-orange-600">{count} ({percentage}%)</span>
                </div>
              );
            })}
          </div>
        </div>
      </div>
    );
  };

  if (!data) {
    return (
      <div className="flex items-center justify-center h-64">
        <div className="text-gray-500">No change classification data available</div>
      </div>
    );
  }

  // Map-only view
  if (mapOnly) {
    return (
      <div className="w-full h-full">
        <div className="bg-white rounded-lg shadow-lg p-4 h-full">
          <h3 className="font-bold text-lg mb-4">
            District Change Classification Map: {data.indicator_info?.indicator_name}
          </h3>
          <div 
            ref={mapContainer} 
            className="w-full rounded-lg"
            style={{ minHeight: '500px', height: 'calc(100vh - 200px)' }}
          />
        </div>
      </div>
    );
  }

  // Chart-only view
  if (chartOnly) {
    return (
      <div className="w-full flex justify-center">
        <div className="w-full max-w-5xl">
          {/* Chart Section - Only Bar Diagram */}
          <div className="bg-gradient-to-br from-slate-50 to-blue-50 rounded-2xl shadow-2xl p-8 border border-blue-100">
            <h3 className="font-bold text-2xl mb-6 text-gray-800 text-center bg-gradient-to-r from-blue-600 to-purple-600 bg-clip-text text-transparent">
              District Count by Change Performance Class: {data.indicator_info?.indicator_name}
            </h3>
            <div className="bg-white rounded-xl p-4 shadow-lg" style={{ height: '450px' }}>
              <canvas ref={chartRef}></canvas>
            </div>
          </div>
        </div>
      </div>
    );
  }

  // Full view (default)
  return (
    <div className="w-full space-y-6">
      {/* Title and Indicator Info */}
      {data.indicator_info && (
        <div className="bg-gradient-to-r from-green-600 to-emerald-700 text-white p-6 rounded-lg">
          <h2 className="text-2xl font-bold mb-2">
            District Change Classification: {data.indicator_info.indicator_name}
          </h2>
          <p className="text-green-100">
            {data.statistics?.total_districts} districts classified into 4 change performance classes using Jenks natural breaks
          </p>
          <p className="text-green-100 text-sm mt-1">
            Based on prevalence change from 2016 to 2021
          </p>
          {data.state_filter && (
            <p className="text-green-100 text-sm mt-1">
              Filtered by: {data.state_filter.join(', ')}
            </p>
          )}
        </div>
      )}

      {/* Main Content Grid */}
      <div className="grid grid-cols-1 lg:grid-cols-3 gap-6">
        {/* Map Column */}
        <div className="lg:col-span-2">
          <div className="bg-white rounded-lg shadow-lg p-4">
            <h3 className="font-bold text-lg mb-4">Change Classification Map</h3>
            <div 
              ref={mapContainer} 
              className="w-full h-96 rounded-lg"
              style={{ minHeight: '400px' }}
            />
          </div>
        </div>

        {/* Sidebar */}
        <div className="space-y-4">
          {renderLegend()}
          {renderStatistics()}
        </div>
      </div>

      {/* Chart Row - Bar Diagram */}
      <div className="flex justify-center">
        <div className="w-full max-w-4xl">
          <div className="bg-gradient-to-br from-slate-50 to-blue-50 rounded-2xl shadow-2xl p-8 border border-blue-100">
            <h3 className="font-bold text-2xl mb-6 text-gray-800 text-center bg-gradient-to-r from-blue-600 to-purple-600 bg-clip-text text-transparent">
              District Count by Change Performance Class
            </h3>
            <div className="bg-white rounded-xl p-4 shadow-lg" style={{ height: '400px' }}>
              <canvas ref={chartRef}></canvas>
            </div>
          </div>
        </div>
      </div>
    </div>
  );
};

export default DistrictClassificationChange;
