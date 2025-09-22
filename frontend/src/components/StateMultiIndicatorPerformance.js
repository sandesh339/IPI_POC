import React, { useState, useMemo, useRef, useEffect, useCallback } from 'react';
import { Map, Source, Layer, Popup } from 'react-map-gl';
import { Bar, Doughnut } from 'react-chartjs-2';
import {
  Chart as ChartJS,
  CategoryScale,
  LinearScale,
  BarElement,
  Title,
  Tooltip,
  Legend,
  ArcElement
} from 'chart.js';

// Register Chart.js components
ChartJS.register(CategoryScale, LinearScale, BarElement, Title, Tooltip, Legend, ArcElement);

// Mapbox access token
const mapboxAccessToken = process.env.REACT_APP_MAPBOX_TOKEN;

const StateMultiIndicatorPerformance = ({ data, mapOnly = false, chartOnly = false }) => {
  const [popupInfo, setPopupInfo] = useState(null);
  const [hoveredFeature, setHoveredFeature] = useState(null);
  const [currentChartPage, setCurrentChartPage] = useState(0);
  const [showStates, setShowStates] = useState(true);
  const [showDistricts, setShowDistricts] = useState(true);
  const mapRef = useRef();

  // Process the data for easier access
  const stateInfo = useMemo(() => {
    if (!data) {
      return {
        states: [],
        stateDistricts: {},
        totalStates: 0,
        totalDistricts: 0,
        totalIndicators: 0,
        performanceType: 'unknown',
        categoryName: null,
        chartData: [],
        analysis: ''
      };
    }

    // Enhanced data extraction
    let states = [];
    let stateDistricts = {};
    let chartData = [];
    
    // Path 1: Direct access
    if (data.states && Array.isArray(data.states)) {
      states = data.states;
      stateDistricts = data.state_districts || {};
      chartData = data.chart_data || [];
    }
    // Path 2: Nested in data array (from function_calls)
    else if (data.data && Array.isArray(data.data)) {
      for (const item of data.data) {
        if (item.result) {
          if (item.result.states && Array.isArray(item.result.states)) {
            states = item.result.states;
            stateDistricts = item.result.state_districts || {};
            chartData = item.result.chart_data || [];
            break;
          }
        }
      }
    }
    // Path 3: Function result structure
    else if (data.data && !Array.isArray(data.data) && data.data.states) {
      states = data.data.states;
      stateDistricts = data.data.state_districts || {};
      chartData = data.data.chart_data || [];
    }

    // Enhanced chart data extraction
    if (!chartData || !Array.isArray(chartData)) {
      chartData = data.chart_data || 
                 data.data?.chart_data || 
                 (data.data && Array.isArray(data.data) && data.data[0]?.result?.chart_data) || 
                 [];
    }

    // Convert chart data object to array if needed
    if (chartData && typeof chartData === 'object' && !Array.isArray(chartData)) {
      chartData = Object.values(chartData).filter(chart => chart && typeof chart === 'object');
    }

    const result = {
      states: states || [],
      stateDistricts: stateDistricts || {},
      totalStates: data.total_states || states.length || 0,
      totalDistricts: data.total_districts || 0,
      totalIndicators: data.total_indicators || 0,
      performanceType: data.performance_type || 'unknown',
      categoryName: data.category_name,
      chartData: Array.isArray(chartData) ? chartData : [],
      analysis: data.analysis || ''
    };

    return result;
  }, [data]);

  // Process boundary data for map visualization
  const boundaryData = useMemo(() => {
    let boundaryData = data?.boundary_data || data?.boundary || [];
    
    // Handle case where boundary data might be nested
    if (!boundaryData.length && data?.data && Array.isArray(data.data)) {
      for (const item of data.data) {
        if (item.result && (item.result.boundary_data || item.result.boundary)) {
          boundaryData = item.result.boundary_data || item.result.boundary;
          break;
        }
      }
    }
    
    if (!boundaryData || !boundaryData.length) {
      return { 
        stateFeatures: { type: 'FeatureCollection', features: [] },
        districtFeatures: { type: 'FeatureCollection', features: [] }
      };
    }

    try {
      // Separate state and district boundaries
      const stateFeatures = [];
      const districtFeatures = [];

      boundaryData.forEach((boundary) => {
        try {
          const geometry = typeof boundary.geometry === 'string' 
            ? JSON.parse(boundary.geometry) 
            : boundary.geometry;

          if (!geometry || !geometry.coordinates || !Array.isArray(geometry.coordinates)) {
            return;
          }

          // Check if this is a state or district boundary
          if (boundary.type === 'state') {
            // State boundary
            const stateName = boundary.name || boundary.state_name;
            const stateData = stateInfo.states.find(s => s.state_name === stateName);

            if (stateData) {
              stateFeatures.push({
                type: 'Feature',
                properties: {
                  type: 'state',
                  name: stateName,
                  state_name: stateName,
                  performance_index_2021: stateData.performance_index_2021,
                  performance_index_2016: stateData.performance_index_2016,
                  absolute_change: stateData.absolute_change,
                  relative_change: stateData.relative_change,
                  performance_score: Math.max(0, Math.min(1, stateData.performance_index_2021 || 0)),
                  total_indicators: stateData.total_indicators
                },
                geometry: geometry
              });
            }
          } else {
            // District boundary
            const districtName = boundary.district_name || boundary.district;
            const stateName = boundary.state_name || boundary.state;
            
            // Find district in state districts
            let districtData = null;
            for (const [state, districts] of Object.entries(stateInfo.stateDistricts)) {
              const found = districts.find(d => d.district_name === districtName);
              if (found) {
                districtData = found;
                break;
              }
            }

            if (districtData) {
              districtFeatures.push({
                type: 'Feature',
                properties: {
                  type: 'district',
                  district_name: districtName,
                  state_name: stateName || districtData.state_name,
                  performance_index_2021: districtData.performance_index_2021,
                  performance_index_2016: districtData.performance_index_2016,
                  absolute_change: districtData.absolute_change,
                  relative_change: districtData.relative_change,
                  performance_score: Math.max(0, Math.min(1, districtData.performance_index_2021 || 0)),
                  total_indicators: districtData.total_indicators
                },
                geometry: geometry
              });
            }
          }
        } catch (error) {
          // Skip invalid boundaries
        }
      });

      return { 
        stateFeatures: { type: 'FeatureCollection', features: stateFeatures },
        districtFeatures: { type: 'FeatureCollection', features: districtFeatures }
      };
    } catch (error) {
      return { 
        stateFeatures: { type: 'FeatureCollection', features: [] },
        districtFeatures: { type: 'FeatureCollection', features: [] }
      };
    }
  }, [data?.boundary_data, data?.boundary, data?.data, stateInfo.states, stateInfo.stateDistricts]);

  // Organize charts by category for pagination
  const chartPages = useMemo(() => {
    if (!stateInfo.chartData || stateInfo.chartData.length === 0) {
      return [];
    }

    // Categorize charts by type/purpose
    const statePerformanceChart = stateInfo.chartData.find(chart => 
      chart.title?.toLowerCase().includes('state') && 
      chart.title?.toLowerCase().includes('performance') && 
      !chart.title?.toLowerCase().includes('change')
    );
    
    const stateChangeChart = stateInfo.chartData.find(chart => 
      chart.title?.toLowerCase().includes('state') && 
      chart.title?.toLowerCase().includes('change')
    );

    // District charts by state
    const districtCharts = stateInfo.chartData.filter(chart => 
      chart.title && 
      !chart.title.toLowerCase().includes('state') &&
      (chart.title.includes(':') || chart.title.toLowerCase().includes('district'))
    );

    const pages = [];

    // Page 1: State Performance Comparison
    if (statePerformanceChart) {
      pages.push({
        title: "🏛️ State Performance Comparison",
        subtitle: "Multi-indicator performance index by state",
        charts: [statePerformanceChart],
        description: "This chart compares the overall health performance across states using a comprehensive composite index calculated from all selected health indicators. Higher values indicate better state-level performance.",
        gradient: "from-blue-500 to-blue-600",
        icon: "🏛️"
      });
    }

    // Page 2: State Performance Change
    if (stateChangeChart) {
      pages.push({
        title: "📈 State Performance Trends",
        subtitle: "Change in state performance from 2016 to 2021",
        charts: [stateChangeChart],
        description: "This chart shows how state-level health performance has evolved over the 5-year period, highlighting states with significant improvements and identifying those needing attention.",
        gradient: "from-green-500 to-green-600",
        icon: "📈"
      });
    }

    // Pages 3+: District charts by state (group by state)
    const stateDistrictCharts = {};
    districtCharts.forEach(chart => {
      const stateName = chart.title.split(':')[0];
      if (!stateDistrictCharts[stateName]) {
        stateDistrictCharts[stateName] = [];
      }
      stateDistrictCharts[stateName].push(chart);
    });

    Object.entries(stateDistrictCharts).forEach(([stateName, charts]) => {
      pages.push({
        title: `🏙️ ${stateName} Districts`,
        subtitle: `Top/bottom performing districts in ${stateName}`,
        charts: charts,
        description: `This analysis shows the ${stateInfo.performanceType} performing districts within ${stateName} based on multi-indicator performance. Districts are ranked using the same methodology as state-level analysis.`,
        gradient: "from-purple-500 to-purple-600",
        icon: "🏙️"
      });
    });

    return pages;
  }, [stateInfo.chartData, stateInfo.performanceType]);

  // Enhanced Chart component
  const EnhancedChart = ({ chartConfig, title = null }) => {
    if (!chartConfig) {
      return <div className="text-gray-500 p-4">No chart configuration provided</div>;
    }

    let chartData = chartConfig.data || chartConfig;

    if (!chartData || (!chartData.labels && !chartData.datasets)) {
      return <div className="text-gray-500 p-4">Invalid chart data structure</div>;
    }

    if (!chartData.datasets || !Array.isArray(chartData.datasets) || chartData.datasets.length === 0) {
      return <div className="text-gray-500 p-4">No chart datasets available</div>;
    }

    // Force bar charts only based on user preference
    const ChartComponent = Bar;
    
    // Calculate dynamic min/max values from actual data
    const allDataValues = chartData.datasets.flatMap(dataset => 
      dataset.data ? dataset.data.filter(v => v !== null && v !== undefined && !isNaN(v)) : []
    );
    
    const minValue = allDataValues.length > 0 ? Math.min(...allDataValues) : 0;
    const maxValue = allDataValues.length > 0 ? Math.max(...allDataValues) : 1;
    const isChangeChart = (chartConfig.title || title || '').toLowerCase().includes('change');
    
    // Add 10% padding to min/max for better visualization
    const padding = (maxValue - minValue) * 0.1;
    const dynamicMin = isChangeChart ? Math.min(minValue - padding, -0.1) : Math.max(0, minValue - padding);
    const dynamicMax = isChangeChart ? Math.max(maxValue + padding, 0.1) : maxValue + padding;

    const chartOptions = {
      responsive: true,
      maintainAspectRatio: false,
      interaction: {
        intersect: false,
        mode: 'index'
      },
      layout: {
        padding: {
          left: 10,
          right: 10,
          top: 30,
          bottom: 60
        }
      },
      plugins: {
        legend: {
          display: false
        },
        title: {
          display: true,
          text: title || chartConfig.title || 'State Multi-Indicator Chart',
          font: {
            size: 22,
            weight: 'bold'
          },
          padding: {
            top: 10,
            bottom: 30
          }
        },
        tooltip: {
          mode: 'index',
          intersect: false,
          titleFont: {
            size: 16
          },
          bodyFont: {
            size: 14
          },
          padding: 12,
          cornerRadius: 8,
          callbacks: {
            label: function(context) {
              const label = context.dataset.label || '';
              const value = context.parsed.y;
              const formattedValue = isChangeChart ? 
                `${value > 0 ? '+' : ''}${(value * 100).toFixed(1)}%` :
                `${(value * 100).toFixed(1)}%`;
              return `${label}: ${formattedValue}`;
            }
          }
        }
      },
      scales: {
        x: {
          ticks: {
            font: {
              size: 11,
              weight: '500'
            },
            maxRotation: 45,
            minRotation: 0,
            padding: 5,
            maxTicksLimit: 15,
            callback: function(value, index, values) {
              const label = this.getLabelForValue(value);
              // Truncate long names
              return label && label.length > 15 ? label.substring(0, 12) + '...' : label;
            }
          },
          grid: {
            display: false
          },
          title: {
            display: true,
            text: chartConfig.title?.toLowerCase().includes('state') ? 'States' : 'Districts',
            font: {
              size: 12,
              weight: 'bold'
            }
          }
        },
        y: {
          ticks: {
            font: {
              size: 12,
              weight: '500'
            },
            padding: 8,
            stepSize: isChangeChart ? 0.05 : 0.1,
            callback: function(value) {
              return isChangeChart ? 
                `${value > 0 ? '+' : ''}${(value * 100).toFixed(1)}%` :
                `${(value * 100).toFixed(0)}%`;
            }
          },
          grid: {
            color: 'rgba(0, 0, 0, 0.1)',
            lineWidth: 1
          },
          title: {
            display: true,
            text: isChangeChart ? 'Change in Performance Index' : 'Performance Index',
            font: {
              size: 13,
              weight: 'bold'
            }
          },
          beginAtZero: !isChangeChart,
          min: dynamicMin,
          max: dynamicMax,
          grace: '5%'
        }
      },
      elements: {
        bar: {
          borderRadius: 6,
          borderSkipped: false,
        }
      },
      datasets: {
        bar: {
          maxBarThickness: 50,
          minBarLength: 5,
          categoryPercentage: 0.9,
          barPercentage: 0.8
        }
      },
      indexAxis: chartConfig.indexAxis || 'x',
      ...(chartConfig.options || {})
    };

    // Prepare final chart data with validation and data cleaning
    const finalChartData = {
      labels: chartData.labels || [],
      datasets: chartData.datasets.map(dataset => {
        const cleanData = (dataset.data || []).map(value => {
          const numValue = Number(value);
          return isNaN(numValue) ? 0 : numValue;
        });
        
        return {
          ...dataset,
          data: cleanData,
          backgroundColor: dataset.backgroundColor || '#3B82F6',
          borderColor: dataset.borderColor || '#2563EB',
          borderWidth: dataset.borderWidth || 1,
          minBarLength: 5
        };
      })
    };

    return (
      <div className="w-full h-full" style={{ position: 'relative', minHeight: '400px' }}>
        <ChartComponent 
          data={finalChartData} 
          options={chartOptions}
          style={{ maxHeight: '100%', maxWidth: '100%' }}
        />
      </div>
    );
  };

  // Map click handler
  const onMapClick = useCallback((event) => {
    if (!event.features || !event.features[0] || !event.lngLat) {
      return;
    }

    const lng = event.lngLat.lng;
    const lat = event.lngLat.lat;
    
    if (isNaN(lng) || isNaN(lat)) {
      return;
    }

    const feature = event.features[0];
    const featureType = feature.properties?.type;
    
    if (featureType === 'state') {
      const stateName = feature.properties?.state_name;
      const stateData = stateInfo.states.find(s => s.state_name === stateName);
      
      if (stateData) {
        setPopupInfo({
          longitude: lng,
          latitude: lat,
          type: 'state',
          data: stateData,
          districts: stateInfo.stateDistricts[stateName] || []
        });
      }
    } else if (featureType === 'district') {
      const districtName = feature.properties?.district_name;
      const stateName = feature.properties?.state_name;
      
      // Find district data
      let districtData = null;
      for (const [state, districts] of Object.entries(stateInfo.stateDistricts)) {
        const found = districts.find(d => d.district_name === districtName);
        if (found) {
          districtData = found;
          break;
        }
      }

      if (districtData) {
        setPopupInfo({
          longitude: lng,
          latitude: lat,
          type: 'district',
          data: districtData
        });
      }
    }
  }, [stateInfo.states, stateInfo.stateDistricts]);

  // Map hover handlers
  const onMouseEnter = useCallback((event) => {
    if (event.features && event.features[0]) {
      const feature = event.features[0];
      const featureName = feature.properties?.state_name || feature.properties?.district_name;
      setHoveredFeature(featureName);
      if (mapRef.current) {
        mapRef.current.getCanvas().style.cursor = 'pointer';
      }
    }
  }, []);

  const onMouseLeave = useCallback(() => {
    setHoveredFeature(null);
    if (mapRef.current) {
      mapRef.current.getCanvas().style.cursor = '';
    }
  }, []);

  if (chartOnly) {
    const currentPage = chartPages[currentChartPage] || null;

    return (
      <div className="space-y-6">
        {chartPages.length > 0 ? (
          <>
            {/* Enhanced Page Navigation Header */}
            <div className="bg-gradient-to-r from-gray-50 to-white rounded-xl shadow-lg border border-gray-100 overflow-hidden">
              <div className={`bg-gradient-to-r ${currentPage?.gradient || 'from-blue-500 to-purple-600'} p-6 text-white`}>
                <div className="flex justify-between items-start">
                  <div className="flex items-center gap-3">
                    <div className="text-3xl">{currentPage?.icon || '🏛️'}</div>
                    <div>
                      <h2 className="text-2xl font-bold">{currentPage?.title}</h2>
                      <p className="text-blue-100 mt-1">{currentPage?.subtitle}</p>
                    </div>
                  </div>
                  
                  <div className="text-right">
                    <div className="bg-white/20 backdrop-blur-sm rounded-lg px-3 py-1 text-sm">
                      Page {currentChartPage + 1} of {chartPages.length}
                    </div>
                  </div>
                </div>
              </div>

              {/* Navigation Controls */}
              <div className="p-6 bg-white">
                <div className="flex justify-between items-center">
                  <button
                    onClick={() => setCurrentChartPage(Math.max(0, currentChartPage - 1))}
                    disabled={currentChartPage === 0}
                    className={`group flex items-center gap-2 px-6 py-3 rounded-xl font-medium transition-all duration-300 ${
                      currentChartPage === 0
                        ? 'bg-gray-100 text-gray-400 cursor-not-allowed'
                        : 'bg-gradient-to-r from-blue-500 to-blue-600 text-white hover:from-blue-600 hover:to-blue-700 shadow-lg hover:shadow-xl transform hover:-translate-y-0.5'
                    }`}
                  >
                    <span className="transition-transform group-hover:-translate-x-1">←</span>
                    Previous
                  </button>

                  <div className="flex gap-3">
                    {chartPages.map((page, index) => (
                      <button
                        key={index}
                        onClick={() => setCurrentChartPage(index)}
                        className={`group relative overflow-hidden rounded-xl transition-all duration-300 ${
                          index === currentChartPage
                            ? 'w-16 h-16 bg-gradient-to-br from-blue-500 to-purple-600 text-white shadow-lg'
                            : 'w-12 h-12 bg-gray-100 text-gray-600 hover:bg-gray-200 hover:shadow-md'
                        }`}
                        title={page.title}
                      >
                        <div className="absolute inset-0 flex items-center justify-center">
                          {index === currentChartPage ? (
                            <div className="text-center">
                              <div className="text-lg">{page.icon}</div>
                              <div className="text-xs font-bold">{index + 1}</div>
                            </div>
                          ) : (
                            <div className="font-semibold">{index + 1}</div>
                          )}
                        </div>
                      </button>
                    ))}
                  </div>

                  <button
                    onClick={() => setCurrentChartPage(Math.min(chartPages.length - 1, currentChartPage + 1))}
                    disabled={currentChartPage === chartPages.length - 1}
                    className={`group flex items-center gap-2 px-6 py-3 rounded-xl font-medium transition-all duration-300 ${
                      currentChartPage === chartPages.length - 1
                        ? 'bg-gray-100 text-gray-400 cursor-not-allowed'
                        : 'bg-gradient-to-r from-green-500 to-green-600 text-white hover:from-green-600 hover:to-green-700 shadow-lg hover:shadow-xl transform hover:-translate-y-0.5'
                    }`}
                  >
                    Next
                    <span className="transition-transform group-hover:translate-x-1">→</span>
                  </button>
                </div>
              </div>
            </div>

            {/* Current Page Chart */}
            {currentPage && (
              <div className="bg-white rounded-xl shadow-xl border border-gray-100 overflow-hidden">
                <div className={`bg-gradient-to-r ${currentPage.gradient} p-6 text-white`}>
                  <div className="flex items-start gap-4">
                    <div className="text-4xl">{currentPage.icon}</div>
                    <div className="flex-1">
                      <h3 className="text-xl font-bold mb-2">Analysis Insights</h3>
                      <p className="text-blue-100 leading-relaxed">
                        {currentPage.description}
                      </p>
                    </div>
                  </div>
                </div>

                <div className="p-8">
                  {currentPage.charts.map((chart, index) => (
                    <div key={index} className="w-full">
                      <div className="relative">
                        <div className={`absolute inset-0 bg-gradient-to-r ${currentPage.gradient} rounded-xl opacity-10`}></div>
                        <div className="relative bg-white rounded-xl border-2 border-gray-100 p-6" style={{ minHeight: '600px' }}>
                          <div className="w-full h-full" style={{ height: '550px', position: 'relative' }}>
                            <EnhancedChart chartConfig={chart} />
                          </div>
                        </div>
                      </div>
                    </div>
                  ))}
                </div>
              </div>
            )}
          </>
        ) : (
          <div className="text-center py-8 text-gray-500">
            <p>No chart data available for state multi-indicator visualization</p>
          </div>
        )}
      </div>
    );
  }

  if (mapOnly) {
    return (
      <div className="space-y-4">
        {/* Layer toggles */}
        <div className="bg-white rounded-lg shadow p-4">
          <h4 className="text-sm font-semibold text-gray-800 mb-3">Map Layers</h4>
          <div className="flex gap-4">
            <label className="flex items-center gap-2">
              <input
                type="checkbox"
                checked={showStates}
                onChange={(e) => setShowStates(e.target.checked)}
                className="rounded"
              />
              <span className="text-sm">State Boundaries</span>
            </label>
            <label className="flex items-center gap-2">
              <input
                type="checkbox"
                checked={showDistricts}
                onChange={(e) => setShowDistricts(e.target.checked)}
                className="rounded"
              />
              <span className="text-sm">District Boundaries</span>
            </label>
          </div>
        </div>

        {/* Map visualization */}
        <div className="bg-white rounded-lg shadow overflow-hidden">
          <div className="h-96 relative">
            <Map
              ref={mapRef}
              mapboxAccessToken={mapboxAccessToken}
              initialViewState={{
                longitude: 78.9629,
                latitude: 20.5937,
                zoom: 4
              }}
              style={{ width: '100%', height: '100%' }}
              mapStyle="mapbox://styles/mapbox/light-v11"
              onClick={onMapClick}
              onMouseMove={onMouseEnter}
              onMouseLeave={onMouseLeave}
              interactiveLayerIds={[
                ...(showStates ? ['state-fill'] : []),
                ...(showDistricts ? ['district-fill'] : [])
              ]}
              cursor="pointer"
            >
              {/* State boundaries */}
              {showStates && boundaryData.stateFeatures.features.length > 0 && (
                <Source id="state-boundaries" type="geojson" data={boundaryData.stateFeatures}>
                  <Layer
                    id="state-fill"
                    type="fill"
                    paint={{
                      'fill-color': [
                        'interpolate',
                        ['linear'],
                        ['get', 'performance_score'],
                        0, '#E74C3C',
                        0.2, '#F39C12',
                        0.4, '#F1C40F',
                        0.6, '#52C41A',
                        0.8, '#27AE60',
                        1.0, '#2ECC71'
                      ],
                      'fill-opacity': [
                        'case',
                        ['==', ['get', 'state_name'], hoveredFeature || ''],
                        0.8,
                        0.5
                      ]
                    }}
                  />
                  <Layer
                    id="state-stroke"
                    type="line"
                    paint={{
                      'line-color': '#2C3E50',
                      'line-width': [
                        'case',
                        ['==', ['get', 'state_name'], hoveredFeature || ''],
                        3,
                        2
                      ]
                    }}
                  />
                </Source>
              )}

              {/* District boundaries */}
              {showDistricts && boundaryData.districtFeatures.features.length > 0 && (
                <Source id="district-boundaries" type="geojson" data={boundaryData.districtFeatures}>
                  <Layer
                    id="district-fill"
                    type="fill"
                    paint={{
                      'fill-color': [
                        'interpolate',
                        ['linear'],
                        ['get', 'performance_score'],
                        0, '#E74C3C',
                        0.2, '#F39C12',
                        0.4, '#F1C40F',
                        0.6, '#52C41A',
                        0.8, '#27AE60',
                        1.0, '#2ECC71'
                      ],
                      'fill-opacity': [
                        'case',
                        ['==', ['get', 'district_name'], hoveredFeature || ''],
                        0.8,
                        0.6
                      ]
                    }}
                  />
                  <Layer
                    id="district-stroke"
                    type="line"
                    paint={{
                      'line-color': [
                        'case',
                        ['==', ['get', 'district_name'], hoveredFeature || ''],
                        '#2C3E50',
                        '#7F8C8D'
                      ],
                      'line-width': [
                        'case',
                        ['==', ['get', 'district_name'], hoveredFeature || ''],
                        2,
                        1
                      ]
                    }}
                  />
                </Source>
              )}

              {/* Popup for feature details */}
              {popupInfo && !isNaN(popupInfo.longitude) && !isNaN(popupInfo.latitude) && (
                <Popup
                  longitude={popupInfo.longitude}
                  latitude={popupInfo.latitude}
                  anchor="top"
                  onClose={() => setPopupInfo(null)}
                  maxWidth="400px"
                >
                  <div className="p-3">
                    {popupInfo.type === 'state' ? (
                      <>
                        <h3 className="font-bold text-base text-gray-800 mb-2">
                          🏛️ {popupInfo.data.state_name}
                        </h3>
                        
                        <div className="space-y-2 text-sm">
                          <div className="flex justify-between">
                            <span className="text-gray-600">State Performance Index (2021):</span>
                            <span className="font-semibold text-blue-600">
                              {(popupInfo.data.performance_index_2021 * 100).toFixed(1)}%
                            </span>
                          </div>
                          
                          <div className="flex justify-between">
                            <span className="text-gray-600">Performance Index (2016):</span>
                            <span className="font-semibold text-gray-700">
                              {(popupInfo.data.performance_index_2016 * 100).toFixed(1)}%
                            </span>
                          </div>
                          
                          <div className="flex justify-between">
                            <span className="text-gray-600">Change:</span>
                            <span className={`font-semibold ${
                              popupInfo.data.absolute_change > 0 ? 'text-green-600' : 
                              popupInfo.data.absolute_change < 0 ? 'text-red-600' : 'text-gray-600'
                            }`}>
                              {popupInfo.data.absolute_change > 0 ? '+' : ''}
                              {(popupInfo.data.absolute_change * 100).toFixed(1)}%
                            </span>
                          </div>

                          <div className="flex justify-between">
                            <span className="text-gray-600">Districts Analyzed:</span>
                            <span className="font-semibold text-gray-700">
                              {popupInfo.districts.length}
                            </span>
                          </div>

                          <div className="flex justify-between">
                            <span className="text-gray-600">Total Indicators:</span>
                            <span className="font-semibold text-gray-700">
                              {popupInfo.data.total_indicators}
                            </span>
                          </div>
                        </div>

                        {/* Districts list */}
                        {popupInfo.districts.length > 0 && (
                          <div className="mt-3 p-2 bg-gray-50 rounded">
                            <div className="text-xs font-medium text-gray-700 mb-1">
                              {stateInfo.performanceType === 'top' ? 'Top' : 
                               stateInfo.performanceType === 'bottom' ? 'Bottom' : 'Selected'} Districts:
                            </div>
                            <div className="text-xs text-gray-600">
                              {popupInfo.districts.map(d => d.district_name).join(', ')}
                            </div>
                          </div>
                        )}
                      </>
                    ) : (
                      <>
                        <h3 className="font-bold text-base text-gray-800 mb-2">
                          🏙️ {popupInfo.data.district_name}
                        </h3>
                        <p className="text-sm text-gray-600 mb-3">
                          {popupInfo.data.state_name}
                        </p>
                        
                        <div className="space-y-2 text-sm">
                          <div className="flex justify-between">
                            <span className="text-gray-600">Performance Index (2021):</span>
                            <span className="font-semibold text-blue-600">
                              {(popupInfo.data.performance_index_2021 * 100).toFixed(1)}%
                            </span>
                          </div>
                          
                          <div className="flex justify-between">
                            <span className="text-gray-600">Performance Index (2016):</span>
                            <span className="font-semibold text-gray-700">
                              {(popupInfo.data.performance_index_2016 * 100).toFixed(1)}%
                            </span>
                          </div>
                          
                          <div className="flex justify-between">
                            <span className="text-gray-600">Change:</span>
                            <span className={`font-semibold ${
                              popupInfo.data.absolute_change > 0 ? 'text-green-600' : 
                              popupInfo.data.absolute_change < 0 ? 'text-red-600' : 'text-gray-600'
                            }`}>
                              {popupInfo.data.absolute_change > 0 ? '+' : ''}
                              {(popupInfo.data.absolute_change * 100).toFixed(1)}%
                            </span>
                          </div>

                          <div className="flex justify-between">
                            <span className="text-gray-600">Total Indicators:</span>
                            <span className="font-semibold text-gray-700">
                              {popupInfo.data.total_indicators}
                            </span>
                          </div>
                        </div>
                      </>
                    )}

                    {/* Performance rating */}
                    <div className="mt-3 p-2 rounded" style={{
                      backgroundColor: popupInfo.data.performance_index_2021 >= 0.8 ? '#d4edda' :
                                     popupInfo.data.performance_index_2021 >= 0.6 ? '#fff3cd' :
                                     popupInfo.data.performance_index_2021 >= 0.4 ? '#ffeaa7' :
                                     '#f8d7da'
                    }}>
                      <div className="text-xs font-medium text-center">
                        {popupInfo.data.performance_index_2021 >= 0.8 ? 'Excellent Performance' :
                         popupInfo.data.performance_index_2021 >= 0.6 ? 'Good Performance' :
                         popupInfo.data.performance_index_2021 >= 0.4 ? 'Average Performance' :
                         'Needs Improvement'}
                      </div>
                    </div>
                  </div>
                </Popup>
              )}
            </Map>
          </div>
          
          {/* Legend */}
          <div className="p-4 bg-gray-50 border-t">
            <h4 className="text-sm font-semibold text-gray-800 mb-2">Performance Index Legend</h4>
            <div className="flex flex-wrap gap-4 text-xs">
              <div className="flex items-center gap-2">
                <div className="w-4 h-4 rounded" style={{ backgroundColor: '#2ECC71' }}></div>
                <span>Excellent (80-100%)</span>
              </div>
              <div className="flex items-center gap-2">
                <div className="w-4 h-4 rounded" style={{ backgroundColor: '#52C41A' }}></div>
                <span>Good (60-80%)</span>
              </div>
              <div className="flex items-center gap-2">
                <div className="w-4 h-4 rounded" style={{ backgroundColor: '#F1C40F' }}></div>
                <span>Average (40-60%)</span>
              </div>
              <div className="flex items-center gap-2">
                <div className="w-4 h-4 rounded" style={{ backgroundColor: '#F39C12' }}></div>
                <span>Below Average (20-40%)</span>
              </div>
              <div className="flex items-center gap-2">
                <div className="w-4 h-4 rounded" style={{ backgroundColor: '#E74C3C' }}></div>
                <span>Poor (0-20%)</span>
              </div>
            </div>
          </div>
        </div>
      </div>
    );
  }

  // Full component with summary, map, and charts
  return (
    <div className="space-y-6">
      {/* Summary Information */}
      <div className="bg-white rounded-lg shadow p-6">
        <div className="grid grid-cols-1 md:grid-cols-4 gap-4">
          <div className="text-center">
            <div className="text-2xl font-bold text-blue-600">
              {stateInfo.totalStates}
            </div>
            <div className="text-sm text-gray-600">States Analyzed</div>
          </div>
          <div className="text-center">
            <div className="text-2xl font-bold text-purple-600">
              {stateInfo.totalDistricts}
            </div>
            <div className="text-sm text-gray-600">Districts Included</div>
          </div>
          <div className="text-center">
            <div className="text-2xl font-bold text-green-600">
              {stateInfo.totalIndicators}
            </div>
            <div className="text-sm text-gray-600">Health Indicators</div>
          </div>
          <div className="text-center">
            <div className="text-2xl font-bold text-orange-600">
              {stateInfo.categoryName || 'All Categories'}
            </div>
            <div className="text-sm text-gray-600">Category Focus</div>
          </div>
        </div>
      </div>

      {/* Map visualization */}
      <div className="bg-white rounded-lg shadow overflow-hidden">
        {/* Layer toggles */}
        <div className="p-4 bg-gray-50 border-b">
          <h4 className="text-sm font-semibold text-gray-800 mb-3">Map Layers</h4>
          <div className="flex gap-4">
            <label className="flex items-center gap-2">
              <input
                type="checkbox"
                checked={showStates}
                onChange={(e) => setShowStates(e.target.checked)}
                className="rounded"
              />
              <span className="text-sm">State Boundaries</span>
            </label>
            <label className="flex items-center gap-2">
              <input
                type="checkbox"
                checked={showDistricts}
                onChange={(e) => setShowDistricts(e.target.checked)}
                className="rounded"
              />
              <span className="text-sm">District Boundaries</span>
            </label>
          </div>
        </div>

        <div className="h-96 relative">
          <Map
            ref={mapRef}
            mapboxAccessToken={mapboxAccessToken}
            initialViewState={{
              longitude: 78.9629,
              latitude: 20.5937,
              zoom: 4
            }}
            style={{ width: '100%', height: '100%' }}
            mapStyle="mapbox://styles/mapbox/light-v11"
            onClick={onMapClick}
            onMouseMove={onMouseEnter}
            onMouseLeave={onMouseLeave}
            interactiveLayerIds={[
              ...(showStates ? ['state-fill'] : []),
              ...(showDistricts ? ['district-fill'] : [])
            ]}
            cursor="pointer"
          >
            {/* State boundaries */}
            {showStates && boundaryData.stateFeatures.features.length > 0 && (
              <Source id="state-boundaries" type="geojson" data={boundaryData.stateFeatures}>
                <Layer
                  id="state-fill"
                  type="fill"
                  paint={{
                    'fill-color': [
                      'interpolate',
                      ['linear'],
                      ['get', 'performance_score'],
                      0, '#E74C3C',
                      0.2, '#F39C12',
                      0.4, '#F1C40F',
                      0.6, '#52C41A',
                      0.8, '#27AE60',
                      1.0, '#2ECC71'
                    ],
                    'fill-opacity': [
                      'case',
                      ['==', ['get', 'state_name'], hoveredFeature || ''],
                      0.8,
                      0.5
                    ]
                  }}
                />
                <Layer
                  id="state-stroke"
                  type="line"
                  paint={{
                    'line-color': '#2C3E50',
                    'line-width': [
                      'case',
                      ['==', ['get', 'state_name'], hoveredFeature || ''],
                      3,
                      2
                    ]
                  }}
                />
              </Source>
            )}

            {/* District boundaries */}
            {showDistricts && boundaryData.districtFeatures.features.length > 0 && (
              <Source id="district-boundaries" type="geojson" data={boundaryData.districtFeatures}>
                <Layer
                  id="district-fill"
                  type="fill"
                  paint={{
                    'fill-color': [
                      'interpolate',
                      ['linear'],
                      ['get', 'performance_score'],
                      0, '#E74C3C',
                      0.2, '#F39C12',
                      0.4, '#F1C40F',
                      0.6, '#52C41A',
                      0.8, '#27AE60',
                      1.0, '#2ECC71'
                    ],
                    'fill-opacity': [
                      'case',
                      ['==', ['get', 'district_name'], hoveredFeature || ''],
                      0.8,
                      0.6
                    ]
                  }}
                />
                <Layer
                  id="district-stroke"
                  type="line"
                  paint={{
                    'line-color': [
                      'case',
                      ['==', ['get', 'district_name'], hoveredFeature || ''],
                      '#2C3E50',
                      '#7F8C8D'
                    ],
                    'line-width': [
                      'case',
                      ['==', ['get', 'district_name'], hoveredFeature || ''],
                      2,
                      1
                    ]
                  }}
                />
              </Source>
            )}

            {/* Popup for feature details */}
            {popupInfo && !isNaN(popupInfo.longitude) && !isNaN(popupInfo.latitude) && (
              <Popup
                longitude={popupInfo.longitude}
                latitude={popupInfo.latitude}
                anchor="top"
                onClose={() => setPopupInfo(null)}
                maxWidth="400px"
              >
                <div className="p-3">
                  {popupInfo.type === 'state' ? (
                    <>
                      <h3 className="font-bold text-base text-gray-800 mb-2">
                        🏛️ {popupInfo.data.state_name}
                      </h3>
                      
                      <div className="space-y-2 text-sm">
                        <div className="flex justify-between">
                          <span className="text-gray-600">State Performance Index (2021):</span>
                          <span className="font-semibold text-blue-600">
                            {(popupInfo.data.performance_index_2021 * 100).toFixed(1)}%
                          </span>
                        </div>
                        
                        <div className="flex justify-between">
                          <span className="text-gray-600">Performance Index (2016):</span>
                          <span className="font-semibold text-gray-700">
                            {(popupInfo.data.performance_index_2016 * 100).toFixed(1)}%
                          </span>
                        </div>
                        
                        <div className="flex justify-between">
                          <span className="text-gray-600">Change:</span>
                          <span className={`font-semibold ${
                            popupInfo.data.absolute_change > 0 ? 'text-green-600' : 
                            popupInfo.data.absolute_change < 0 ? 'text-red-600' : 'text-gray-600'
                          }`}>
                            {popupInfo.data.absolute_change > 0 ? '+' : ''}
                            {(popupInfo.data.absolute_change * 100).toFixed(1)}%
                          </span>
                        </div>

                        <div className="flex justify-between">
                          <span className="text-gray-600">Districts Analyzed:</span>
                          <span className="font-semibold text-gray-700">
                            {popupInfo.districts.length}
                          </span>
                        </div>

                        <div className="flex justify-between">
                          <span className="text-gray-600">Total Indicators:</span>
                          <span className="font-semibold text-gray-700">
                            {popupInfo.data.total_indicators}
                          </span>
                        </div>
                      </div>

                      {/* Districts list */}
                      {popupInfo.districts.length > 0 && (
                        <div className="mt-3 p-2 bg-gray-50 rounded">
                          <div className="text-xs font-medium text-gray-700 mb-1">
                            {stateInfo.performanceType === 'top' ? 'Top' : 
                             stateInfo.performanceType === 'bottom' ? 'Bottom' : 'Selected'} Districts:
                          </div>
                          <div className="text-xs text-gray-600">
                            {popupInfo.districts.map(d => d.district_name).join(', ')}
                          </div>
                        </div>
                      )}
                    </>
                  ) : (
                    <>
                      <h3 className="font-bold text-base text-gray-800 mb-2">
                        🏙️ {popupInfo.data.district_name}
                      </h3>
                      <p className="text-sm text-gray-600 mb-3">
                        {popupInfo.data.state_name}
                      </p>
                      
                      <div className="space-y-2 text-sm">
                        <div className="flex justify-between">
                          <span className="text-gray-600">Performance Index (2021):</span>
                          <span className="font-semibold text-blue-600">
                            {(popupInfo.data.performance_index_2021 * 100).toFixed(1)}%
                          </span>
                        </div>
                        
                        <div className="flex justify-between">
                          <span className="text-gray-600">Performance Index (2016):</span>
                          <span className="font-semibold text-gray-700">
                            {(popupInfo.data.performance_index_2016 * 100).toFixed(1)}%
                          </span>
                        </div>
                        
                        <div className="flex justify-between">
                          <span className="text-gray-600">Change:</span>
                          <span className={`font-semibold ${
                            popupInfo.data.absolute_change > 0 ? 'text-green-600' : 
                            popupInfo.data.absolute_change < 0 ? 'text-red-600' : 'text-gray-600'
                          }`}>
                            {popupInfo.data.absolute_change > 0 ? '+' : ''}
                            {(popupInfo.data.absolute_change * 100).toFixed(1)}%
                          </span>
                        </div>

                        <div className="flex justify-between">
                          <span className="text-gray-600">Total Indicators:</span>
                          <span className="font-semibold text-gray-700">
                            {popupInfo.data.total_indicators}
                          </span>
                        </div>
                      </div>
                    </>
                  )}

                  {/* Performance rating */}
                  <div className="mt-3 p-2 rounded" style={{
                    backgroundColor: popupInfo.data.performance_index_2021 >= 0.8 ? '#d4edda' :
                                   popupInfo.data.performance_index_2021 >= 0.6 ? '#fff3cd' :
                                   popupInfo.data.performance_index_2021 >= 0.4 ? '#ffeaa7' :
                                   '#f8d7da'
                  }}>
                    <div className="text-xs font-medium text-center">
                      {popupInfo.data.performance_index_2021 >= 0.8 ? 'Excellent Performance' :
                       popupInfo.data.performance_index_2021 >= 0.6 ? 'Good Performance' :
                       popupInfo.data.performance_index_2021 >= 0.4 ? 'Average Performance' :
                       'Needs Improvement'}
                    </div>
                  </div>
                </div>
              </Popup>
            )}
          </Map>
        </div>
        
        {/* Legend */}
        <div className="p-4 bg-gray-50 border-t">
          <h4 className="text-sm font-semibold text-gray-800 mb-2">Performance Index Legend</h4>
          <div className="flex flex-wrap gap-4 text-xs">
            <div className="flex items-center gap-2">
              <div className="w-4 h-4 rounded" style={{ backgroundColor: '#2ECC71' }}></div>
              <span>Excellent (80-100%)</span>
            </div>
            <div className="flex items-center gap-2">
              <div className="w-4 h-4 rounded" style={{ backgroundColor: '#52C41A' }}></div>
              <span>Good (60-80%)</span>
            </div>
            <div className="flex items-center gap-2">
              <div className="w-4 h-4 rounded" style={{ backgroundColor: '#F1C40F' }}></div>
              <span>Average (40-60%)</span>
            </div>
            <div className="flex items-center gap-2">
              <div className="w-4 h-4 rounded" style={{ backgroundColor: '#F39C12' }}></div>
              <span>Below Average (20-40%)</span>
            </div>
            <div className="flex items-center gap-2">
              <div className="w-4 h-4 rounded" style={{ backgroundColor: '#E74C3C' }}></div>
              <span>Poor (0-20%)</span>
            </div>
          </div>
        </div>
      </div>

      {/* Chart visualizations */}
      {stateInfo.chartData.length > 0 && (
        <div className="grid grid-cols-1 lg:grid-cols-2 gap-6">
          {stateInfo.chartData.map((chart, index) => (
            <div key={index} className="bg-white rounded-lg shadow p-4">
              <EnhancedChart chartConfig={chart} />
              {chart.description && (
                <p className="text-sm text-gray-600 mt-2 text-center">
                  {chart.description}
                </p>
              )}
            </div>
          ))}
        </div>
      )}

      {/* Analysis text */}
      {stateInfo.analysis && (
        <div className="bg-gray-50 rounded-lg p-6">
          <h3 className="text-lg font-semibold text-gray-800 mb-4">📋 Analysis</h3>
          <div 
            className="text-gray-700 whitespace-pre-line leading-relaxed"
            dangerouslySetInnerHTML={{ 
              __html: stateInfo.analysis.replace(/\n/g, '<br/>') 
            }}
          />
        </div>
      )}
    </div>
  );
};

export default StateMultiIndicatorPerformance;
