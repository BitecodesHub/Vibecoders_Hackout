import React, { useEffect, useRef, useState } from "react";
import {
  Search,
  MapPin,
  Layers,
  Satellite,
  Moon,
  Zap,
  Navigation,
  AlertCircle,
  RefreshCw,
  TrendingUp,
  Droplets,
  Wind,
  Sun,
  Activity,
  Target,
} from "lucide-react";

const MapComponent = () => {
  const mapRef = useRef(null);
  const mapInstanceRef = useRef(null);
  const markersRef = useRef([]);
  const [searchQuery, setSearchQuery] = useState("");
  const [currentLayer, setCurrentLayer] = useState("street");
  const [isSearching, setIsSearching] = useState(false);
  const [searchResults, setSearchResults] = useState([]);
  const [selectedCity, setSelectedCity] = useState("");
  const [isGenerating, setIsGenerating] = useState(false);
  const [showResults, setShowResults] = useState(false);
  const [backendSites, setBackendSites] = useState([]);
  const [loading, setLoading] = useState(false);
  const [error, setError] = useState(null);
  const [cap, setCap] = useState("");

  useEffect(() => {
    const fetchBackendSites = async () => {
      try {
        setLoading(true);
        setError(null);
        const response = await fetch(
          "http://localhost:8080/ml/sites/topp"
        );
        if (!response.ok) {
          throw new Error(`Failed to fetch sites data: ${response.status}`);
        }
        const data = await response.json();
        setBackendSites(data);
      } catch (err) {
        setError(err.message);
        console.error("Error fetching backend sites:", err);
      } finally {
        setLoading(false);
      }
    };

    fetchBackendSites();
  }, []);

  // Map layer configurations
  const mapLayers = {
    street: {
      url: "https://{s}.tile.openstreetmap.org/{z}/{x}/{y}.png",
      attribution: "© OpenStreetMap contributors",
      name: "Street",
    },
    satellite: {
      url: "https://server.arcgisonline.com/ArcGIS/rest/services/World_Imagery/MapServer/tile/{z}/{y}/{x}",
      attribution: "© Esri",
      name: "Satellite",
    },
    dark: {
      url: "https://{s}.basemaps.cartocdn.com/dark_all/{z}/{x}/{y}{r}.png",
      attribution: "© CartoDB",
      name: "Dark",
    },
  };

  useEffect(() => {
    const loadLeaflet = async () => {
      if (!document.querySelector('link[href*="leaflet"]')) {
        const cssLink = document.createElement("link");
        cssLink.rel = "stylesheet";
        cssLink.href =
          "https://cdnjs.cloudflare.com/ajax/libs/leaflet/1.9.4/leaflet.min.css";
        document.head.appendChild(cssLink);
      }

      if (!window.L) {
        const script = document.createElement("script");
        script.src =
          "https://cdnjs.cloudflare.com/ajax/libs/leaflet/1.9.4/leaflet.min.js";
        document.head.appendChild(script);

        script.onload = () => {
          initializeMap();
        };
      } else {
        initializeMap();
      }
    };

    loadLeaflet();

    return () => {
      if (mapInstanceRef.current) {
        mapInstanceRef.current.remove();
      }
    };
  }, []);

  useEffect(() => {
    if (mapInstanceRef.current && backendSites.length > 0) {
      addHydrogenSiteMarkers(mapInstanceRef.current);
    }
  }, [backendSites]);

  const initializeMap = () => {
    if (!mapRef.current || mapInstanceRef.current) return;

    const map = window.L.map(mapRef.current, {
      zoomControl: false,
      attributionControl: false,
    }).setView([23.0225, 72.5714], 7);

    window.L.control.zoom({ position: "bottomright" }).addTo(map);

    window.L.control
      .attribution({ position: "bottomleft", prefix: false })
      .addTo(map);

    const initialLayer = window.L.tileLayer(mapLayers[currentLayer].url, {
      attribution: mapLayers[currentLayer].attribution,
      maxZoom: 19,
    }).addTo(map);

    mapInstanceRef.current = map;

    if (backendSites.length > 0) {
      addHydrogenSiteMarkers(map);
    }
  };

  const getEfficiencyColor = (score) => {
    if (score >= 90) return "emerald";
    if (score >= 80) return "green";
    if (score >= 70) return "yellow";
    if (score >= 60) return "orange";
    return "red";
  };

  const getFeasibilityIcon = (score) => {
    if (score >= 0.9) return "⭐";
    if (score >= 0.8) return "🟢";
    if (score >= 0.7) return "🟡";
    if (score >= 0.6) return "🟠";
    return "🔴";
  };

  const addHydrogenSiteMarkers = (map) => {
    markersRef.current.forEach((marker) => map.removeLayer(marker));
    markersRef.current = [];

    backendSites.forEach((site, index) => {
      const latitude = Number(site.Latitude);
      const longitude = Number(site.Longitude);
      const siteName = site.City || `Hydrogen Site ${index + 1}`;
      const production = site["Hydrogen_Production_kg/day"] || 0;
      const feasibility = site.Feasibility_Score || 0;
      const efficiency = site["System_Efficiency_%"] || 0;
      const pvPower = site.PV_Power_kW || 0;
      const windPower = site.Wind_Power_kW || 0;
      const solarIrradiance = site["Solar_Irradiance_kWh/m²/day"] || 0;
      const windSpeed = site["Wind_Speed_m/s"] || 0;
      const temperature = site.Temperature_C || 0;
      const electrolyzerEff = site["Electrolyzer_Efficiency_%"] || 0;

      if (isNaN(latitude) || isNaN(longitude)) {
        return;
      }

      // Enhanced marker design with modern styling
      const getFeasibilityTier = (score) => {
        if (score >= 0.9)
          return {
            tier: "excellent",
            color: "emerald",
            ring: "ring-emerald-400/50",
          };
        if (score >= 0.8)
          return { tier: "good", color: "green", ring: "ring-green-400/50" };
        if (score >= 0.7)
          return { tier: "fair", color: "yellow", ring: "ring-yellow-400/50" };
        if (score >= 0.6)
          return { tier: "poor", color: "orange", ring: "ring-orange-400/50" };
        return { tier: "critical", color: "red", ring: "ring-red-400/50" };
      };

      const { tier, color, ring } = getFeasibilityTier(feasibility);
      const isHighPerformance = feasibility >= 0.9;
      const totalPower = pvPower + windPower;

      const customIcon = window.L.divIcon({
        html: `
        <div class="relative">
          <!-- Main marker with glassmorphism effect -->
          <div class="w-12 h-12 bg-gradient-to-br from-${color}-400 via-${color}-500 to-${color}-600 rounded-xl border-2 border-white/60 shadow-xl backdrop-blur-sm flex items-center justify-center transform hover:scale-110 transition-all duration-300 ${
          isHighPerformance ? "animate-pulse ring-4 " + ring : ""
        } group cursor-pointer">
            <!-- Inner glow effect -->
            <div class="absolute inset-0 bg-gradient-to-br from-white/20 to-transparent rounded-xl"></div>
            
            <!-- Hydrogen icon -->
            <div class="relative z-10 w-7 h-7 bg-white/25 backdrop-blur-sm rounded-lg flex items-center justify-center group-hover:bg-white/35 transition-all duration-200">
              <svg class="w-4 h-4 text-white drop-shadow-sm" fill="currentColor" viewBox="0 0 24 24">
                <path d="M4.5 7c0-1.93 1.57-3.5 3.5-3.5s3.5 1.57 3.5 3.5-1.57 3.5-3.5 3.5S4.5 8.93 4.5 7zm8.5 0c0-1.93 1.57-3.5 3.5-3.5S20 5.07 20 7s-1.57 3.5-3.5 3.5S13 8.93 13 7z"/>
                <path d="M9 12h6v2H9z"/>
                <path d="M4.5 17c0-1.93 1.57-3.5 3.5-3.5s3.5 1.57 3.5 3.5-1.57 3.5-3.5 3.5S4.5 18.93 4.5 17zm8.5 0c0-1.93 1.57-3.5 3.5-3.5s3.5 1.57 3.5 3.5-1.57 3.5-3.5 3.5S13 18.93 13 17z"/>
              </svg>
            </div>
          </div>
          
          <!-- Performance indicator badge -->
          <div class="absolute -top-2 -right-2 w-6 h-6 bg-white rounded-full border-2 border-${color}-400 shadow-lg flex items-center justify-center text-xs font-bold text-${color}-600">
            ${getFeasibilityIcon(feasibility)}
          </div>
          
          <!-- Power indicator dots -->
          ${
            totalPower > 1000
              ? `
            <div class="absolute -bottom-1 -left-1 w-3 h-3 bg-yellow-400 rounded-full border border-white shadow animate-ping"></div>
            <div class="absolute -bottom-1 -left-1 w-3 h-3 bg-yellow-400 rounded-full border border-white shadow"></div>
          `
              : ""
          }
        </div>
      `,
        className: "enhanced-hydrogen-marker",
        iconSize: [48, 48],
        iconAnchor: [24, 48],
      });

      const marker = window.L.marker([latitude, longitude], {
        icon: customIcon,
      }).addTo(map);

      // Compact horizontal popup design
      const getStatusColor = (value, thresholds) => {
        if (value >= thresholds.excellent)
          return "text-emerald-600 bg-emerald-50";
        if (value >= thresholds.good) return "text-green-600 bg-green-50";
        if (value >= thresholds.fair) return "text-yellow-600 bg-yellow-50";
        return "text-red-600 bg-red-50";
      };

      const productionStatus = getStatusColor(production, {
        excellent: 100,
        good: 50,
        fair: 20,
      });
      const efficiencyStatus = getStatusColor(efficiency, {
        excellent: 85,
        good: 75,
        fair: 65,
      });

      const popupContent = `
      <div class="relative overflow-hidden">
        <!-- Glassmorphism background -->
        <div class="absolute inset-0 bg-gradient-to-br from-white/90 via-white/80 to-white/70 backdrop-blur-xl"></div>
        
        <!-- Content container - Horizontal Layout -->
        <div class="relative z-10 p-4 w-[500px]">
          
          <!-- Header Section - Compact -->
          <div class="flex items-center justify-between mb-4">
            <div class="flex items-center space-x-3">
              <div class="w-10 h-10 bg-gradient-to-br from-${color}-400 to-${color}-600 rounded-lg flex items-center justify-center shadow-lg">
                <svg class="w-5 h-5 text-white" fill="currentColor" viewBox="0 0 24 24">
                  <path d="M4.5 7c0-1.93 1.57-3.5 3.5-3.5s3.5 1.57 3.5 3.5-1.57 3.5-3.5 3.5S4.5 8.93 4.5 7zm8.5 0c0-1.93 1.57-3.5 3.5-3.5S20 5.07 20 7s-1.57 3.5-3.5 3.5S13 8.93 13 7z"/>
                  <path d="M9 12h6v2H9z"/>
                  <path d="M4.5 17c0-1.93 1.57-3.5 3.5-3.5s3.5 1.57 3.5 3.5-1.57 3.5-3.5 3.5S4.5 18.93 4.5 17zm8.5 0c0-1.93 1.57-3.5 3.5-3.5s3.5 1.57 3.5 3.5-1.57 3.5-3.5 3.5S13 18.93 13 17z"/>
                </svg>
              </div>
              <div>
                <h3 class="text-lg font-bold text-gray-900">${siteName}</h3>
                <p class="text-xs text-gray-500">Green Hydrogen Facility</p>
              </div>
            </div>
            
            <div class="flex items-center space-x-2">
              <div class="text-2xl">${getFeasibilityIcon(feasibility)}</div>
              <div class="text-right">
                <div class="px-3 py-1 bg-${color}-100 text-${color}-700 rounded-full text-xs font-bold">
                  ${(feasibility * 100).toFixed(1)}% viable
                </div>
                <div class="text-xs text-gray-500 mt-1">${tier.toUpperCase()}</div>
              </div>
            </div>
          </div>

          <!-- Main Content Grid - 3 Columns -->
          <div class="grid grid-cols-3 gap-4 mb-4">
            
            <!-- Column 1: Production & Efficiency -->
            <div class="space-y-3">
              <div class="p-3 bg-gradient-to-br from-blue-50 to-blue-100/50 border border-blue-200/70 rounded-lg">
                <div class="flex items-center justify-between mb-2">
                  <div class="p-1.5 bg-blue-500 rounded-md">
                    <svg class="w-3 h-3 text-white" fill="currentColor" viewBox="0 0 24 24">
                      <path d="M12 2L2 7v10c0 5.55 3.84 9.739 9 11 5.16-1.261 9-5.45 9-11V7l-10-5z"/>
                    </svg>
                  </div>
                  <div class="text-right">
                    <div class="text-lg font-bold text-blue-900">${production}</div>
                    <div class="text-xs text-blue-600">kg/day</div>
                  </div>
                </div>
                <div class="text-xs font-semibold text-blue-800">H₂ Production</div>
              </div>
              
              <div class="p-3 bg-gradient-to-br from-purple-50 to-purple-100/50 border border-purple-200/70 rounded-lg">
                <div class="flex items-center justify-between mb-2">
                  <div class="p-1.5 bg-purple-500 rounded-md">
                    <svg class="w-3 h-3 text-white" fill="currentColor" viewBox="0 0 24 24">
                      <path d="M9 12l2 2 4-4m6 2a9 9 0 11-18 0 9 9 0 0118 0z"/>
                    </svg>
                  </div>
                  <div class="text-right">
                    <div class="text-lg font-bold text-purple-900">${efficiency.toFixed(
                      1
                    )}</div>
                    <div class="text-xs text-purple-600">% system</div>
                  </div>
                </div>
                <div class="text-xs font-semibold text-purple-800">Efficiency</div>
              </div>
            </div>

            <!-- Column 2: Renewable Energy -->
            <div class="p-3 bg-gradient-to-br from-gray-50 to-white border border-gray-200/70 rounded-lg">
              <h4 class="text-xs font-bold text-gray-800 mb-3 flex items-center">
                <svg class="w-3 h-3 mr-1 text-indigo-500" fill="currentColor" viewBox="0 0 24 24">
                  <path d="M13 10V3L4 14h7v7l9-11h-7z"/>
                </svg>
                Renewable Sources
              </h4>
              
              <div class="space-y-2">
                <!-- Solar -->
                <div class="flex items-center justify-between p-2 bg-gradient-to-r from-yellow-50 to-amber-50 border border-yellow-200/50 rounded-lg">
                  <div class="flex items-center space-x-2">
                    <div class="w-6 h-6 bg-yellow-500 rounded-md flex items-center justify-center">
                      <svg class="w-3 h-3 text-white" fill="currentColor" viewBox="0 0 24 24">
                        <path d="M12 2.25a.75.75 0 01.75.75v2.25a.75.75 0 01-1.5 0V3a.75.75 0 01.75-.75zM7.5 12a4.5 4.5 0 119 0 4.5 4.5 0 01-9 0z"/>
                      </svg>
                    </div>
                    <div>
                      <div class="text-sm font-bold text-yellow-900">${pvPower.toLocaleString()}</div>
                      <div class="text-xs text-yellow-700">kW Solar</div>
                    </div>
                  </div>
                  <div class="text-xs text-yellow-600">${solarIrradiance.toFixed(
                    1
                  )} kWh/m²</div>
                </div>
                
                <!-- Wind -->
                <div class="flex items-center justify-between p-2 bg-gradient-to-r from-cyan-50 to-sky-50 border border-cyan-200/50 rounded-lg">
                  <div class="flex items-center space-x-2">
                    <div class="w-6 h-6 bg-cyan-500 rounded-md flex items-center justify-center">
                      <svg class="w-3 h-3 text-white" fill="currentColor" viewBox="0 0 24 24">
                        <path d="M6 3v12h4l-1 9 9-9h-4l1-12H6z"/>
                      </svg>
                    </div>
                    <div>
                      <div class="text-sm font-bold text-cyan-900">${windPower.toLocaleString()}</div>
                      <div class="text-xs text-cyan-700">kW Wind</div>
                    </div>
                  </div>
                  <div class="text-xs text-cyan-600">${windSpeed.toFixed(
                    1
                  )} m/s</div>
                </div>
                
                <!-- Total Power -->
                <div class="p-2 bg-gradient-to-r from-indigo-100 to-purple-100 border border-indigo-200 rounded-lg">
                  <div class="flex items-center justify-between">
                    <span class="text-xs font-semibold text-indigo-800">Total Renewable</span>
                    <span class="text-sm font-bold text-indigo-900">${totalPower.toLocaleString()} kW</span>
                  </div>
                </div>
              </div>
            </div>

            <!-- Column 3: Advanced Metrics -->
            <div class="space-y-3">
              <!-- Electrolyzer efficiency -->
              <div class="p-3 bg-gradient-to-br from-teal-50 to-emerald-50 border border-teal-200/70 rounded-lg">
                <div class="flex items-center justify-between mb-2">
                  <div class="p-1.5 bg-teal-500 rounded-md">
                    <svg class="w-3 h-3 text-white" fill="currentColor" viewBox="0 0 24 24">
                      <path d="M4.5 7c0-1.93 1.57-3.5 3.5-3.5s3.5 1.57 3.5 3.5-1.57 3.5-3.5 3.5S4.5 8.93 4.5 7zm8.5 0c0-1.93 1.57-3.5 3.5-3.5S20 5.07 20 7s-1.57 3.5-3.5 3.5S13 8.93 13 7z"/>
                    </svg>
                  </div>
                  <div class="text-right">
                    <div class="text-lg font-bold text-teal-900">${electrolyzerEff.toFixed(
                      1
                    )}</div>
                    <div class="text-xs text-teal-600">% electrolyzer</div>
                  </div>
                </div>
                <div class="w-full bg-teal-200 rounded-full h-1.5">
                  <div class="bg-gradient-to-r from-teal-400 to-teal-600 h-1.5 rounded-full transition-all duration-1000" style="width: ${electrolyzerEff}%"></div>
                </div>
              </div>

              <!-- Environmental conditions -->
              <div class="p-3 bg-gradient-to-br from-orange-50 to-red-50 border border-orange-200/70 rounded-lg">
                <div class="flex items-center justify-between mb-2">
                  <div class="p-1.5 bg-orange-500 rounded-md">
                    <svg class="w-3 h-3 text-white" fill="currentColor" viewBox="0 0 24 24">
                      <path d="M12 2v2m6.364.636l-1.414 1.414M21 12h-2M18.364 18.364l-1.414-1.414M12 21v-2m-6.364-.636l1.414-1.414M3 12h2m3.636-6.364l1.414 1.414"/>
                    </svg>
                  </div>
                  <div class="text-right">
                    <div class="text-lg font-bold text-orange-900">${temperature.toFixed(
                      1
                    )}</div>
                    <div class="text-xs text-orange-600">°C temp</div>
                  </div>
                </div>
                <div class="text-xs font-semibold text-orange-800">Environment</div>
              </div>
            </div>
          </div>

          <!-- Bottom Status Bar -->
          <div class="flex items-center justify-between pt-3 border-t border-gray-200/70">
            <!-- Location -->
            <div class="flex items-center text-gray-500 text-xs">
              <svg class="w-3 h-3 mr-1" fill="currentColor" viewBox="0 0 24 24">
                <path d="M12 2C8.13 2 5 5.13 5 9c0 5.25 7 13 7 13s7-7.75 7-13c0-3.87-3.13-7-7-7zm0 9.5c-1.38 0-2.5-1.12-2.5-2.5s1.12-2.5 2.5-2.5 2.5 1.12 2.5 2.5-1.12 2.5-2.5 2.5z"/>
              </svg>
              ${latitude.toFixed(3)}°, ${longitude.toFixed(3)}°
            </div>
            
            <!-- Status indicators -->
            <div class="flex items-center space-x-4">
              <!-- Performance badges -->
              <div class="flex items-center space-x-1">
                <span class="text-xs ${productionStatus} px-2 py-1 rounded-full font-medium">
                  ${
                    production > 100
                      ? "High Output"
                      : production > 50
                      ? "Good Output"
                      : "Low Output"
                  }
                </span>
                <span class="text-xs ${efficiencyStatus} px-2 py-1 rounded-full font-medium">
                  ${
                    efficiency > 85
                      ? "Excellent Eff"
                      : efficiency > 75
                      ? "Good Eff"
                      : "Fair Eff"
                  }
                </span>
              </div>
              
              <!-- Online status -->
              <div class="flex items-center space-x-1">
                <div class="w-2 h-2 bg-green-500 rounded-full animate-pulse"></div>
                <span class="text-xs text-green-700 font-semibold">Online</span>
              </div>
            </div>
          </div>
        </div>

        <!-- Subtle animated background pattern -->
        <div class="absolute inset-0 opacity-5">
          <div class="absolute inset-0 bg-gradient-to-br from-${color}-500/20 via-transparent to-${color}-500/20 animate-pulse"></div>
        </div>
      </div>

      <style>
        .enhanced-hydrogen-marker {
          filter: drop-shadow(0 4px 6px rgba(0, 0, 0, 0.1));
        }
        
        .leaflet-popup-content-wrapper {
          background: transparent !important;
          box-shadow: 0 25px 50px -12px rgba(0, 0, 0, 0.25) !important;
          border-radius: 1rem !important;
          border: 1px solid rgba(255, 255, 255, 0.2) !important;
          padding: 0 !important;
        }
        
        .leaflet-popup-content {
          margin: 0 !important;
          border-radius: 1rem !important;
          width: 500px !important;
        }
        
        .leaflet-popup-tip {
          background: rgba(255, 255, 255, 0.9) !important;
          border: 1px solid rgba(255, 255, 255, 0.2) !important;
          box-shadow: 0 4px 6px -1px rgba(0, 0, 0, 0.1) !important;
        }
      </style>
    `;

      marker.bindPopup(popupContent, {
        maxWidth: 520,
        className: "enhanced-hydrogen-popup",
        closeButton: true,
        autoPan: true,
      });

      markersRef.current.push(marker);
    });
  };

  const switchLayer = (layerType) => {
    if (!mapInstanceRef.current) return;

    setCurrentLayer(layerType);

    mapInstanceRef.current.eachLayer((layer) => {
      if (layer instanceof window.L.TileLayer) {
        mapInstanceRef.current.removeLayer(layer);
      }
    });

    window.L.tileLayer(mapLayers[layerType].url, {
      attribution: mapLayers[layerType].attribution,
      maxZoom: 19,
    }).addTo(mapInstanceRef.current);
  };

  const handleSearch = async (query) => {
    if (!query.trim()) {
      setSearchResults([]);
      setShowResults(false);
      return;
    }

    setIsSearching(true);

    try {
      const response = await fetch(
        `https://nominatim.openstreetmap.org/search?format=json&q=${encodeURIComponent(
          query
        )}&limit=5&countrycodes=in`
      );
      const data = await response.json();

      const formattedResults = data.map((item) => ({
        id: item.place_id,
        name: item.display_name,
        lat: parseFloat(item.lat),
        lng: parseFloat(item.lon),
        type: item.type,
      }));

      setSearchResults(formattedResults);
      setShowResults(true);
    } catch (error) {
      console.error("Search error:", error);
      setSearchResults([]);
    } finally {
      setIsSearching(false);
    }
  };

  const selectSearchResult = (result) => {
    if (!mapInstanceRef.current) return;

    mapInstanceRef.current.setView([result.lat, result.lng], 12);

    // Extract city name from the search result
    const cityName = result.name.split(",")[0].trim();
    setSelectedCity(cityName);

    const tempIcon = window.L.divIcon({
      html: `
        <div class="w-8 h-8 bg-red-500 rounded-full border-3 border-white shadow-xl animate-bounce flex items-center justify-center">
          <svg class="w-4 h-4 text-white" fill="currentColor" viewBox="0 0 20 20">
            <path d="M5.05 4.05a7 7 0 119.9 9.9L10 18.9l-4.95-4.95a7 7 0 010-9.9zM10 11a2 2 0 100-4 2 2 0 000 4z"/>
          </svg>
        </div>
      `,
      className: "search-result-marker",
      iconSize: [32, 32],
      iconAnchor: [16, 16],
    });

    const tempMarker = window.L.marker([result.lat, result.lng], {
      icon: tempIcon,
    }).addTo(mapInstanceRef.current);

    setTimeout(() => {
      mapInstanceRef.current.removeLayer(tempMarker);
    }, 4000);

    setShowResults(false);
    setSearchQuery("");
  };

  const generateAIRecommendation = async () => {
    if (!selectedCity.trim()) {
      alert(
        "Please select a city first by searching and clicking on a location"
      );
      return;
    }

    if (!mapInstanceRef.current) return;

    setIsGenerating(true);

    try {
      // Build the API URL with city and cap parameters
      const apiUrl = new URL(
        "http://localhost:8080/ml/sites/city"
      );
      apiUrl.searchParams.append("name", selectedCity);

      // Add cap parameter - send null if no cap value
      if (cap !== null && cap !== undefined && cap !== "") {
        apiUrl.searchParams.append("cap", cap);
      } else {
        apiUrl.searchParams.append("cap", "null");
      }

      console.log("API URL:", apiUrl.toString());
      console.log("Requesting for city:", selectedCity);
      console.log("Cap parameter:", cap || "null");

      const response = await fetch(apiUrl.toString(), {
        method: "GET",
        headers: {
          "Content-Type": "application/json",
        },
      });
      const recommendation = await response.json();

      // DEBUG: Log the entire response to see what we're getting
      console.log("Full API Response:", recommendation);
      console.log("Response type:", typeof recommendation);
      console.log("Response keys:", Object.keys(recommendation));

      // Check if recommendation is an array and get the first item
      const data = Array.isArray(recommendation)
        ? recommendation[0]
        : recommendation;

      console.log("Data after array check:", data);

      // More comprehensive coordinate extraction
      const lat = Number(
        data?.lat ||
          data?.latitude ||
          data?.Latitude ||
          data?.LAT ||
          data?.coordinates?.lat ||
          data?.coordinates?.latitude ||
          data?.geometry?.coordinates?.[1] || // GeoJSON format [lng, lat]
          data?.location?.lat ||
          data?.position?.lat
      );

      const lng = Number(
        data?.lng ||
          data?.longitude ||
          data?.Longitude ||
          data?.LNG ||
          data?.LON ||
          data?.lon ||
          data?.coordinates?.lng ||
          data?.coordinates?.longitude ||
          data?.geometry?.coordinates?.[0] || // GeoJSON format [lng, lat]
          data?.location?.lng ||
          data?.location?.longitude ||
          data?.position?.lng ||
          data?.position?.longitude
      );

      console.log("Extracted coordinates:", { lat, lng });
      console.log("Lat type:", typeof lat, "Lng type:", typeof lng);
      console.log("Lat isNaN:", isNaN(lat), "Lng isNaN:", isNaN(lng));

      if (isNaN(lat) || isNaN(lng)) {
        console.error(
          "Coordinate extraction failed. Available fields in data:",
          Object.keys(data || {})
        );
        throw new Error(
          `Invalid coordinates in AI recommendation. Available fields: ${Object.keys(
            data || {}
          ).join(", ")}`
        );
      }

      // Validate coordinate ranges
      if (lat < -90 || lat > 90 || lng < -180 || lng > 180) {
        throw new Error(
          `Coordinates out of valid range. Lat: ${lat}, Lng: ${lng}`
        );
      }

      mapInstanceRef.current.setView([lat, lng], 14);

      const aiIcon = window.L.divIcon({
        html: `
        <div class="relative">
          <div class="w-12 h-12 bg-gradient-to-br from-purple-500 via-pink-500 to-rose-500 rounded-full border-4 border-white shadow-2xl flex items-center justify-center transform animate-pulse">
            <div class="w-8 h-8 bg-white/30 backdrop-blur-sm rounded-full flex items-center justify-center">
              <svg class="w-5 h-5 text-white" fill="currentColor" viewBox="0 0 20 20">
                <path d="M9.049 2.927c.3-.921 1.603-.921 1.902 0l1.07 3.292a1 1 0 00.95.69h3.462c.969 0 1.371 1.24.588 1.81l-2.8 2.034a1 1 0 00-.364 1.118l1.07 3.292c.3.921-.755 1.688-1.54 1.118l-2.8-2.034a1 1 0 00-1.175 0l-2.8 2.034c-.784.57-1.838-.197-1.539-1.118l1.07-3.292a1 1 0 00-.364-1.118L2.98 8.72c-.783-.57-.38-1.81.588-1.81h3.461a1 1 0 00.951-.69l1.07-3.292z"/>
              </svg>
            </div>
          </div>
          <div class="absolute -bottom-1 left-1/2 transform -translate-x-1/2 w-4 h-4 bg-gradient-to-br from-purple-500 to-rose-500 rotate-45 border-2 border-white"></div>
          <div class="absolute top-0 left-0 w-full h-full rounded-full border-4 border-purple-300 animate-ping opacity-60"></div>
        </div>
      `,
        className: "ai-recommendation-icon",
        iconSize: [48, 48],
        iconAnchor: [24, 48],
      });

      const aiMarker = window.L.marker([lat, lng], {
        icon: aiIcon,
      }).addTo(mapInstanceRef.current);

      const aiPopupContent = `
      <div class="relative overflow-hidden">
        <!-- Glassmorphism background -->
        <div class="absolute inset-0 bg-gradient-to-br from-white/90 via-white/80 to-white/70 backdrop-blur-xl"></div>
        
        <!-- Content container - Wide Layout -->
        <div class="relative z-10 p-4 w-[600px]">
          
          <!-- Header Section - Compact -->
          <div class="flex items-center justify-between mb-4">
            <div class="flex items-center space-x-3">
              <div class="w-10 h-10 bg-gradient-to-br from-purple-400 to-pink-600 rounded-lg flex items-center justify-center shadow-lg">
                <svg class="w-5 h-5 text-white" fill="currentColor" viewBox="0 0 20 20">
                  <path d="M9.049 2.927c.3-.921 1.603-.921 1.902 0l1.07 3.292a1 1 0 00.95.69h3.462c.969 0 1.371 1.24.588 1.81l-2.8 2.034a1 1 0 00-.364 1.118l1.07 3.292c.3.921-.755 1.688-1.54 1.118l-2.8-2.034a1 1 0 00-1.175 0l-2.8 2.034c-.784.57-1.838-.197-1.539-1.118l1.07-3.292a1 1 0 00-.364-1.118L2.98 8.72c-.783-.57-.38-1.81.588-1.81h3.461a1 1 0 00.951-.69l1.07-3.292z"/>
                </svg>
              </div>
              <div>
                <h3 class="text-lg font-bold text-gray-900">${
                  data?.name ||
                  data?.location ||
                  data?.City ||
                  selectedCity ||
                  "AI Optimal Site"
                }</h3>
                <p class="text-xs text-gray-500">AI Generated Recommendation (${
                  cap || "50"
                }cr Investment)</p>
              </div>
            </div>
            
            <div class="flex items-center space-x-2">
              <div class="text-2xl">🤖</div>
              <div class="text-right">
                <div class="px-3 py-1 bg-purple-100 text-purple-700 rounded-full text-xs font-bold">
                  AI Score: ${
                    data?.Feasibility_Score
                      ? (data.Feasibility_Score * 10).toFixed(1)
                      : "9.5"
                  }/10
                </div>
                <div class="text-xs text-gray-500 mt-1">AI OPTIMIZED</div>
              </div>
            </div>
          </div>

          <!-- AI Recommendation Metrics Row -->
          <div class="grid grid-cols-2 gap-4 mb-4">
            <div class="bg-gradient-to-r from-emerald-50 to-green-50 rounded-lg p-3 border border-emerald-200">
              <div class="flex items-center space-x-3">
                <div class="w-8 h-8 bg-emerald-500 rounded-md flex items-center justify-center">
                  <svg class="w-4 h-4 text-white" fill="currentColor" viewBox="0 0 24 24">
                    <path d="M12 2C6.48 2 2 6.48 2 12s4.48 10 10 10 10-4.48 10-10S17.52 2 12 2zm-2 15l-5-5 1.41-1.41L10 14.17l7.59-7.59L19 8l-9 9z"/>
                  </svg>
                </div>
                <div>
                  <p class="text-xs text-gray-600">LCOH</p>
                  <p class="text-lg font-bold text-gray-900">
                    ₹${data?.lcoh?.toFixed(2) || "4.50"}/kg
                  </p>
                  <p class="text-xs text-emerald-600 font-medium">
                    Levelized Cost of Hydrogen
                  </p>
                </div>
              </div>
            </div>
            
            <div class="bg-gradient-to-r from-blue-50 to-indigo-50 rounded-lg p-3 border border-blue-200">
              <div class="flex items-center space-x-3">
                <div class="w-8 h-8 bg-blue-500 rounded-md flex items-center justify-center">
                  <svg class="w-4 h-4 text-white" fill="currentColor" viewBox="0 0 24 24">
                    <path d="M7 14l5-5 5 5z"/>
                  </svg>
                </div>
                <div>
                  <p class="text-xs text-gray-600">ROI</p>
                  <p class="text-lg font-bold text-gray-900">
                    ${data?.roi?.toFixed(2) || "12.5"}%
                  </p>
                  <p class="text-xs text-blue-600 font-medium">
                    Return on Investment
                  </p>
                </div>
              </div>
            </div>
          </div>

          <!-- Main Content Grid - 4 Columns Wide Layout -->
          <div class="grid grid-cols-4 gap-3 mb-4">
            
            <!-- Column 1: Production Capacity -->
            <div class="space-y-2">
              <div class="p-3 bg-gradient-to-br from-blue-50 to-blue-100/50 border border-blue-200/70 rounded-lg">
                <div class="flex items-center justify-between mb-1">
                  <div class="p-1 bg-blue-500 rounded">
                    <svg class="w-3 h-3 text-white" fill="currentColor" viewBox="0 0 24 24">
                      <path d="M12 2L2 7v10c0 5.55 3.84 9.739 9 11 5.16-1.261 9-5.45 9-11V7l-10-5z"/>
                    </svg>
                  </div>
                  <div class="text-right">
                    <div class="text-sm font-bold text-blue-900">${
                      data?.["Hydrogen_Production_kg/day"] || "150"
                    }</div>
                    <div class="text-xs text-blue-600">kg/day</div>
                  </div>
                </div>
                <div class="text-xs font-semibold text-blue-800">H₂ Production</div>
              </div>
              
              <div class="p-3 bg-gradient-to-br from-purple-50 to-purple-100/50 border border-purple-200/70 rounded-lg">
                <div class="flex items-center justify-between mb-1">
                  <div class="p-1 bg-purple-500 rounded">
                    <svg class="w-3 h-3 text-white" fill="currentColor" viewBox="0 0 24 24">
                      <path d="M9 12l2 2 4-4m6 2a9 9 0 11-18 0 9 9 0 0118 0z"/>
                    </svg>
                  </div>
                  <div class="text-right">
                    <div class="text-sm font-bold text-purple-900">${
                      data?.["System_Efficiency_%"]?.toFixed(1) || "78.5"
                    }</div>
                    <div class="text-xs text-purple-600">% system</div>
                  </div>
                </div>
                <div class="text-xs font-semibold text-purple-800">Efficiency</div>
              </div>
            </div>

            <!-- Column 2: Environmental Conditions -->
            <div class="space-y-2">
              <div class="p-3 bg-gradient-to-br from-yellow-50 to-amber-50 border border-yellow-200/70 rounded-lg">
                <div class="flex items-center justify-between mb-1">
                  <div class="p-1 bg-yellow-500 rounded">
                    <svg class="w-3 h-3 text-white" fill="currentColor" viewBox="0 0 24 24">
                      <path d="M12 2.25a.75.75 0 01.75.75v2.25a.75.75 0 01-1.5 0V3a.75.75 0 01.75-.75zM7.5 12a4.5 4.5 0 119 0 4.5 4.5 0 01-9 0z"/>
                    </svg>
                  </div>
                  <div class="text-right">
                    <div class="text-sm font-bold text-yellow-900">${
                      data?.["Solar_Irradiance_kWh/m²/day"]?.toFixed(1) || "6.2"
                    }</div>
                    <div class="text-xs text-yellow-600">kWh/m²</div>
                  </div>
                </div>
                <div class="text-xs font-semibold text-yellow-800">Solar Irradiance</div>
              </div>

              <div class="p-3 bg-gradient-to-br from-cyan-50 to-sky-50 border border-cyan-200/70 rounded-lg">
                <div class="flex items-center justify-between mb-1">
                  <div class="p-1 bg-cyan-500 rounded">
                    <svg class="w-3 h-3 text-white" fill="currentColor" viewBox="0 0 24 24">
                      <path d="M6 3v12h4l-1 9 9-9h-4l1-12H6z"/>
                    </svg>
                  </div>
                  <div class="text-right">
                    <div class="text-sm font-bold text-cyan-900">${
                      data?.["Wind_Speed_m/s"]?.toFixed(1) || "7.8"
                    }</div>
                    <div class="text-xs text-cyan-600">m/s</div>
                  </div>
                </div>
                <div class="text-xs font-semibold text-cyan-800">Wind Speed</div>
              </div>
            </div>

            <!-- Column 3: Power Generation -->
            <div class="space-y-2">
              <div class="p-3 bg-gradient-to-br from-orange-50 to-red-50 border border-orange-200/70 rounded-lg">
                <div class="flex items-center justify-between mb-1">
                  <div class="p-1 bg-orange-500 rounded">
                    <svg class="w-3 h-3 text-white" fill="currentColor" viewBox="0 0 24 24">
                      <path d="M13 10V3L4 14h7v7l9-11h-7z"/>
                    </svg>
                  </div>
                  <div class="text-right">
                    <div class="text-sm font-bold text-orange-900">${
                      data?.PV_Power_kW?.toLocaleString() || "2,500"
                    }</div>
                    <div class="text-xs text-orange-600">kW PV</div>
                  </div>
                </div>
                <div class="text-xs font-semibold text-orange-800">Solar Power</div>
              </div>

              <div class="p-3 bg-gradient-to-br from-teal-50 to-emerald-50 border border-teal-200/70 rounded-lg">
                <div class="flex items-center justify-between mb-1">
                  <div class="p-1 bg-teal-500 rounded">
                    <svg class="w-3 h-3 text-white" fill="currentColor" viewBox="0 0 24 24">
                      <path d="M6 3v12h4l-1 9 9-9h-4l1-12H6z"/>
                    </svg>
                  </div>
                  <div class="text-right">
                    <div class="text-sm font-bold text-teal-900">${
                      data?.Wind_Power_kW?.toLocaleString() || "1,200"
                    }</div>
                    <div class="text-xs text-teal-600">kW Wind</div>
                  </div>
                </div>
                <div class="text-xs font-semibold text-teal-800">Wind Power</div>
              </div>
            </div>

            <!-- Column 4: Advanced Metrics -->
            <div class="space-y-2">
              <div class="p-3 bg-gradient-to-br from-indigo-50 to-blue-50 border border-indigo-200/70 rounded-lg">
                <div class="flex items-center justify-between mb-1">
                  <div class="p-1 bg-indigo-500 rounded">
                    <svg class="w-3 h-3 text-white" fill="currentColor" viewBox="0 0 24 24">
                      <path d="M12 2v2m6.364.636l-1.414 1.414M21 12h-2M18.364 18.364l-1.414-1.414M12 21v-2m-6.364-.636l1.414-1.414M3 12h2m3.636-6.364l1.414 1.414"/>
                    </svg>
                  </div>
                  <div class="text-right">
                    <div class="text-sm font-bold text-indigo-900">${
                      data?.["Temperature_C"]?.toFixed(1) || "28.5"
                    }</div>
                    <div class="text-xs text-indigo-600">°C temp</div>
                  </div>
                </div>
                <div class="text-xs font-semibold text-indigo-800">Temperature</div>
              </div>

              <div class="p-3 bg-gradient-to-br from-purple-50 to-pink-50 border border-purple-200/70 rounded-lg">
                <div class="flex items-center justify-between mb-1">
                  <div class="p-1 bg-purple-500 rounded">
                    <svg class="w-3 h-3 text-white" fill="currentColor" viewBox="0 0 20 20">
                      <path d="M9 12l2 2 4-4m6 2a9 9 0 11-18 0 9 9 0 0118 0z"/>
                    </svg>
                  </div>
                  <div class="text-right">
                    <div class="text-sm font-bold text-purple-900">${
                      data?.["Electrolyzer_Efficiency_%"]?.toFixed(1) || "85.2"
                    }</div>
                    <div class="text-xs text-purple-600">% electro</div>
                  </div>
                </div>
                <div class="text-xs font-semibold text-purple-800">Electrolyzer</div>
              </div>
            </div>
          </div>

          <!-- Additional Systems Row -->
          <div class="grid grid-cols-3 gap-3 mb-4">
            <div class="flex items-center justify-between p-2 bg-gradient-to-r from-cyan-50 to-teal-50 border border-cyan-200/50 rounded-lg">
              <div class="flex items-center space-x-2">
                <div class="w-6 h-6 bg-cyan-500 rounded flex items-center justify-center">
                  <svg class="w-3 h-3 text-white" fill="currentColor" viewBox="0 0 24 24">
                    <path d="M3 17h18v2H3zm3.24-8.24L12 4l5.76 4.76L16.5 10 12 6.5 7.5 10z"/>
                  </svg>
                </div>
                <div>
                  <div class="text-sm font-bold text-cyan-900">Desalination</div>
                  <div class="text-xs text-cyan-700">${
                    data?.Desalination_Power_kW || "150"
                  } kW</div>
                </div>
              </div>
            </div>
            
            <div class="flex items-center justify-between p-2 bg-gradient-to-r from-emerald-50 to-green-50 border border-emerald-200/50 rounded-lg">
              <div class="flex items-center space-x-2">
                <div class="w-6 h-6 bg-emerald-500 rounded flex items-center justify-center">
                  <svg class="w-3 h-3 text-white" fill="currentColor" viewBox="0 0 20 20">
                    <path d="M9 12l2 2 4-4m6 2a9 9 0 11-18 0 9 9 0 0118 0z"/>
                  </svg>
                </div>
                <div>
                  <div class="text-sm font-bold text-emerald-900">Feasibility</div>
                  <div class="text-xs text-emerald-700">${
                    data?.Feasibility_Score
                      ? (data.Feasibility_Score * 100).toFixed(1)
                      : "85.5"
                  }% Score</div>
                </div>
              </div>
            </div>
            
            <div class="flex items-center justify-between p-2 bg-gradient-to-r from-purple-50 to-pink-50 border border-purple-200/50 rounded-lg">
              <div class="flex items-center space-x-2">
                <div class="w-6 h-6 bg-purple-500 rounded flex items-center justify-center">
                  <svg class="w-3 h-3 text-white" fill="currentColor" viewBox="0 0 20 20">
                    <path d="M9.049 2.927c.3-.921 1.603-.921 1.902 0l1.07 3.292a1 1 0 00.95.69h3.462c.969 0 1.371 1.24.588 1.81l-2.8 2.034a1 1 0 00-.364 1.118l1.07 3.292c.3.921-.755 1.688-1.54 1.118l-2.8-2.034a1 1 0 00-1.175 0l-2.8 2.034c-.784.57-1.838-.197-1.539-1.118l1.07-3.292a1 1 0 00-.364-1.118L2.98 8.72c-.783-.57-.38-1.81.588-1.81h3.461a1 1 0 00.951-.69l1.07-3.292z"/>
                  </svg>
                </div>
                <div>
                  <div class="text-sm font-bold text-purple-900">AI Confidence</div>
                  <div class="text-xs text-purple-700">${
                    data?.confidence ? (data.confidence * 100).toFixed(0) : "95"
                  }% Smart</div>
                </div>
              </div>
            </div>
          </div>

          <!-- Bottom Status Bar -->
          <div class="flex items-center justify-between pt-3 border-t border-gray-200/70">
            <!-- Location -->
            <div class="flex items-center text-gray-500 text-xs">
              <svg class="w-3 h-3 mr-1" fill="currentColor" viewBox="0 0 24 24">
                <path d="M12 2C8.13 2 5 5.13 5 9c0 5.25 7 13 7 13s7-7.75 7-13c0-3.87-3.13-7-7-7zm0 9.5c-1.38 0-2.5-1.12-2.5-2.5s1.12-2.5 2.5-2.5 2.5 1.12 2.5 2.5-1.12 2.5-2.5 2.5z"/>
              </svg>
              ${lat.toFixed(3)}°, ${lng.toFixed(3)}°
            </div>
            
            <!-- Status indicators -->
            <div class="flex items-center space-x-4">
              <!-- Performance badges -->
              <div class="flex items-center space-x-1">
                <span class="text-xs bg-emerald-100 text-emerald-700 px-2 py-1 rounded-full font-medium">
                  Based on 12+ factors
                </span>
                <span class="text-xs bg-purple-100 text-purple-700 px-2 py-1 rounded-full font-medium">
                  50cr Investment
                </span>
              </div>
              
              <!-- AI Online status -->
              <div class="flex items-center space-x-1">
                <div class="w-2 h-2 bg-purple-500 rounded-full animate-pulse"></div>
                <span class="text-xs text-purple-700 font-semibold">AI Optimized</span>
              </div>
            </div>
          </div>
        </div>

        <!-- Subtle animated background pattern -->
        <div class="absolute inset-0 opacity-5">
          <div class="absolute inset-0 bg-gradient-to-br from-purple-500/20 via-transparent to-pink-500/20 animate-pulse"></div>
        </div>
      </div>

      <style>
        .enhanced-hydrogen-marker {
          filter: drop-shadow(0 4px 6px rgba(0, 0, 0, 0.1));
        }
        
        .leaflet-popup-content-wrapper {
          background: transparent !important;
          box-shadow: 0 25px 50px -12px rgba(0, 0, 0, 0.25) !important;
          border-radius: 1rem !important;
          border: 1px solid rgba(255, 255, 255, 0.2) !important;
          padding: 0 !important;
        }
        
        .leaflet-popup-content {
          margin: 0 !important;
          border-radius: 1rem !important;
          width: 600px !important;
        }
        
        .leaflet-popup-tip {
          background: rgba(255, 255, 255, 0.9) !important;
          border: 1px solid rgba(255, 255, 255, 0.2) !important;
          box-shadow: 0 4px 6px -1px rgba(0, 0, 0, 0.1) !important;
        }
      </style>
    `;
      aiMarker
        .bindPopup(aiPopupContent, {
          maxWidth: 400,
          className: "ai-recommendation-popup",
        })
        .openPopup();

      setSelectedCity("");
    } catch (error) {
      console.error("AI generation error:", error);
      alert(`Failed to generate AI recommendation: ${error.message}`);
    } finally {
      setIsGenerating(false);
    }
  };
  const getCurrentLocation = () => {
    if (!navigator.geolocation) {
      alert("Geolocation is not supported by this browser.");
      return;
    }

    navigator.geolocation.getCurrentPosition(
      (position) => {
        const { latitude, longitude } = position.coords;
        if (mapInstanceRef.current) {
          mapInstanceRef.current.setView([latitude, longitude], 15);

          const userIcon = window.L.divIcon({
            html: `
              <div class="relative">
                <div class="w-6 h-6 bg-blue-500 rounded-full border-3 border-white shadow-lg flex items-center justify-center">
                  <div class="w-3 h-3 bg-white rounded-full"></div>
                </div>
                <div class="absolute inset-0 bg-blue-300 rounded-full animate-ping opacity-60"></div>
                <div class="absolute inset-0 bg-blue-400 rounded-full animate-ping opacity-40" style="animation-delay: 0.5s"></div>
              </div>
            `,
            className: "user-location-marker",
            iconSize: [24, 24],
            iconAnchor: [12, 12],
          });

          window.L.marker([latitude, longitude], { icon: userIcon })
            .addTo(mapInstanceRef.current)
            .bindPopup(
              `
              <div class="p-3 text-center">
                <div class="w-8 h-8 bg-blue-500 rounded-full mx-auto mb-2 flex items-center justify-center">
                  <svg class="w-4 h-4 text-white" fill="currentColor" viewBox="0 0 24 24">
                    <path d="M12 2v2m6.364.636l-1.414 1.414M21 12h-2M18.364 18.364l-1.414-1.414M12 21v-2m-6.364-.636l1.414-1.414M3 12h2m3.636-6.364l1.414 1.414"/>
                  </svg>
                </div>
                <h3 class="font-semibold text-gray-900">Your Current Location</h3>
                <p class="text-sm text-gray-600 mt-1">${latitude.toFixed(
                  4
                )}°, ${longitude.toFixed(4)}°</p>
              </div>
            `
            )
            .openPopup();
        }
      },
      (error) => {
        console.error("Error getting location:", error);
        alert("Unable to retrieve your location.");
      }
    );
  };

  const refreshData = async () => {
    try {
      setLoading(true);
      setError(null);
      const response = await fetch(
        "http://localhost:8080/ml/sites/top"
      );
      if (!response.ok) {
        throw new Error(`Failed to fetch sites data: ${response.status}`);
      }
      const data = await response.json();
      setBackendSites(data);
    } catch (err) {
      setError(err.message);
      console.error("Error refreshing backend sites:", err);
    } finally {
      setLoading(false);
    }
  };

  // Calculate total production from backend sites
  const totalProduction = backendSites.reduce((total, site) => {
    const production = site["Hydrogen_Production_kg/day"] || 0;
    return total + (Number(production) || 0);
  }, 0);

  // Calculate average feasibility from backend sites
  const avgFeasibility =
    backendSites.length > 0
      ? (backendSites.reduce((total, site) => {
          const feasibility = site.Feasibility_Score || 0;
          return total + (Number(feasibility) || 0);
        }, 0) /
          backendSites.length) *
        100
      : 0;

  return (
    <div className="relative h-screen w-full bg-gray-100">
      {/* Header with Search and AI Features */}
      <div className="absolute top-0 left-0 right-0 z-[1000] bg-white/95 backdrop-blur-sm shadow-lg border-b border-gray-200">
        <div className="px-6 py-4">
          {/* Top Row - Title and Stats */}
          <div className="flex items-center justify-between mb-4">
            <div className="flex items-center space-x-3">
              <div className="p-3 bg-gradient-to-br from-emerald-500 to-green-600 rounded-xl shadow-lg">
                <Zap className="h-7 w-7 text-white" />
              </div>
              <div>
                <h1 className="text-2xl font-bold text-gray-900 bg-gradient-to-r from-emerald-600 to-green-600 bg-clip-text text-transparent">
                  Green Hydrogen Infrastructure Map
                </h1>
                <p className="text-sm text-gray-600 font-medium">
                  AI-Optimized Site Locations & Analytics
                  {loading && (
                    <span className="ml-2 text-emerald-600 animate-pulse">
                      (Syncing...)
                    </span>
                  )}
                </p>
              </div>
            </div>

            {/* Enhanced Stats Dashboard */}
            <div className="hidden lg:flex items-center space-x-6">
              <div className="bg-gradient-to-br from-emerald-50 to-green-50 border border-emerald-200 rounded-xl p-4 text-center min-w-20">
                <div className="text-2xl font-bold text-emerald-600">
                  {loading ? "..." : backendSites.length}
                </div>
                <div className="text-xs text-emerald-700 font-medium">
                  Active Sites
                </div>
              </div>
              <div className="bg-gradient-to-br from-blue-50 to-cyan-50 border border-blue-200 rounded-xl p-4 text-center min-w-24">
                <div className="text-2xl font-bold text-blue-600">
                  {loading ? "..." : `${totalProduction.toFixed(0)}`}
                </div>
                <div className="text-xs text-blue-700 font-medium">
                  kg/day H2
                </div>
              </div>
              <div className="bg-gradient-to-br from-purple-50 to-pink-50 border border-purple-200 rounded-xl p-4 text-center min-w-20">
                <div className="text-2xl font-bold text-purple-600">
                  {loading ? "..." : `${avgFeasibility.toFixed(1)}%`}
                </div>
                <div className="text-xs text-purple-700 font-medium">
                  Feasibility
                </div>
              </div>
              <button
                onClick={refreshData}
                disabled={loading}
                className="p-3 bg-gradient-to-br from-gray-50 to-gray-100 border border-gray-200 rounded-xl shadow-sm hover:shadow-md transition-all duration-200 disabled:opacity-50 group"
                title="Refresh data"
              >
                <RefreshCw
                  className={`h-5 w-5 text-gray-600 group-hover:text-emerald-600 transition-colors ${
                    loading ? "animate-spin" : ""
                  }`}
                />
              </button>
            </div>
          </div>

          {/* Bottom Row - Enhanced Search and AI Controls */}
          <div className="flex items-center space-x-4">
            {/* Enhanced Location Search */}
            <div className="relative flex-1 max-w-lg">
              <div className="relative">
                <Search className="absolute left-4 top-1/2 transform -translate-y-1/2 h-5 w-5 text-gray-400" />
                <input
                  type="text"
                  placeholder="Search for locations to explore potential hydrogen sites..."
                  value={searchQuery}
                  onChange={(e) => {
                    setSearchQuery(e.target.value);
                    handleSearch(e.target.value);
                  }}
                  className="w-full pl-12 pr-4 py-3 bg-gray-50 border border-gray-200 rounded-xl focus:ring-2 focus:ring-emerald-500 focus:border-emerald-500 outline-none text-sm shadow-sm transition-all duration-200"
                />
                {isSearching && (
                  <div className="absolute right-4 top-1/2 transform -translate-y-1/2">
                    <div className="animate-spin rounded-full h-5 w-5 border-b-2 border-emerald-500"></div>
                  </div>
                )}
              </div>

              {/* Enhanced Search Results */}
              {showResults && searchResults.length > 0 && (
                <div className="absolute top-full mt-2 w-full bg-white border border-gray-200 rounded-xl shadow-xl max-h-72 overflow-y-auto">
                  {searchResults.map((result, index) => (
                    <button
                      key={result.id}
                      onClick={() => selectSearchResult(result)}
                      className={`w-full text-left px-5 py-4 hover:bg-gradient-to-r hover:from-emerald-50 hover:to-green-50 transition-all duration-200 ${
                        index !== searchResults.length - 1
                          ? "border-b border-gray-100"
                          : ""
                      }`}
                    >
                      <div className="flex items-center space-x-4">
                        <div className="p-2 bg-emerald-100 rounded-lg">
                          <MapPin className="h-4 w-4 text-emerald-600" />
                        </div>
                        <div className="flex-1 truncate">
                          <div className="font-semibold text-gray-900 truncate">
                            {result.name.split(",")[0]}
                          </div>
                          <div className="text-sm text-gray-500 truncate">
                            {result.name.split(",").slice(1).join(",").trim()}
                          </div>
                        </div>
                        <div className="text-xs text-emerald-600 font-medium">
                          Explore →
                        </div>
                      </div>
                    </button>
                  ))}
                </div>
              )}
            </div>

            {/* City Selection and AI Generation */}
            <div className="flex items-center space-x-3">
              {/* Selected City Indicator */}
              {selectedCity ? (
                <div className="flex items-center space-x-2 px-4 py-3 bg-emerald-50 border border-emerald-200 rounded-lg">
                  <MapPin className="h-4 w-4 text-emerald-600" />
                  <span className="text-sm font-medium text-emerald-800">
                    Selected: {selectedCity}
                  </span>
                  <button
                    onClick={() => setSelectedCity("")}
                    className="text-emerald-600 hover:text-emerald-800 transition-colors ml-2"
                  >
                    ×
                  </button>
                </div>
              ) : (
                <div className="px-4 py-3 bg-gray-50 border border-gray-200 rounded-lg text-sm text-gray-500">
                  Search and select a city first
                </div>
              )}
              {/* Cap Input Field */}
              <div className="flex items-center space-x-2">
                <label className="text-sm font-medium text-gray-700 whitespace-nowrap">
                  Investment Cap:
                </label>
                <input
                  type="number"
                  placeholder="50"
                  value={cap}
                  onChange={(e) => setCap(e.target.value)}
                  className="w-24 px-3 py-2 bg-gray-50 border border-gray-200 rounded-lg focus:ring-2 focus:ring-emerald-500 focus:border-emerald-500 outline-none text-sm"
                />
                <span className="text-sm text-gray-500">cr</span>
              </div>
              <button
                onClick={generateAIRecommendation}
                disabled={isGenerating || !selectedCity.trim()}
                className="flex items-center space-x-3 px-6 py-3 bg-gradient-to-r from-purple-500 via-pink-500 to-rose-500 text-white rounded-xl hover:from-purple-600 hover:via-pink-600 hover:to-rose-600 transition-all duration-300 font-semibold text-sm shadow-lg hover:shadow-xl transform hover:scale-105 disabled:opacity-50 disabled:cursor-not-allowed disabled:transform-none"
              >
                {isGenerating ? (
                  <>
                    <div className="animate-spin rounded-full h-5 w-5 border-b-2 border-white"></div>
                    <span>Generating AI Recommendation...</span>
                  </>
                ) : (
                  <>
                    <svg
                      className="h-5 w-5"
                      fill="currentColor"
                      viewBox="0 0 20 20"
                    >
                      <path d="M9.049 2.927c.3-.921 1.603-.921 1.902 0l1.07 3.292a1 1 0 00.95.69h3.462c.969 0 1.371 1.24.588 1.81l-2.8 2.034a1 1 0 00-.364 1.118l1.07 3.292c.3.921-.755 1.688-1.54 1.118l-2.8-2.034a1 1 0 00-1.175 0l-2.8 2.034c-.784.57-1.838-.197-1.539-1.118l1.07-3.292a1 1 0 00-.364-1.118L2.98 8.72c-.783-.57-.38-1.81.588-1.81h3.461a1 1 0 00.951-.69l1.07-3.292z" />
                    </svg>
                    <span>
                      Generate AI Site
                      {selectedCity ? ` for ${selectedCity}` : ""}
                      {cap ? ` (${cap}cr)` : ""}
                    </span>
                  </>
                )}
              </button>
            </div>
          </div>
        </div>
      </div>

      {/* Error/Loading Overlay */}
      {(loading || error) && (
        <div className="absolute top-36 left-6 right-6 z-[1000]">
          <div className="bg-white/95 backdrop-blur-sm border border-gray-200 rounded-xl shadow-lg p-5">
            {loading && (
              <div className="flex items-center space-x-4 text-emerald-600">
                <div className="animate-spin rounded-full h-6 w-6 border-b-2 border-emerald-500"></div>
                <div>
                  <div className="font-semibold">Syncing with AI Backend</div>
                  <div className="text-sm text-gray-600">
                    Fetching optimized hydrogen site data...
                  </div>
                </div>
              </div>
            )}
            {error && (
              <div className="flex items-center justify-between">
                <div className="flex items-center space-x-4 text-red-600">
                  <AlertCircle className="h-6 w-6" />
                  <div>
                    <div className="font-semibold">
                      Backend Connection Failed
                    </div>
                    <div className="text-sm text-gray-600">{error}</div>
                  </div>
                </div>
                <button
                  onClick={refreshData}
                  className="ml-4 px-4 py-2 bg-red-100 text-red-700 rounded-lg hover:bg-red-200 transition-colors font-medium"
                >
                  Retry Connection
                </button>
              </div>
            )}
          </div>
        </div>
      )}

      {/* Enhanced Layer Controls */}
      <div className="absolute top-36 right-6 z-[1000]">
        <div className="bg-white/95 backdrop-blur-sm border border-gray-200 rounded-xl shadow-lg p-2">
          <div className="flex flex-col space-y-2">
            <button
              onClick={() => switchLayer("street")}
              className={`flex items-center space-x-3 px-4 py-3 rounded-lg text-sm font-semibold transition-all duration-200 ${
                currentLayer === "street"
                  ? "bg-gradient-to-r from-emerald-500 to-green-600 text-white shadow-md"
                  : "text-gray-700 hover:bg-gray-100"
              }`}
            >
              <MapPin className="h-4 w-4" />
              <span>Street View</span>
            </button>
            <button
              onClick={() => switchLayer("satellite")}
              className={`flex items-center space-x-3 px-4 py-3 rounded-lg text-sm font-semibold transition-all duration-200 ${
                currentLayer === "satellite"
                  ? "bg-gradient-to-r from-emerald-500 to-green-600 text-white shadow-md"
                  : "text-gray-700 hover:bg-gray-100"
              }`}
            >
              <Satellite className="h-4 w-4" />
              <span>Satellite</span>
            </button>
            <button
              onClick={() => switchLayer("dark")}
              className={`flex items-center space-x-3 px-4 py-3 rounded-lg text-sm font-semibold transition-all duration-200 ${
                currentLayer === "dark"
                  ? "bg-gradient-to-r from-emerald-500 to-green-600 text-white shadow-md"
                  : "text-gray-700 hover:bg-gray-100"
              }`}
            >
              <Moon className="h-4 w-4" />
              <span>Dark Mode</span>
            </button>
          </div>
        </div>
      </div>

      {/* Enhanced Current Location Button */}
      <div className="absolute bottom-36 right-6 z-[1000]">
        <button
          onClick={getCurrentLocation}
          className="p-4 bg-white/95 backdrop-blur-sm border border-gray-200 rounded-xl shadow-lg hover:shadow-xl hover:bg-blue-50 transition-all duration-200 group"
          title="Find my location"
        >
          <Navigation className="h-6 w-6 text-gray-700 group-hover:text-blue-600 transition-colors" />
        </button>
      </div>

      {/* Enhanced Legend */}
      <div className="absolute bottom-6 left-6 z-[1000]">
        <div className="bg-white/95 backdrop-blur-sm border border-gray-200 rounded-xl shadow-lg p-5 max-w-xs">
          <h3 className="font-bold text-gray-900 mb-4 flex items-center text-lg">
            <Layers className="h-5 w-5 mr-2 text-emerald-600" />
            Site Legend
          </h3>
          <div className="space-y-3">
            <div className="flex items-center justify-between">
              <div className="flex items-center space-x-3">
                <div className="w-5 h-5 bg-gradient-to-br from-emerald-400 to-emerald-600 rounded-full border-2 border-white shadow-md animate-pulse"></div>
                <span className="text-sm font-medium text-gray-700">
                  Premium Sites
                </span>
              </div>
              <span className="text-xs bg-emerald-100 text-emerald-800 px-2 py-1 rounded-full font-medium">
                90%+
              </span>
            </div>
            <div className="flex items-center justify-between">
              <div className="flex items-center space-x-3">
                <div className="w-5 h-5 bg-gradient-to-br from-green-400 to-green-600 rounded-full border-2 border-white shadow-md"></div>
                <span className="text-sm font-medium text-gray-700">
                  High Quality
                </span>
              </div>
              <span className="text-xs bg-green-100 text-green-800 px-2 py-1 rounded-full font-medium">
                80-90%
              </span>
            </div>
            <div className="flex items-center justify-between">
              <div className="flex items-center space-x-3">
                <div className="w-5 h-5 bg-gradient-to-br from-yellow-400 to-yellow-600 rounded-full border-2 border-white shadow-md"></div>
                <span className="text-sm font-medium text-gray-700">
                  Good Sites
                </span>
              </div>
              <span className="text-xs bg-yellow-100 text-yellow-800 px-2 py-1 rounded-full font-medium">
                70-80%
              </span>
            </div>
            <div className="flex items-center space-x-3">
              <div className="w-5 h-5 bg-gradient-to-br from-purple-500 to-pink-500 rounded-full border-2 border-white shadow-md animate-pulse"></div>
              <span className="text-sm font-medium text-gray-700">
                AI Generated
              </span>
            </div>
            <div className="flex items-center space-x-3">
              <div className="w-5 h-5 bg-blue-500 rounded-full border-2 border-white shadow-md">
                <div className="w-full h-full bg-blue-300 rounded-full animate-ping opacity-60"></div>
              </div>
              <span className="text-sm font-medium text-gray-700">
                Your Location
              </span>
            </div>
          </div>

          {backendSites.length > 0 && (
            <div className="mt-4 pt-4 border-t border-gray-200">
              <div className="flex items-center justify-between text-xs">
                <span className="text-gray-500">Backend Status:</span>
                <div className="flex items-center space-x-1">
                  <div className="w-2 h-2 bg-green-500 rounded-full animate-pulse"></div>
                  <span className="text-green-600 font-medium">Connected</span>
                </div>
              </div>
              <div className="text-xs text-gray-500 mt-1">
                {backendSites.length} sites • {markersRef.current.length}{" "}
                markers displayed
              </div>
            </div>
          )}
        </div>
      </div>

      {/* Map Container */}
      <div ref={mapRef} className="h-full w-full" />

      {/* Enhanced Custom Styles */}
      <style>{`
  .enhanced-popup .leaflet-popup-content-wrapper {
    background: linear-gradient(135deg, #f8fafc 0%, #ffffff 100%);
    border-radius: 16px;
    box-shadow: 0 20px 40px rgba(0, 0, 0, 0.1), 0 4px 12px rgba(0, 0, 0, 0.05);
    border: 1px solid rgba(229, 231, 235, 0.8);
  }
  .enhanced-popup .leaflet-popup-tip {
    background: linear-gradient(135deg, #f8fafc 0%, #ffffff 100%);
    border: 1px solid rgba(229, 231, 235, 0.8);
  }
  .ai-recommendation-popup .leaflet-popup-content-wrapper {
    background: linear-gradient(135deg, #faf5ff 0%, #fdf2f8 50%, #fff1f2 100%);
    border-radius: 16px;
    box-shadow: 0 25px 50px rgba(147, 51, 234, 0.2), 0 8px 16px rgba(236, 72, 153, 0.1);
    border: 2px solid rgba(196, 181, 253, 0.3);
  }
  .ai-recommendation-popup .leaflet-popup-tip {
    background: linear-gradient(135deg, #faf5ff 0%, #fdf2f8 100%);
    border: 2px solid rgba(196, 181, 253, 0.3);
  }
  .enhanced-div-icon, .ai-recommendation-icon, .search-result-marker, .user-location-marker {
    background: transparent;
    border: none;
  }
  .leaflet-control-attribution {
    background: rgba(255, 255, 255, 0.9);
    backdrop-filter: blur(8px);
    border-radius: 8px;
    border: 1px solid rgba(229, 231, 235, 0.5);
  }
  .leaflet-control-zoom {
    border: none;
    box-shadow: 0 4px 12px rgba(0, 0, 0, 0.1);
  }
  .leaflet-control-zoom a {
    background: rgba(255, 255, 255, 0.95);
    backdrop-filter: blur(8px);
    border: 1px solid rgba(229, 231, 235, 0.5);
    color: #374151;
    font-weight: 600;
    transition: all 0.2s ease;
  }
  .leaflet-control-zoom a:hover {
    background: #f3f4f6;
    color: #059669;
  }
  .leaflet-popup-close-button {
    color: #9ca3af;
    font-size: 18px;
    font-weight: bold;
    padding: 8px;
    margin: 4px;
    border-radius: 6px;
    transition: all 0.2s ease;
  }
  .leaflet-popup-close-button:hover {
    background: #f3f4f6;
    color: #374151;
  }
`}</style>
    </div>
  );
};

export default MapComponent;
