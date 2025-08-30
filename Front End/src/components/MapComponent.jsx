import React, { useEffect, useRef, useState } from "react";
import {
  Search,
  MapPin,
  Layers,
  Satellite,
  Globe,
  Moon,
  Zap,
  Navigation,
} from "lucide-react";

const MapComponent = () => {
  const mapRef = useRef(null);
  const mapInstanceRef = useRef(null);
  const markersRef = useRef([]);
  const [searchQuery, setSearchQuery] = useState("");
  const [currentLayer, setCurrentLayer] = useState("street");
  const [isSearching, setIsSearching] = useState(false);
  const [searchResults, setSearchResults] = useState([]);
  const [capitalInput, setCapitalInput] = useState("");
  const [isGenerating, setIsGenerating] = useState(false);
  const [showResults, setShowResults] = useState(false);

  // Sample hydrogen sites data
  const hydrogenSites = [
    {
      id: 1,
      name: "Bhuj Green Hydrogen Hub",
      lat: 23.242,
      lng: 69.6669,
      production: "485 kg/day",
      lcoh: "$3.2/kg",
      roi: "18.5%",
      status: "AI Recommended",
    },
    {
      id: 2,
      name: "Rajkot Solar-H2 Plant",
      lat: 22.3039,
      lng: 70.8022,
      production: "450 kg/day",
      lcoh: "$3.8/kg",
      roi: "15.2%",
      status: "AI Recommended",
    },
    {
      id: 3,
      name: "Gandhinagar Wind-H2 Facility",
      lat: 23.2156,
      lng: 72.6369,
      production: "420 kg/day",
      lcoh: "$4.1/kg",
      roi: "14.8%",
      status: "Under Analysis",
    },
  ];

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
    // Dynamically load Leaflet CSS and JS
    const loadLeaflet = async () => {
      // Load CSS
      if (!document.querySelector('link[href*="leaflet"]')) {
        const cssLink = document.createElement("link");
        cssLink.rel = "stylesheet";
        cssLink.href =
          "https://cdnjs.cloudflare.com/ajax/libs/leaflet/1.9.4/leaflet.min.css";
        document.head.appendChild(cssLink);
      }

      // Load JS
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

  const initializeMap = () => {
    if (!mapRef.current || mapInstanceRef.current) return;

    // Initialize map centered on Gujarat, India
    const map = window.L.map(mapRef.current, {
      zoomControl: false,
      attributionControl: false,
    }).setView([23.0225, 72.5714], 7);

    // Add zoom control to bottom right
    window.L.control.zoom({ position: "bottomright" }).addTo(map);

    // Add attribution
    window.L.control
      .attribution({ position: "bottomleft", prefix: false })
      .addTo(map);

    // Add initial tile layer
    const initialLayer = window.L.tileLayer(mapLayers[currentLayer].url, {
      attribution: mapLayers[currentLayer].attribution,
      maxZoom: 19,
    }).addTo(map);

    mapInstanceRef.current = map;

    // Add hydrogen site markers
    addHydrogenSiteMarkers(map);
  };

  const addHydrogenSiteMarkers = (map) => {
    // Clear existing markers
    markersRef.current.forEach((marker) => map.removeLayer(marker));
    markersRef.current = [];

    hydrogenSites.forEach((site) => {
      // Create custom icon
      const customIcon = window.L.divIcon({
        html: `
          <div class="relative">
            <div class="w-8 h-8 bg-emerald-500 rounded-full border-2 border-white shadow-lg flex items-center justify-center transform hover:scale-110 transition-transform">
              <svg class="w-4 h-4 text-white" fill="currentColor" viewBox="0 0 20 20">
                <path d="M13 6a3 3 0 11-6 0 3 3 0 016 0zM18 8A6 6 0 006 8c0 7-3 9-3 9h18s-3-2-3-9z"/>
              </svg>
            </div>
            <div class="absolute -bottom-1 left-1/2 transform -translate-x-1/2 w-2 h-2 bg-emerald-500 rotate-45"></div>
          </div>
        `,
        className: "custom-div-icon",
        iconSize: [32, 32],
        iconAnchor: [16, 32],
      });

      const marker = window.L.marker([site.lat, site.lng], {
        icon: customIcon,
      }).addTo(map);

      // Create popup content
      const popupContent = `
        <div class="p-4 min-w-64">
          <div class="flex items-center justify-between mb-3">
            <h3 class="text-lg font-bold text-gray-900">${site.name}</h3>
            <span class="px-2 py-1 text-xs font-medium bg-emerald-100 text-emerald-800 rounded-full">${
              site.status
            }</span>
          </div>
          <div class="space-y-2">
            <div class="flex justify-between">
              <span class="text-gray-600">Production:</span>
              <span class="font-semibold text-blue-600">${
                site.production
              }</span>
            </div>
            <div class="flex justify-between">
              <span class="text-gray-600">LCOH:</span>
              <span class="font-semibold text-emerald-600">${site.lcoh}</span>
            </div>
            <div class="flex justify-between">
              <span class="text-gray-600">ROI:</span>
              <span class="font-semibold text-purple-600">${site.roi}</span>
            </div>
          </div>
          <div class="mt-3 pt-3 border-t border-gray-200">
            <div class="flex items-center text-sm text-gray-500">
              <svg class="w-4 h-4 mr-1" fill="currentColor" viewBox="0 0 20 20">
                <path d="M5.05 4.05a7 7 0 119.9 9.9L10 18.9l-4.95-4.95a7 7 0 010-9.9zM10 11a2 2 0 100-4 2 2 0 000 4z"/>
              </svg>
              ${site.lat.toFixed(4)}°, ${site.lng.toFixed(4)}°
            </div>
          </div>
        </div>
      `;

      marker.bindPopup(popupContent, {
        maxWidth: 300,
        className: "custom-popup",
      });

      markersRef.current.push(marker);
    });
  };

  const switchLayer = (layerType) => {
    if (!mapInstanceRef.current) return;

    setCurrentLayer(layerType);

    // Remove all tile layers
    mapInstanceRef.current.eachLayer((layer) => {
      if (layer instanceof window.L.TileLayer) {
        mapInstanceRef.current.removeLayer(layer);
      }
    });

    // Add new layer
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
      // Using Nominatim API for geocoding
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

    // Fly to location
    mapInstanceRef.current.setView([result.lat, result.lng], 12);

    // Add temporary marker
    const tempIcon = window.L.divIcon({
      html: `
        <div class="w-6 h-6 bg-red-500 rounded-full border-2 border-white shadow-lg animate-pulse"></div>
      `,
      className: "temp-marker",
      iconSize: [24, 24],
      iconAnchor: [12, 12],
    });

    const tempMarker = window.L.marker([result.lat, result.lng], {
      icon: tempIcon,
    }).addTo(mapInstanceRef.current);

    // Remove temp marker after 3 seconds
    setTimeout(() => {
      mapInstanceRef.current.removeLayer(tempMarker);
    }, 3000);

    setShowResults(false);
    setSearchQuery("");
  };

  const generateAIRecommendation = async () => {
    if (!capitalInput.trim()) {
      alert("Please enter a capital amount");
      return;
    }

    if (!mapInstanceRef.current) return;

    setIsGenerating(true);

    try {
      // Simulate AI generation delay
      await new Promise((resolve) => setTimeout(resolve, 2000));

      // For demonstration, zoom to a random location in India
      const indianCities = [
        { name: "Mumbai", lat: 19.076, lng: 72.8777 },
        { name: "Delhi", lat: 28.7041, lng: 77.1025 },
        { name: "Bangalore", lat: 12.9716, lng: 77.5946 },
        { name: "Chennai", lat: 13.0827, lng: 80.2707 },
        { name: "Hyderabad", lat: 17.385, lng: 78.4867 },
        { name: "Pune", lat: 18.5204, lng: 73.8567 },
        { name: "Kolkata", lat: 22.5726, lng: 88.3639 },
        { name: "Ahmedabad", lat: 23.0225, lng: 72.5714 },
      ];

      const randomCity =
        indianCities[Math.floor(Math.random() * indianCities.length)];

      // Zoom to the selected city
      mapInstanceRef.current.setView([randomCity.lat, randomCity.lng], 12);

      // Add AI recommendation marker
      const aiIcon = window.L.divIcon({
        html: `
          <div class="relative">
            <div class="w-10 h-10 bg-gradient-to-r from-purple-500 to-pink-500 rounded-full border-3 border-white shadow-lg flex items-center justify-center transform animate-bounce">
              <svg class="w-5 h-5 text-white" fill="currentColor" viewBox="0 0 20 20">
                <path d="M9.049 2.927c.3-.921 1.603-.921 1.902 0l1.07 3.292a1 1 0 00.95.69h3.462c.969 0 1.371 1.24.588 1.81l-2.8 2.034a1 1 0 00-.364 1.118l1.07 3.292c.3.921-.755 1.688-1.54 1.118l-2.8-2.034a1 1 0 00-1.175 0l-2.8 2.034c-.784.57-1.838-.197-1.539-1.118l1.07-3.292a1 1 0 00-.364-1.118L2.98 8.72c-.783-.57-.38-1.81.588-1.81h3.461a1 1 0 00.951-.69l1.07-3.292z"/>
              </svg>
            </div>
            <div class="absolute -bottom-1 left-1/2 transform -translate-x-1/2 w-3 h-3 bg-gradient-to-r from-purple-500 to-pink-500 rotate-45"></div>
          </div>
        `,
        className: "ai-recommendation-icon",
        iconSize: [40, 40],
        iconAnchor: [20, 40],
      });

      const aiMarker = window.L.marker([randomCity.lat, randomCity.lng], {
        icon: aiIcon,
      }).addTo(mapInstanceRef.current);

      // Create AI recommendation popup
      const aiPopupContent = `
        <div class="p-4 min-w-64">
          <div class="flex items-center justify-between mb-3">
            <h3 class="text-lg font-bold text-gray-900">${
              randomCity.name
            } AI Recommendation</h3>
            <span class="px-2 py-1 text-xs font-medium bg-gradient-to-r from-purple-100 to-pink-100 text-purple-800 rounded-full">AI Generated</span>
          </div>
          <div class="space-y-2">
            <div class="flex justify-between">
              <span class="text-gray-600">Investment:</span>
              <span class="font-semibold text-green-600">₹${capitalInput} Cr</span>
            </div>
            <div class="flex justify-between">
              <span class="text-gray-600">Predicted LCOH:</span>
              <span class="font-semibold text-emerald-600">${(
                Math.random() * 2 +
                2.5
              ).toFixed(1)}/kg</span>
            </div>
            <div class="flex justify-between">
              <span class="text-gray-600">Expected ROI:</span>
              <span class="font-semibold text-purple-600">${(
                Math.random() * 10 +
                12
              ).toFixed(1)}%</span>
            </div>
            <div class="flex justify-between">
              <span class="text-gray-600">Feasibility:</span>
              <span class="font-semibold text-blue-600">${(
                Math.random() * 20 +
                75
              ).toFixed(1)}/100</span>
            </div>
          </div>
          <div class="mt-3 pt-3 border-t border-gray-200">
            <div class="flex items-center text-sm text-purple-600">
              <svg class="w-4 h-4 mr-1" fill="currentColor" viewBox="0 0 20 20">
                <path d="M9.049 2.927c.3-.921 1.603-.921 1.902 0l1.07 3.292a1 1 0 00.95.69h3.462c.969 0 1.371 1.24.588 1.81l-2.8 2.034a1 1 0 00-.364 1.118l1.07 3.292c.3.921-.755 1.688-1.54 1.118l-2.8-2.034a1 1 0 00-1.175 0l-2.8 2.034c-.784.57-1.838-.197-1.539-1.118l1.07-3.292a1 1 0 00-.364-1.118L2.98 8.72c-.783-.57-.38-1.81.588-1.81h3.461a1 1 0 00.951-.69l1.07-3.292z"/>
              </svg>
              AI-Optimized based on your capital
            </div>
          </div>
        </div>
      `;

      aiMarker
        .bindPopup(aiPopupContent, {
          maxWidth: 300,
          className: "ai-recommendation-popup",
        })
        .openPopup();

      // Clear inputs
      setCapitalInput("");
    } catch (error) {
      console.error("AI generation error:", error);
      alert("Failed to generate AI recommendation. Please try again.");
    } finally {
      setIsGenerating(false);
    }
  };

  // Get current location and show on map
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

          // Add user location marker
          const userIcon = window.L.divIcon({
            html: `
              <div class="w-4 h-4 bg-blue-500 rounded-full border-2 border-white shadow-lg relative">
                <div class="absolute inset-0 bg-blue-300 rounded-full animate-ping"></div>
              </div>
            `,
            className: "user-location-marker",
            iconSize: [16, 16],
            iconAnchor: [8, 8],
          });

          window.L.marker([latitude, longitude], { icon: userIcon })
            .addTo(mapInstanceRef.current)
            .bindPopup("Your Location")
            .openPopup();
        }
      },
      (error) => {
        console.error("Error getting location:", error);
        alert("Unable to retrieve your location.");
      }
    );
  };

  return (
    <div className="relative h-screen w-full bg-gray-100">
      {/* Header with Search and AI Features */}
      <div className="absolute top-0 left-0 right-0 z-[1000] bg-white/95 backdrop-blur-sm shadow-sm border-b border-gray-200">
        <div className="px-6 py-4">
          {/* Top Row - Title and Stats */}
          <div className="flex items-center justify-between mb-4">
            <div className="flex items-center space-x-3">
              <div className="p-2 bg-emerald-500 rounded-lg">
                <Zap className="h-6 w-6 text-white" />
              </div>
              <div>
                <h1 className="text-xl font-bold text-gray-900">
                  Green Hydrogen Infrastructure Map
                </h1>
                <p className="text-sm text-gray-600">
                  AI-Optimized Site Locations
                </p>
              </div>
            </div>

            {/* Stats */}
            <div className="hidden md:flex items-center space-x-6 text-sm">
              <div className="text-center">
                <div className="font-bold text-emerald-600">
                  {hydrogenSites.length}
                </div>
                <div className="text-gray-600">Sites</div>
              </div>
              <div className="text-center">
                <div className="font-bold text-blue-600">1,355 kg/day</div>
                <div className="text-gray-600">Total Production</div>
              </div>
              <div className="text-center">
                <div className="font-bold text-purple-600">16.2%</div>
                <div className="text-gray-600">Avg ROI</div>
              </div>
            </div>
          </div>

          {/* Bottom Row - Search and AI Controls */}
          <div className="flex items-center space-x-4">
            {/* Location Search */}
            <div className="relative flex-1 max-w-md">
              <div className="relative">
                <Search className="absolute left-3 top-1/2 transform -translate-y-1/2 h-4 w-4 text-gray-400" />
                <input
                  type="text"
                  placeholder="Search for a location for new site recommendation..."
                  value={searchQuery}
                  onChange={(e) => {
                    setSearchQuery(e.target.value);
                    handleSearch(e.target.value);
                  }}
                  className="w-full pl-10 pr-4 py-2 bg-gray-50 border border-gray-200 rounded-lg focus:ring-2 focus:ring-emerald-500 focus:border-emerald-500 outline-none text-sm"
                />
                {isSearching && (
                  <div className="absolute right-3 top-1/2 transform -translate-y-1/2">
                    <div className="animate-spin rounded-full h-4 w-4 border-b-2 border-emerald-500"></div>
                  </div>
                )}
              </div>

              {/* Search Results */}
              {showResults && searchResults.length > 0 && (
                <div className="absolute top-full mt-1 w-full bg-white border border-gray-200 rounded-lg shadow-lg max-h-64 overflow-y-auto">
                  {searchResults.map((result) => (
                    <button
                      key={result.id}
                      onClick={() => selectSearchResult(result)}
                      className="w-full text-left px-4 py-3 hover:bg-gray-50 border-b border-gray-100 last:border-b-0 transition-colors"
                    >
                      <div className="flex items-center space-x-3">
                        <MapPin className="h-4 w-4 text-gray-400 flex-shrink-0" />
                        <div className="truncate">
                          <div className="font-medium text-gray-900 truncate">
                            {result.name.split(",")[0]}
                          </div>
                          <div className="text-sm text-gray-500 truncate">
                            {result.name}
                          </div>
                        </div>
                      </div>
                    </button>
                  ))}
                </div>
              )}
            </div>

            {/* Capital Input */}
            <div className="flex items-center space-x-3">
              <input
                type="text"
                placeholder="Enter your capital"
                value={capitalInput}
                onChange={(e) => setCapitalInput(e.target.value)}
                className="px-4 py-2 bg-gray-50 border border-gray-200 rounded-lg focus:ring-2 focus:ring-blue-500 focus:border-blue-500 outline-none text-sm w-40"
              />

              {/* Generate AI Button */}
              <button
                onClick={generateAIRecommendation}
                disabled={isGenerating || !capitalInput.trim()}
                className="flex items-center space-x-2 px-4 py-2 bg-gradient-to-r from-emerald-500 to-green-600 text-white rounded-lg hover:from-emerald-600 hover:to-green-700 transition-all duration-200 font-medium text-sm shadow-sm disabled:opacity-50 disabled:cursor-not-allowed"
              >
                {isGenerating ? (
                  <>
                    <div className="animate-spin rounded-full h-4 w-4 border-b-2 border-white"></div>
                    <span>Generating...</span>
                  </>
                ) : (
                  <>
                    <svg
                      className="h-4 w-4"
                      fill="currentColor"
                      viewBox="0 0 20 20"
                    >
                      <path d="M9.049 2.927c.3-.921 1.603-.921 1.902 0l1.07 3.292a1 1 0 00.95.69h3.462c.969 0 1.371 1.24.588 1.81l-2.8 2.034a1 1 0 00-.364 1.118l1.07 3.292c.3.921-.755 1.688-1.54 1.118l-2.8-2.034a1 1 0 00-1.175 0l-2.8 2.034c-.784.57-1.838-.197-1.539-1.118l1.07-3.292a1 1 0 00-.364-1.118L2.98 8.72c-.783-.57-.38-1.81.588-1.81h3.461a1 1 0 00.951-.69l1.07-3.292z" />
                    </svg>
                    <span>Generate from AI</span>
                  </>
                )}
              </button>
            </div>
          </div>
        </div>
      </div>

      {/* Layer Controls */}
      <div className="absolute top-32 right-6 z-[1000]">
        <div className="bg-white/95 backdrop-blur-sm border border-gray-200 rounded-lg shadow-sm p-1">
          <div className="flex flex-col space-y-1">
            <button
              onClick={() => switchLayer("street")}
              className={`flex items-center space-x-2 px-3 py-2 rounded-md text-sm font-medium transition-colors ${
                currentLayer === "street"
                  ? "bg-emerald-500 text-white"
                  : "text-gray-700 hover:bg-gray-100"
              }`}
            >
              <MapPin className="h-4 w-4" />
              <span>Street</span>
            </button>
            <button
              onClick={() => switchLayer("satellite")}
              className={`flex items-center space-x-2 px-3 py-2 rounded-md text-sm font-medium transition-colors ${
                currentLayer === "satellite"
                  ? "bg-emerald-500 text-white"
                  : "text-gray-700 hover:bg-gray-100"
              }`}
            >
              <Satellite className="h-4 w-4" />
              <span>Satellite</span>
            </button>
            <button
              onClick={() => switchLayer("dark")}
              className={`flex items-center space-x-2 px-3 py-2 rounded-md text-sm font-medium transition-colors ${
                currentLayer === "dark"
                  ? "bg-emerald-500 text-white"
                  : "text-gray-700 hover:bg-gray-100"
              }`}
            >
              <Moon className="h-4 w-4" />
              <span>Dark</span>
            </button>
          </div>
        </div>
      </div>

      {/* Current Location Button */}
      <div className="absolute bottom-32 right-6 z-[1000]">
        <button
          onClick={getCurrentLocation}
          className="p-3 bg-white/95 backdrop-blur-sm border border-gray-200 rounded-lg shadow-sm hover:bg-gray-50 transition-colors"
          title="Go to current location"
        >
          <Navigation className="h-5 w-5 text-gray-700" />
        </button>
      </div>

      {/* Legend */}
      <div className="absolute bottom-6 left-6 z-[1000]">
        <div className="bg-white/95 backdrop-blur-sm border border-gray-200 rounded-lg shadow-sm p-4">
          <h3 className="font-semibold text-gray-900 mb-3 flex items-center">
            <Layers className="h-4 w-4 mr-2" />
            Legend
          </h3>
          <div className="space-y-2">
            <div className="flex items-center space-x-3">
              <div className="w-4 h-4 bg-emerald-500 rounded-full border border-white shadow-sm"></div>
              <span className="text-sm text-gray-700">
                AI Recommended Sites
              </span>
            </div>
            <div className="flex items-center space-x-3">
              <div className="w-4 h-4 bg-blue-500 rounded-full border border-white shadow-sm"></div>
              <span className="text-sm text-gray-700">Your Location</span>
            </div>
            <div className="flex items-center space-x-3">
              <div className="w-4 h-4 bg-red-500 rounded-full border border-white shadow-sm animate-pulse"></div>
              <span className="text-sm text-gray-700">Search Result</span>
            </div>
          </div>
        </div>
      </div>

      {/* Map Container */}
      <div ref={mapRef} className="h-full w-full" />

      {/* Custom Styles */}
      <style jsx>{`
        .custom-popup .leaflet-popup-content-wrapper {
          background: white;
          border-radius: 8px;
          box-shadow: 0 10px 25px rgba(0, 0, 0, 0.1);
        }
        .ai-recommendation-popup .leaflet-popup-content-wrapper {
          background: linear-gradient(135deg, #f3e8ff 0%, #fce7f3 100%);
          border-radius: 12px;
          box-shadow: 0 20px 40px rgba(147, 51, 234, 0.15);
        }
        .ai-recommendation-popup .leaflet-popup-tip {
          background: linear-gradient(135deg, #f3e8ff 0%, #fce7f3 100%);
        }
        .custom-div-icon {
          background: transparent;
          border: none;
        }
        .leaflet-control-attribution {
          background: rgba(255, 255, 255, 0.8);
          backdrop-filter: blur(4px);
        }
      `}</style>
    </div>
  );
};

export default MapComponent;
