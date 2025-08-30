import React, { useState, useEffect } from "react";
import {
  MapPin,
  Zap,
  Wind,
  Droplets,
  Gauge,
  TrendingUp,
  Sun,
  Thermometer,
  Activity,
  Battery,
  Brain,
  DollarSign,
  Target,
} from "lucide-react";

import { useNavigate } from "react-router-dom";

const GreenHydrogenHome = () => {
  const [backendSites, setBackendSites] = useState([]);
  const [loading, setLoading] = useState(true);
  const [error, setError] = useState(null);
  const navigate = useNavigate();

  // Hardcoded AI-recommended sites
  const hardcodedSites = [
    {
      City: "Bhuj",
      Latitude: 23.242,
      Longitude: 69.6669,
      "Solar_Irradiance_kWh/m²/day": 6.1,
      Temperature_C: 27.8,
      "Wind_Speed_m/s": 8.1,
      PV_Power_kW: 2800,
      Wind_Power_kW: 2100,
      "Electrolyzer_Efficiency_%": 80,
      "Hydrogen_Production_kg/day": 485,
      Desalination_Power_kW: 340,
      "System_Efficiency_%": 75,
      Feasibility_Score: 96.5,
      LCOH: 3.2,
      ROI: 18.5,
    },
    {
      City: "Rajkot",
      Latitude: 22.3039,
      Longitude: 70.8022,
      "Solar_Irradiance_kWh/m²/day": 5.8,
      Temperature_C: 28.5,
      "Wind_Speed_m/s": 7.2,
      PV_Power_kW: 2500,
      Wind_Power_kW: 1800,
      "Electrolyzer_Efficiency_%": 78,
      "Hydrogen_Production_kg/day": 450,
      Desalination_Power_kW: 320,
      "System_Efficiency_%": 72,
      Feasibility_Score: 94.2,
      LCOH: 3.8,
      ROI: 15.2,
    },
  ];

  useEffect(() => {
    const fetchBackendSites = async () => {
      try {
        setLoading(true);
        const response = await fetch(
          "https://hackout2025-backend.onrender.com/ml/sites/top"
        );
        if (!response.ok) {
          throw new Error("Failed to fetch sites data");
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

  // Combine hardcoded sites with backend sites
  const allSites = [...hardcodedSites, ...backendSites];

  const getScoreColor = (score) => {
    if (score >= 90) return "text-emerald-600 bg-emerald-50";
    if (score >= 80) return "text-blue-600 bg-blue-50";
    if (score >= 70) return "text-amber-600 bg-amber-50";
    return "text-red-600 bg-red-50";
  };

  const getScoreBadgeColor = (score) => {
    if (score >= 90) return "bg-gradient-to-r from-emerald-500 to-green-600";
    if (score >= 80) return "bg-gradient-to-r from-blue-500 to-indigo-600";
    if (score >= 70) return "bg-gradient-to-r from-amber-500 to-orange-600";
    return "bg-gradient-to-r from-red-500 to-pink-600";
  };

  const MetricCard = ({
    icon: Icon,
    label,
    value,
    unit,
    color = "text-gray-700",
  }) => (
    <div className="flex items-center space-x-3 p-3 bg-white/50 rounded-lg backdrop-blur-sm">
      <Icon className={`h-5 w-5 ${color}`} />
      <div>
        <p className="text-sm text-gray-600">{label}</p>
        <p className="font-semibold text-gray-900">
          {value} {unit}
        </p>
      </div>
    </div>
  );

  return (
    <div className="min-h-screen bg-gradient-to-br from-blue-50 via-green-50 to-emerald-50">
      {/* Hero Section */}
      <div className="relative overflow-hidden bg-gradient-to-r from-emerald-600 via-green-700 to-blue-800">
        <div className="absolute inset-0 bg-black/20"></div>
        <div className="absolute inset-0 opacity-10">
          <div className="absolute top-1/4 left-1/4 w-64 h-64 bg-white rounded-full blur-3xl"></div>
          <div className="absolute bottom-1/4 right-1/4 w-96 h-96 bg-emerald-300 rounded-full blur-3xl"></div>
        </div>
        <div className="relative max-w-7xl mx-auto px-4 sm:px-6 lg:px-8 py-24">
          <div className="text-center">
            <h1 className="text-5xl md:text-6xl font-bold text-white mb-6 tracking-tight">
              AI-Powered Green Hydrogen
              <span className="block text-emerald-300">
                Site Recommendations
              </span>
            </h1>
            <p className="text-xl md:text-2xl text-green-100 mb-8 max-w-3xl mx-auto leading-relaxed">
              Machine learning-optimized locations for sustainable hydrogen
              infrastructure based on LCOH and ROI analysis
            </p>
            <div className="flex flex-wrap justify-center gap-4 text-white/90 mb-6">
              <div className="flex items-center space-x-2 bg-white/10 px-4 py-2 rounded-full backdrop-blur-sm">
                <Brain className="h-5 w-5" />
                <span>AI Optimized</span>
              </div>
              <div className="flex items-center space-x-2 bg-white/10 px-4 py-2 rounded-full backdrop-blur-sm">
                <DollarSign className="h-5 w-5" />
                <span>LCOH Analysis</span>
              </div>
              <div className="flex items-center space-x-2 bg-white/10 px-4 py-2 rounded-full backdrop-blur-sm">
                <Target className="h-5 w-5" />
                <span>ROI Optimized</span>
              </div>
              <div className="flex items-center space-x-2 bg-white/10 px-4 py-2 rounded-full backdrop-blur-sm">
                <Droplets className="h-5 w-5" />
                <span>Hydrogen Production</span>
              </div>
            </div>
          </div>
        </div>
      </div>

      {/* Stats Overview */}
      <div className="relative -mt-16 max-w-7xl mx-auto px-4 sm:px-6 lg:px-8 mb-16">
        <div className="grid grid-cols-1 md:grid-cols-4 gap-6">
          <div className="bg-white/80 backdrop-blur-lg rounded-xl p-6 shadow-lg border border-white/20">
            <div className="flex items-center justify-between">
              <div>
                <p className="text-sm font-medium text-gray-600">Total Sites</p>
                <p className="text-2xl font-bold text-gray-900">
                  {allSites.length}
                </p>
              </div>
              <MapPin className="h-8 w-8 text-emerald-600" />
            </div>
          </div>
          <div className="bg-white/80 backdrop-blur-lg rounded-xl p-6 shadow-lg border border-white/20">
            <div className="flex items-center justify-between">
              <div>
                <p className="text-sm font-medium text-gray-600">
                  Avg Production
                </p>
                <p className="text-2xl font-bold text-gray-900">
                  {allSites.length > 0
                    ? Math.round(
                        allSites.reduce(
                          (sum, site) =>
                            sum + site["Hydrogen_Production_kg/day"],
                          0
                        ) / allSites.length
                      )
                    : 0}{" "}
                  kg/day
                </p>
              </div>
              <Droplets className="h-8 w-8 text-blue-600" />
            </div>
          </div>
          <div className="bg-white/80 backdrop-blur-lg rounded-xl p-6 shadow-lg border border-white/20">
            <div className="flex items-center justify-between">
              <div>
                <p className="text-sm font-medium text-gray-600">
                  Avg Efficiency
                </p>
                <p className="text-2xl font-bold text-gray-900">
                  {allSites.length > 0
                    ? Math.round(
                        allSites.reduce(
                          (sum, site) => sum + site["System_Efficiency_%"],
                          0
                        ) / allSites.length
                      )
                    : 0}
                  %
                </p>
              </div>
              <Gauge className="h-8 w-8 text-purple-600" />
            </div>
          </div>
          <div className="bg-white/80 backdrop-blur-lg rounded-xl p-6 shadow-lg border border-white/20">
            <div className="flex items-center justify-between">
              <div>
                <p className="text-sm font-medium text-gray-600">
                  Avg Feasibility
                </p>
                <p className="text-2xl font-bold text-gray-900">
                  {allSites.length > 0
                    ? (
                        allSites.reduce(
                          (sum, site) => sum + site.Feasibility_Score,
                          0
                        ) / allSites.length
                      ).toFixed(1)
                    : 0}
                </p>
              </div>
              <TrendingUp className="h-8 w-8 text-green-600" />
            </div>
          </div>
        </div>
      </div>

      {/* Main Content */}
      <div className="max-w-7xl mx-auto px-4 sm:px-6 lg:px-8 pb-20">
        <div className="text-center mb-12">
          <div className="flex items-center justify-center mb-4">
            <Brain className="h-8 w-8 text-emerald-600 mr-3" />
            <h2 className="text-3xl md:text-4xl font-bold text-gray-900">
              AI-Recommended Sites
            </h2>
          </div>
          <p className="text-lg text-gray-600 max-w-3xl mx-auto mb-6">
            Machine learning algorithms analyzed renewable energy potential,
            infrastructure costs, and market conditions to recommend these
            optimal locations for green hydrogen production
          </p>
          <div className="flex items-center justify-center space-x-8 text-sm text-gray-500">
            <div className="flex items-center space-x-2">
              <DollarSign className="h-4 w-4 text-emerald-600" />
              <span>
                Optimized for <strong>LCOH</strong> (Levelized Cost of Hydrogen)
              </span>
            </div>
            <div className="flex items-center space-x-2">
              <Target className="h-4 w-4 text-blue-600" />
              <span>
                Maximized <strong>ROI</strong> (Return on Investment)
              </span>
            </div>
          </div>
        </div>

        {error && (
          <div className="bg-amber-50 border border-amber-200 rounded-lg p-4 mb-8">
            <p className="text-amber-800">
              <span className="font-medium">Note:</span> Using demonstration
              data. Backend connection: {error}
            </p>
          </div>
        )}

        {loading && backendSites.length === 0 ? (
          <div className="text-center py-12">
            <div className="animate-spin rounded-full h-12 w-12 border-b-2 border-emerald-600 mx-auto mb-4"></div>
            <p className="text-gray-600">
              Loading additional sites from backend...
            </p>
          </div>
        ) : (
          <div className="grid grid-cols-1 lg:grid-cols-2 gap-8">
            {allSites.map((site, index) => (
              <div key={`${site.City}-${index}`} className="group">
                <div className="bg-white/80 backdrop-blur-lg rounded-2xl shadow-xl border border-white/20 overflow-hidden hover:shadow-2xl transition-all duration-300 hover:-translate-y-1">
                  {/* Header */}
                  <div className="bg-gradient-to-r from-emerald-600 to-green-700 p-6 relative overflow-hidden">
                    <div className="absolute top-2 right-2 bg-white/20 backdrop-blur-sm rounded-full px-3 py-1">
                      <div className="flex items-center space-x-1 text-white text-xs font-medium">
                        <Brain className="h-3 w-3" />
                        <span>AI Recommended</span>
                      </div>
                    </div>
                    <div className="flex items-center justify-between">
                      <div>
                        <h3 className="text-2xl font-bold text-white mb-2">
                          {site.City}
                        </h3>
                        <div className="flex items-center text-emerald-100">
                          <MapPin className="h-4 w-4 mr-2" />
                          <span className="text-sm">
                            {site.Latitude.toFixed(4)}°,{" "}
                            {site.Longitude.toFixed(4)}°
                          </span>
                        </div>
                      </div>
                      <div
                        className={`px-4 py-2 rounded-full text-white font-bold ${getScoreBadgeColor(
                          site.Feasibility_Score
                        )}`}
                      >
                        {site.Feasibility_Score.toFixed(1)}
                      </div>
                    </div>
                  </div>

                  {/* Content */}
                  <div className="p-6">
                    {/* AI Recommendation Metrics */}
                    <div className="mb-6">
                      <h4 className="text-lg font-semibold text-gray-900 mb-4 flex items-center">
                        <Brain className="h-5 w-5 text-emerald-600 mr-2" />
                        AI Recommendation Factors
                      </h4>
                      <div className="grid grid-cols-2 gap-4">
                        <div className="bg-gradient-to-r from-emerald-50 to-green-50 rounded-lg p-4 border border-emerald-200">
                          <div className="flex items-center space-x-3">
                            <DollarSign className="h-6 w-6 text-emerald-600" />
                            <div>
                              <p className="text-sm text-gray-600">LCOH</p>
                              <p className="text-xl font-bold text-gray-900">
                                ${site.LCOH}/kg
                              </p>
                              <p className="text-xs text-emerald-600 font-medium">
                                Levelized Cost of Hydrogen
                              </p>
                            </div>
                          </div>
                        </div>
                        <div className="bg-gradient-to-r from-blue-50 to-indigo-50 rounded-lg p-4 border border-blue-200">
                          <div className="flex items-center space-x-3">
                            <Target className="h-6 w-6 text-blue-600" />
                            <div>
                              <p className="text-sm text-gray-600">ROI</p>
                              <p className="text-xl font-bold text-gray-900">
                                {site.ROI}%
                              </p>
                              <p className="text-xs text-blue-600 font-medium">
                                Return on Investment
                              </p>
                            </div>
                          </div>
                        </div>
                      </div>
                    </div>

                    {/* Key Production Metrics */}
                    <div className="mb-6">
                      <h4 className="text-lg font-semibold text-gray-900 mb-4">
                        Production Capacity
                      </h4>
                      <div className="grid grid-cols-2 gap-4">
                        <div className="bg-gradient-to-r from-blue-50 to-indigo-50 rounded-lg p-4">
                          <div className="flex items-center space-x-3">
                            <Droplets className="h-6 w-6 text-blue-600" />
                            <div>
                              <p className="text-sm text-gray-600">
                                H₂ Production
                              </p>
                              <p className="text-xl font-bold text-gray-900">
                                {site["Hydrogen_Production_kg/day"]} kg/day
                              </p>
                            </div>
                          </div>
                        </div>
                        <div className="bg-gradient-to-r from-purple-50 to-pink-50 rounded-lg p-4">
                          <div className="flex items-center space-x-3">
                            <Gauge className="h-6 w-6 text-purple-600" />
                            <div>
                              <p className="text-sm text-gray-600">
                                System Efficiency
                              </p>
                              <p className="text-xl font-bold text-gray-900">
                                {site["System_Efficiency_%"]}%
                              </p>
                            </div>
                          </div>
                        </div>
                      </div>
                    </div>

                    {/* Environmental Conditions */}
                    <div className="mb-6">
                      <h4 className="text-lg font-semibold text-gray-900 mb-4">
                        Environmental Conditions
                      </h4>
                      <div className="grid grid-cols-2 gap-3">
                        <MetricCard
                          icon={Sun}
                          label="Solar Irradiance"
                          value={site["Solar_Irradiance_kWh/m²/day"]}
                          unit="kWh/m²/day"
                          color="text-yellow-600"
                        />
                        <MetricCard
                          icon={Thermometer}
                          label="Temperature"
                          value={site.Temperature_C}
                          unit="°C"
                          color="text-red-500"
                        />
                        <MetricCard
                          icon={Wind}
                          label="Wind Speed"
                          value={site["Wind_Speed_m/s"]}
                          unit="m/s"
                          color="text-sky-600"
                        />
                        <MetricCard
                          icon={Activity}
                          label="Electrolyzer Eff."
                          value={site["Electrolyzer_Efficiency_%"]}
                          unit="%"
                          color="text-indigo-600"
                        />
                      </div>
                    </div>

                    {/* Power Generation */}
                    <div className="mb-6">
                      <h4 className="text-lg font-semibold text-gray-900 mb-4">
                        Power Generation
                      </h4>
                      <div className="space-y-3">
                        <div className="flex items-center justify-between p-3 bg-gradient-to-r from-yellow-50 to-orange-50 rounded-lg">
                          <div className="flex items-center space-x-3">
                            <Zap className="h-5 w-5 text-yellow-600" />
                            <span className="font-medium text-gray-900">
                              PV Power
                            </span>
                          </div>
                          <span className="font-bold text-gray-900">
                            {site.PV_Power_kW.toLocaleString()} kW
                          </span>
                        </div>
                        <div className="flex items-center justify-between p-3 bg-gradient-to-r from-sky-50 to-blue-50 rounded-lg">
                          <div className="flex items-center space-x-3">
                            <Wind className="h-5 w-5 text-sky-600" />
                            <span className="font-medium text-gray-900">
                              Wind Power
                            </span>
                          </div>
                          <span className="font-bold text-gray-900">
                            {site.Wind_Power_kW.toLocaleString()} kW
                          </span>
                        </div>
                        <div className="flex items-center justify-between p-3 bg-gradient-to-r from-cyan-50 to-teal-50 rounded-lg">
                          <div className="flex items-center space-x-3">
                            <Battery className="h-5 w-5 text-cyan-600" />
                            <span className="font-medium text-gray-900">
                              Desalination
                            </span>
                          </div>
                          <span className="font-bold text-gray-900">
                            {site.Desalination_Power_kW} kW
                          </span>
                        </div>
                      </div>
                    </div>

                    {/* Feasibility Score */}
                    <div className="text-center">
                      <div
                        className={`inline-flex items-center px-6 py-3 rounded-full font-semibold ${getScoreColor(
                          site.Feasibility_Score
                        )} border-2 border-current/20`}
                      >
                        <TrendingUp className="h-5 w-5 mr-2" />
                        AI Feasibility Score:{" "}
                        {site.Feasibility_Score.toFixed(1)}/100
                      </div>
                      <p className="text-xs text-gray-500 mt-2">
                        Based on LCOH, ROI, and 12+ environmental factors
                      </p>
                    </div>
                  </div>
                </div>
              </div>
            ))}
          </div>
        )}

        {/* Loading State for Backend Data */}
        {loading && (
          <div className="text-center mt-8">
            <div className="inline-flex items-center px-4 py-2 bg-white/80 rounded-full shadow-lg">
              <div className="animate-spin rounded-full h-4 w-4 border-b-2 border-emerald-600 mr-3"></div>
              <span className="text-gray-700">
                Loading additional sites from backend...
              </span>
            </div>
          </div>
        )}
      </div>

      {/* Footer CTA */}
      <div className="bg-gradient-to-r from-gray-900 to-emerald-900">
        <div className="max-w-7xl mx-auto px-4 sm:px-6 lg:px-8 py-16">
          <div className="text-center">
            <h3 className="text-3xl font-bold text-white mb-4">
              Ready to Optimize Your Green Hydrogen Infrastructure?
            </h3>
            <p className="text-xl text-gray-300 mb-8 max-w-2xl mx-auto">
              Leverage our AI-powered mapping and optimization tools to identify
              the best locations for sustainable hydrogen production
            </p>
            <button
              onClick={() => navigate("/map")}
              className="bg-gradient-to-r from-emerald-500 to-green-600 text-white px-8 py-4 rounded-full font-semibold text-lg hover:from-emerald-600 hover:to-green-700 transition-all duration-300 transform hover:scale-105 shadow-lg"
            >
              Explore All Sites
            </button>
          </div>
        </div>
      </div>
    </div>
  );
};

const MetricCard = ({
  icon: Icon,
  label,
  value,
  unit,
  color = "text-gray-700",
}) => (
  <div className="flex items-center space-x-2 p-2 bg-white/60 rounded-lg">
    <Icon className={`h-4 w-4 ${color}`} />
    <div>
      <p className="text-xs text-gray-600">{label}</p>
      <p className="font-semibold text-gray-900 text-sm">
        {value} {unit}
      </p>
    </div>
  </div>
);

export default GreenHydrogenHome;
