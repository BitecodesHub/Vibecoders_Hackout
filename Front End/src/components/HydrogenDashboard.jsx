import React, { useState } from "react";
import {
  TrendingUp,
  MapPin,
  Zap,
  Activity,
  Target,
  DollarSign,
  Wind,
  Sun,
  BarChart3,
  RefreshCw,
  Download,
  Factory,
  Gauge,
} from "lucide-react";
import {
  LineChart,
  Line,
  XAxis,
  YAxis,
  CartesianGrid,
  Tooltip,
  ResponsiveContainer,
  BarChart,
  Bar,
  PieChart,
  Cell,
} from "recharts";

const HydrogenDashboard = () => {
  const [selectedTimeframe, setSelectedTimeframe] = useState("all");
  const [isLoading, setIsLoading] = useState(false);

  const kpiData = {
    totalSites: 2535,
    dailyProduction: 131.9,
    avgLCOH: 6.42,
    systemEfficiency: 78.5,
  };

  const monthlyData = [
    { month: "Jan", production: 105, efficiency: 76, cost: 6.8 },
    { month: "Feb", production: 112, efficiency: 77, cost: 6.7 },
    { month: "Mar", production: 128, efficiency: 79, cost: 6.5 },
    { month: "Apr", production: 139, efficiency: 79, cost: 6.4 },
    { month: "May", production: 145, efficiency: 79, cost: 6.3 },
    { month: "Jun", production: 152, efficiency: 80, cost: 6.2 },
  ];

  const stateData = [
    { state: "Tamil Nadu", sites: 4, production: 354 },
    { state: "Maharashtra", sites: 3, production: 282 },
    { state: "Rajasthan", sites: 4, production: 249 },
    { state: "Karnataka", sites: 4, production: 223 },
    { state: "Gujarat", sites: 5, production: 212 },
  ];

  const energyMixData = [
    { name: "Solar", value: 658, color: "#f59e0b" },
    { name: "Wind", value: 475, color: "#06b6d4" },
  ];

  const refreshData = () => {
    setIsLoading(true);
    setTimeout(() => setIsLoading(false), 1500);
  };

  const StatCard = ({
    icon: Icon,
    title,
    value,
    subtitle,
    trend,
    gradient,
  }) => (
    <div
      className={`relative overflow-hidden rounded-xl border border-white/20 backdrop-blur-md shadow-lg ${gradient} p-5 group hover:shadow-xl transition-all duration-300`}
    >
      <div className="absolute inset-0 bg-gradient-to-br from-white/10 to-transparent"></div>
      <div className="relative z-10">
        <div className="flex items-center justify-between mb-3">
          <div className="p-2 bg-white/20 rounded-lg">
            <Icon className="h-5 w-5 text-white" />
          </div>
          {trend && (
            <div className="flex items-center space-x-1 px-2 py-1 rounded-full text-xs font-medium bg-white/20 text-white">
              <TrendingUp
                className={`h-3 w-3 ${trend === "down" ? "rotate-180" : ""}`}
              />
            </div>
          )}
        </div>
        <div className="space-y-1">
          <h3 className="text-xl font-bold text-white">{value}</h3>
          <p className="text-sm text-white/90 font-medium">{title}</p>
          {subtitle && <p className="text-xs text-white/70">{subtitle}</p>}
        </div>
      </div>
    </div>
  );

  return (
    <div className="min-h-screen bg-gradient-to-br from-emerald-50 via-blue-50 to-purple-50 relative overflow-hidden">
      {/* Background elements */}
      <div className="absolute inset-0 overflow-hidden">
        <div className="absolute -top-40 -right-40 w-80 h-80 bg-emerald-300/20 rounded-full blur-3xl animate-pulse"></div>
        <div
          className="absolute -bottom-40 -left-40 w-80 h-80 bg-blue-300/20 rounded-full blur-3xl animate-pulse"
          style={{ animationDelay: "2s" }}
        ></div>
      </div>

      <div className="relative z-10 container mx-auto px-6 py-8">
        {/* Header */}
        <div className="mb-8">
          <div className="bg-white/95 backdrop-blur-md border border-white/20 rounded-xl shadow-lg p-6">
            <div className="flex items-center justify-between">
              <div className="flex items-center space-x-3">
                <div className="p-3 bg-gradient-to-br from-emerald-500 to-green-600 rounded-xl">
                  <BarChart3 className="h-6 w-6 text-white" />
                </div>
                <div>
                  <h1 className="text-2xl font-bold bg-gradient-to-r from-emerald-600 to-green-600 bg-clip-text text-transparent">
                    Green Hydrogen Analytics
                  </h1>
                  <p className="text-gray-600">
                    Infrastructure Performance Dashboard
                  </p>
                </div>
              </div>

              <div className="flex items-center space-x-3">
                <select
                  value={selectedTimeframe}
                  onChange={(e) => setSelectedTimeframe(e.target.value)}
                  className="px-3 py-2 bg-gray-50 border border-gray-200 rounded-lg text-sm"
                >
                  <option value="all">All Time</option>
                  <option value="year">This Year</option>
                  <option value="month">This Month</option>
                </select>

                <button
                  onClick={refreshData}
                  disabled={isLoading}
                  className="flex items-center space-x-2 px-3 py-2 bg-emerald-500 text-white rounded-lg hover:bg-emerald-600 transition-all duration-200 disabled:opacity-50"
                >
                  <RefreshCw
                    className={`h-4 w-4 ${isLoading ? "animate-spin" : ""}`}
                  />
                  <span>Refresh</span>
                </button>
              </div>
            </div>
          </div>
        </div>

        {/* KPI Cards */}
        <div className="grid grid-cols-1 md:grid-cols-2 lg:grid-cols-4 gap-6 mb-8">
          <StatCard
            icon={Factory}
            title="Total Sites"
            value={kpiData.totalSites.toLocaleString()}
            subtitle="Active installations"
            trend="up"
            gradient="bg-gradient-to-br from-emerald-500 to-green-600"
          />

          <StatCard
            icon={Gauge}
            title="Daily Production"
            value={`${kpiData.dailyProduction}t`}
            subtitle="Hydrogen output"
            trend="up"
            gradient="bg-gradient-to-br from-blue-500 to-cyan-600"
          />

          <StatCard
            icon={DollarSign}
            title="Average LCOH"
            value={`$${kpiData.avgLCOH}`}
            subtitle="Per kg hydrogen"
            trend="down"
            gradient="bg-gradient-to-br from-purple-500 to-pink-600"
          />

          <StatCard
            icon={Activity}
            title="System Efficiency"
            value={`${kpiData.systemEfficiency}%`}
            subtitle="Overall performance"
            trend="up"
            gradient="bg-gradient-to-br from-orange-500 to-red-600"
          />
        </div>

        {/* Charts Section */}
        <div className="grid grid-cols-1 lg:grid-cols-2 gap-8 mb-8">
          {/* Production Trends */}
          <div className="bg-white/95 backdrop-blur-md border border-white/20 rounded-xl shadow-lg p-6">
            <div className="flex items-center space-x-3 mb-6">
              <div className="p-2 bg-gradient-to-br from-emerald-500 to-green-600 rounded-lg">
                <TrendingUp className="h-5 w-5 text-white" />
              </div>
              <h3 className="text-lg font-bold text-gray-900">
                Production & Efficiency Trends
              </h3>
            </div>
            <div className="h-64">
              <ResponsiveContainer width="100%" height="100%">
                <LineChart data={monthlyData}>
                  <CartesianGrid strokeDasharray="3 3" stroke="#e5e7eb" />
                  <XAxis dataKey="month" stroke="#6b7280" fontSize={12} />
                  <YAxis yAxisId="left" stroke="#6b7280" fontSize={12} />
                  <YAxis
                    yAxisId="right"
                    orientation="right"
                    stroke="#6b7280"
                    fontSize={12}
                  />
                  <Tooltip
                    contentStyle={{
                      backgroundColor: "rgba(255, 255, 255, 0.95)",
                      backdropFilter: "blur(10px)",
                      borderRadius: "8px",
                      border: "none",
                      boxShadow: "0 10px 25px rgba(0, 0, 0, 0.1)",
                    }}
                  />
                  <Line
                    yAxisId="left"
                    type="monotone"
                    dataKey="production"
                    stroke="#10b981"
                    strokeWidth={3}
                    dot={{ fill: "#10b981", r: 4 }}
                  />
                  <Line
                    yAxisId="right"
                    type="monotone"
                    dataKey="efficiency"
                    stroke="#8b5cf6"
                    strokeWidth={3}
                    dot={{ fill: "#8b5cf6", r: 4 }}
                  />
                </LineChart>
              </ResponsiveContainer>
            </div>
          </div>

          {/* State Performance */}
          <div className="bg-white/95 backdrop-blur-md border border-white/20 rounded-xl shadow-lg p-6">
            <div className="flex items-center space-x-3 mb-6">
              <div className="p-2 bg-gradient-to-br from-blue-500 to-cyan-600 rounded-lg">
                <MapPin className="h-5 w-5 text-white" />
              </div>
              <h3 className="text-lg font-bold text-gray-900">
                State-wise Distribution
              </h3>
            </div>
            <div className="h-64">
              <ResponsiveContainer width="100%" height="100%">
                <BarChart data={stateData}>
                  <CartesianGrid strokeDasharray="3 3" stroke="#e5e7eb" />
                  <XAxis
                    dataKey="state"
                    stroke="#6b7280"
                    fontSize={11}
                    angle={-45}
                    textAnchor="end"
                    height={60}
                  />
                  <YAxis stroke="#6b7280" fontSize={12} />
                  <Tooltip
                    contentStyle={{
                      backgroundColor: "rgba(255, 255, 255, 0.95)",
                      backdropFilter: "blur(10px)",
                      borderRadius: "8px",
                      border: "none",
                      boxShadow: "0 10px 25px rgba(0, 0, 0, 0.1)",
                    }}
                  />
                  <Bar
                    dataKey="sites"
                    fill="url(#barGradient)"
                    radius={[4, 4, 0, 0]}
                  />
                  <defs>
                    <linearGradient
                      id="barGradient"
                      x1="0"
                      y1="0"
                      x2="0"
                      y2="1"
                    >
                      <stop offset="0%" stopColor="#06b6d4" />
                      <stop offset="100%" stopColor="#0891b2" />
                    </linearGradient>
                  </defs>
                </BarChart>
              </ResponsiveContainer>
            </div>
          </div>
        </div>

        {/* Secondary Metrics */}
        <div className="grid grid-cols-1 lg:grid-cols-3 gap-8 mb-8">
          {/* Energy Mix Pie Chart */}
          <div className="bg-white/95 backdrop-blur-md border border-white/20 rounded-xl shadow-lg p-6">
            <div className="flex items-center space-x-3 mb-6">
              <div className="p-2 bg-gradient-to-br from-orange-500 to-red-600 rounded-lg">
                <Zap className="h-5 w-5 text-white" />
              </div>
              <h3 className="text-lg font-bold text-gray-900">Energy Mix</h3>
            </div>
            <div className="h-48">
              <ResponsiveContainer width="100%" height="100%">
                <PieChart>
                  <Tooltip />
                  <PieChart
                    data={energyMixData}
                    cx="50%"
                    cy="50%"
                    outerRadius={70}
                    dataKey="value"
                  >
                    {energyMixData.map((entry, index) => (
                      <Cell key={`cell-${index}`} fill={entry.color} />
                    ))}
                  </PieChart>
                </PieChart>
              </ResponsiveContainer>
            </div>
            <div className="flex justify-center space-x-4 mt-4">
              {energyMixData.map((item, index) => (
                <div
                  key={index}
                  className="flex items-center space-x-2 text-sm"
                >
                  <div
                    className="w-3 h-3 rounded-full"
                    style={{ backgroundColor: item.color }}
                  ></div>
                  <span className="text-gray-600">{item.name}</span>
                  <span className="font-semibold">{item.value}MW</span>
                </div>
              ))}
            </div>
          </div>

          {/* Performance Metrics */}
          <div className="bg-white/95 backdrop-blur-md border border-white/20 rounded-xl shadow-lg p-6">
            <div className="flex items-center space-x-3 mb-6">
              <div className="p-2 bg-gradient-to-br from-purple-500 to-pink-600 rounded-lg">
                <Target className="h-5 w-5 text-white" />
              </div>
              <h3 className="text-lg font-bold text-gray-900">Key Metrics</h3>
            </div>
            <div className="space-y-4">
              {[
                { label: "Capacity Factor", value: "42.3%", trend: "up" },
                { label: "Water Usage", value: "8.9 L/kg", trend: "down" },
                { label: "CO₂ Avoided", value: "125.6 kt/yr", trend: "up" },
                { label: "Jobs Created", value: "15,420", trend: "up" },
              ].map((metric, index) => (
                <div
                  key={index}
                  className="flex items-center justify-between p-3 bg-gray-50 rounded-lg"
                >
                  <span className="text-sm text-gray-600">{metric.label}</span>
                  <div className="flex items-center space-x-2">
                    <span className="font-semibold text-gray-900">
                      {metric.value}
                    </span>
                    <TrendingUp
                      className={`h-3 w-3 ${
                        metric.trend === "up"
                          ? "text-emerald-500"
                          : "text-red-500 rotate-180"
                      }`}
                    />
                  </div>
                </div>
              ))}
            </div>
          </div>

          {/* Top Sites */}
          <div className="bg-white/95 backdrop-blur-md border border-white/20 rounded-xl shadow-lg p-6">
            <div className="flex items-center space-x-3 mb-6">
              <div className="p-2 bg-gradient-to-br from-indigo-500 to-purple-600 rounded-lg">
                <Activity className="h-5 w-5 text-white" />
              </div>
              <h3 className="text-lg font-bold text-gray-900">Top Sites</h3>
            </div>
            <div className="space-y-3">
              {[
                { name: "Gandhinagar-1", state: "Gujarat", score: 97.1 },
                { name: "Chennai-3", state: "Madhya Pradesh", score: 96.1 },
                { name: "Pune-2", state: "Maharashtra", score: 97.7 },
                { name: "Jodhpur-1", state: "Rajasthan", score: 96.9 },
              ].map((site, index) => (
                <div
                  key={index}
                  className="flex items-center justify-between p-3 bg-gray-50 rounded-lg"
                >
                  <div>
                    <div className="font-medium text-gray-900 text-sm">
                      {site.name}
                    </div>
                    <div className="text-xs text-gray-500">{site.state}</div>
                  </div>
                  <div
                    className={`px-2 py-1 rounded-full text-xs font-medium ${
                      site.score >= 90
                        ? "bg-emerald-100 text-emerald-700"
                        : "bg-yellow-100 text-yellow-700"
                    }`}
                  >
                    {site.score}%
                  </div>
                </div>
              ))}
            </div>
          </div>
        </div>

        {/* Cost Analysis */}
        <div className="bg-white/95 backdrop-blur-md border border-white/20 rounded-xl shadow-lg p-6 mb-8">
          <div className="flex items-center justify-between mb-6">
            <div className="flex items-center space-x-3">
              <div className="p-2 bg-gradient-to-br from-emerald-500 to-green-600 rounded-lg">
                <DollarSign className="h-5 w-5 text-white" />
              </div>
              <h3 className="text-lg font-bold text-gray-900">
                LCOH Trends & Cost Analysis
              </h3>
            </div>
          </div>

          <div className="grid grid-cols-1 lg:grid-cols-2 gap-8">
            {/* LCOH Chart */}
            <div>
              <h4 className="text-sm font-semibold text-gray-700 mb-4">
                Cost Reduction Over Time
              </h4>
              <div className="h-48">
                <ResponsiveContainer width="100%" height="100%">
                  <LineChart data={monthlyData}>
                    <CartesianGrid strokeDasharray="3 3" stroke="#e5e7eb" />
                    <XAxis dataKey="month" stroke="#6b7280" fontSize={12} />
                    <YAxis stroke="#6b7280" fontSize={12} />
                    <Tooltip />
                    <Line
                      type="monotone"
                      dataKey="cost"
                      stroke="#8b5cf6"
                      strokeWidth={3}
                      dot={{ fill: "#8b5cf6", r: 4 }}
                    />
                  </LineChart>
                </ResponsiveContainer>
              </div>
            </div>

            {/* Cost Breakdown */}
            <div>
              <h4 className="text-sm font-semibold text-gray-700 mb-4">
                Cost Components
              </h4>
              <div className="space-y-3">
                {[
                  {
                    component: "Electrolyzer",
                    percentage: 42,
                    color: "bg-emerald-500",
                  },
                  {
                    component: "Solar PV",
                    percentage: 27,
                    color: "bg-yellow-500",
                  },
                  {
                    component: "Wind Turbine",
                    percentage: 17,
                    color: "bg-cyan-500",
                  },
                  {
                    component: "Infrastructure",
                    percentage: 14,
                    color: "bg-purple-500",
                  },
                ].map((item, index) => (
                  <div key={index} className="space-y-1">
                    <div className="flex justify-between text-sm">
                      <span className="text-gray-600">{item.component}</span>
                      <span className="font-semibold">{item.percentage}%</span>
                    </div>
                    <div className="w-full bg-gray-200 rounded-full h-2">
                      <div
                        className={`h-2 rounded-full ${item.color} transition-all duration-1000`}
                        style={{ width: `${item.percentage}%` }}
                      ></div>
                    </div>
                  </div>
                ))}
              </div>
            </div>
          </div>
        </div>

        {/* Status Footer */}
        <div className="bg-white/95 backdrop-blur-md border border-white/20 rounded-xl shadow-lg p-4">
          <div className="flex items-center justify-between">
            <div className="flex items-center space-x-4">
              <div className="flex items-center space-x-2">
                <div className="w-2 h-2 bg-emerald-500 rounded-full animate-pulse"></div>
                <span className="text-sm font-medium text-gray-700">
                  System Online
                </span>
              </div>
              <div className="text-sm text-gray-500">
                Last updated: {new Date().toLocaleTimeString()}
              </div>
            </div>

            <div className="flex items-center space-x-3">
              <div className="text-sm text-gray-600">Network Health: 99.7%</div>
              <div className="w-1 h-6 bg-gradient-to-t from-emerald-400 to-emerald-600 rounded-full"></div>
            </div>
          </div>
        </div>
      </div>
    </div>
  );
};

export default HydrogenDashboard;
