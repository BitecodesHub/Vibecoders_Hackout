import { Routes, Route } from "react-router-dom";
import GreenHydrogenHome from "./components/GreenHydrogenHome";
import MapComponent from "./components/MapComponent";
import HydrogenDashboard from "./components/HydrogenDashboard";

function App() {
  return (
    <Routes>
      <Route path="/" element={<GreenHydrogenHome />} />
      <Route path="/map" element={<MapComponent />} />
      <Route path="/stats" element={<HydrogenDashboard />} />
    </Routes>
  );
}

export default App;
