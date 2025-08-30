import { Routes, Route } from "react-router-dom";
import GreenHydrogenHome from "./components/GreenHydrogenHome";
import MapComponent from "./components/MapComponent";

function App() {
  return (
    <Routes>
      <Route path="/" element={<GreenHydrogenHome />} />
      <Route path="/map" element={<MapComponent />} />
    </Routes>
  );
}

export default App;
