# 🌱 Vibecoders Hackout – Green Hydrogen Infrastructure Mapping & Optimization

**Smarter Site Selection for Maximum ROI in Green Hydrogen**  
A full-stack AI-powered platform designed to revolutionize green hydrogen project planning by enabling data-driven site selection, maximizing ROI, and accelerating the global clean energy transition.

---

## 📌 Project Description

The **green hydrogen sector** faces a paradox: it demands **immense investments**, but poor site selection often leads to **suboptimal returns** and **billions in wasted capital**.  

Our solution is an **AI-powered decision intelligence platform** that empowers **energy executives, infrastructure planners, and policymakers** to make smarter, evidence-based investment choices.  

By combining **AI/ML modeling, geospatial data, interactive mapping, and scalable cloud-based backend services**, this platform ensures **optimized site selection** and accelerates the adoption of **sustainable hydrogen infrastructure**.

---

## 🔑 Core Features

### 🛰️ Precision Planning Tools
- **Location Feasibility Check**: Enter coordinates to instantly assess ROI potential.  
- **Interactive Map Layers**: Visualize hydrogen plants, pipelines, storage, renewable energy sources, and demand hubs.  
- **Smart Recommendations**: AI suggests top-performing investment zones.  
- **Optimization Criteria**: Tailor analysis based on demand, cost, regulation, and logistics.  

### 📊 Benefits
- **Reduces Investment Risks** – Mitigate exposure with data-backed site selection.  
- **Accelerates Adoption** – Fast-tracks hydrogen projects globally.  
- **Aligns with Renewables** – Ensures integration with solar/wind for efficiency.  
- **Evidence-Based Planning** – Provides policymakers with robust insights.  

---

## 🏗️ Tech Stack

| Layer        | Technology |
|--------------|------------|
| **Frontend** | React.js (Interactive mapping interface) |
| **Backend**  | Spring Boot (Scalable, secure REST API services) |
| **Database** | PostgreSQL + PostGIS (Geospatial queries & site data) |
| **AI/ML**    | Python (Predictive ROI model, feasibility analysis) |
| **Hosting**  | Docker / Cloud-ready deployment |

---

## ⚙️ Project Structure

\`\`\`
Vibecoders_Hackout/
│
├── Ai Model/       # Machine learning models for ROI & site feasibility
├── Back End/       # Spring Boot backend (REST APIs, business logic)
├── Front End/      # React frontend (UI, interactive maps, dashboards)
└── README.md       # Documentation
\`\`\`

---

## 🚀 Getting Started

### 1️⃣ Prerequisites
- **Java 17+** (for Spring Boot backend)  
- **Maven/Gradle** (dependency management)  
- **Node.js 16+ & npm** (for frontend)  
- **Python 3.9+** (for AI model training & inference)  
- **PostgreSQL + PostGIS** (database with spatial extensions)  
- **Docker** (optional for containerized deployment)  

---

### 2️⃣ Installation & Setup

#### Clone the Repository
\`\`\`bash
git clone https://github.com/BitecodesHub/Vibecoders_Hackout.git
cd Vibecoders_Hackout
\`\`\`

---

#### 🔹 Backend (Spring Boot)
1. Navigate to backend folder:
   \`\`\`bash
   cd Back\ End
   \`\`\`
2. Configure database in \`application.properties\`:
   \`\`\`properties
   spring.datasource.url=jdbc:postgresql://localhost:5432/hydrogen_db
   spring.datasource.username=postgres
   spring.datasource.password=your_password
   spring.jpa.hibernate.ddl-auto=update
   \`\`\`
3. Build & run:
   \`\`\`bash
   mvn spring-boot:run
   \`\`\`
   Backend will start at: **http://localhost:8080**

---

#### 🔹 Frontend (React)
1. Navigate to frontend folder:
   \`\`\`bash
   cd Front\ End
   \`\`\`
2. Install dependencies:
   \`\`\`bash
   npm install
   \`\`\`
3. Start development server:
   \`\`\`bash
   npm run dev
   \`\`\`
   Frontend will run at: **http://localhost:3000**

---

#### 🔹 AI Model
1. Navigate to AI folder:
   \`\`\`bash
   cd Ai\ Model
   \`\`\`
2. Create virtual environment & install dependencies:
   \`\`\`bash
   python -m venv venv
   source venv/bin/activate   # (Linux/Mac)
   venv\Scripts\activate      # (Windows)

   pip install -r requirements.txt
   \`\`\`
3. Run training/inference:
   \`\`\`bash
   python train.py
   python predict.py --lat <LAT> --lon <LON>
   \`\`\`

---

## 🧩 API Endpoints (Spring Boot)

| Method | Endpoint              | Description |
|--------|-----------------------|-------------|
| `POST` | `/api/sites/nearest`  | Find nearest feasible hydrogen site |
| `POST` | `/api/sites/predict`  | Get ROI prediction for coordinates |
| `GET`  | `/api/sites/all`      | Fetch all stored site data |
| `GET`  | `/api/health`         | Check backend health |

---

## 📌 Future Scope
- Integration with **real-time energy market data**  
- **Policy & regulatory frameworks** embedded into analysis  
- **Mobile app support** for field engineers  
- **Advanced optimization models** (supply chain, cost, emissions)  

---

## 📜 License
This project is licensed under the MIT License – free to use, modify, and distribute.  

---

## 🤝 Contributing
We welcome contributions! Please fork the repo, create a feature branch, and submit a PR.  

---

## 📬 Contact
**Team Vibecoders** – HackOut 2025 Submission  
📧 Email: your-team-email@example.com  
🌍 Repo: [Vibecoders_Hackout](https://github.com/BitecodesHub/Vibecoders_Hackout)  
