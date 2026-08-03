# 🎓 EduSip — Sip Knowledge & Level Up

**EduSip** is an adaptive learning web platform that personalizes quiz difficulty in real time based on a student's actual behaviour — not just whether they got the last answer right. It tracks response time, answer correctness, and hint usage, feeds these signals into a trained machine learning model, and continuously recalibrates each student's next question to keep them in their optimal challenge zone.

> "Learn like you were born for it." — *It's not just practice. It adapts to you.*

---

## 📌 Problem Statement

Traditional learning platforms use a one-size-fits-all question flow: some questions feel too easy and boring, others frustratingly hard. Students lose focus, waste time repeating familiar problems, and swing between motivation and discouragement. EduSip's adaptive engine adjusts each question to match the student's real skill level in real time, keeping them challenged, confident, and consistently engaged.

---

## ✨ Key Features

- **Real-time adaptive difficulty** — every answer updates the model's prediction for the next question's difficulty.
- **Smart difficulty scaling** — too easy? It ramps up. Too hard? It adapts down — no repetition, no wasted effort.
- **AI-generated explanations** — a Groq-powered LLM (LLaMA 3) explains *why* an answer is right or wrong in plain language.
- **Performance insights dashboard** — streaks, accuracy trends, subject-wise strengths/weaknesses, difficulty progression charts.
- **Flexible subject/topic selection** — students pick a category (e.g. Digital System Design, Data Structures, Java, Software Engineering) and topic before starting a quiz.
- **Quiz review mode** — revisit past attempts with AI explanations for missed questions.
- **Admin workspace** — manual question entry, question bank analytics (totals & difficulty distribution per subject).
- **Anti-cheating / proctoring system**:
  - Browser & tab control — fullscreen enforcement, tab/window switch detection, copy-paste blocking, keyboard shortcut lockdown
  - Webcam-based face & object detection — flags multiple faces or forbidden objects (phones, books, papers)
  - Audio/speech detection — voice activity detection to flag talking during a session
  - Violation logging with automatic quiz termination past a threshold

---

## 🏗️ Architecture

```
React Frontend  ──API request──▶  Node.js Backend (Express)  ──HTTP request──▶  Python ML API (FastAPI)
      ▲                                    │      ▲                                    │
      │                                    │      └──────────return prediction─────────┘
      └────────────receives next question──┘
```

**Flow example:**
1. User answers a question
2. Frontend sends the response data (correctness, response time, hint usage) to the backend
3. Backend stores the response
4. Backend calls the ML API to predict the next difficulty
5. ML model returns the predicted difficulty
6. Frontend receives and displays the next question

---

## 🧩 Tech Stack

| Layer | Technology |
|---|---|
| **Frontend** | React.js, HTML, CSS, JavaScript |
| **Backend** | Node.js + Express.js (REST APIs), JWT authentication |
| **ML Service** | Python, scikit-learn, XGBoost, TensorFlow/Keras, served via FastAPI |
| **Database** | MongoDB (NoSQL) |
| **LLM** | Groq API (LLaMA 3) for explanations & weak-topic identification |
| **Proctoring** | exam-guard, MediaPipe Tasks Vision (face/object detection), `@ricky0123/vad-web` (voice activity detection), MediaDevices API, Page Visibility API |

### Frontend
- Built with React.js — component-based architecture for fast, reusable UI elements
- Dynamic content updates without full page reloads
- Clean, modern, mobile-friendly, responsive dashboard UI

### Backend
- Node.js + Express.js powers the main server and REST APIs
- **Authentication:** JWT-based secure login/signup, token-based session management, protected admin/private routes
- **Question management:** upload (admin), fetch, and filter questions by subject/topic/difficulty
- **User data handling:** profiles, learning progress, difficulty levels, attempt history
- **Performance tracking:** records response time, correctness, and topic performance — the raw signal for the ML model

### Machine Learning
- **Goal:** real-time adaptive difficulty adjustment based on student behaviour
- **Models compared:** Random Forest, XGBoost, and Linear Regression (best model selected by MAE / R²)
- **Deep learning:** RNN/LSTM (TensorFlow/Keras) explored for capturing behavioural patterns over time
- **Input features:** response time, answer correctness, hint usage, current difficulty level
- **Output:** predicted difficulty score mapped to Easy / Medium / Hard
- **Training dataset:** Junyi Academy educational dataset
- **Deployment:** FastAPI REST endpoints integrated with the Node.js/Express backend

Example model comparison from evaluation:

| Model | MAE | R² |
|---|---|---|
| Linear Regression | 0.1020 | 0.7565 |
| Random Forest | 0.0977 | 0.7571 |
| XGBoost | 0.0978 | 0.7582 |

---

## 🗄️ Database Schema (MongoDB)

**Users Collection**
- `name` — user's full name
- `email` — login email
- `password` — encrypted (bcrypt)

**Questions Collection**
- `subject`, `topic` — categorization
- `difficulty` — Easy / Medium / Hard
- `options` — multiple-choice options
- `correctAnswer` — correct option index

**Responses Collection**
- `userId`, `questionId`
- `correctness` — 0 or 1
- `responseTime` — time taken to answer
- `timestamp`
- `hintUsed` — whether a hint was requested

**QuizAttempts Collection**
- Full record of each quiz session, enabling progress tracking across attempts

---

## 🖥️ App Pages / UI

- Landing page
- Sign up / Login (student & admin roles, Google OAuth option)
- Student dashboard (streaks, accuracy, weak/strong subjects, charts, AI feedback)
- Quiz interface (subject/topic selection → adaptive question flow)
- Quiz review (per-question breakdown with AI explanations)
- Admin workspace (manual question entry + question bank analytics)

---

## 🔐 Security & Standards

- JWT for stateless authentication; bcrypt for password hashing (OWASP-aligned)
- Middleware-level route protection for admin/private endpoints
- RESTful API design following standard HTTP methods and JSON response conventions
- Responsive design per W3C guidelines; tested across Chrome, Firefox, and Edge
- Test coverage across registration, login, quiz flow, difficulty adjustment, LLM explanations, proctoring, and admin functions (see full project report for the detailed test matrix)

---

## 🚀 Getting Started

> Note: exact setup commands depend on the repository layout (frontend / backend / ML service). At a high level:

```bash
# Clone the repo
git clone https://github.com/ShubhayanBhattacharjee/Adaptive-Learning-Web-App-Using-Behaviour-Based-Difficulty-Adjustment.git
cd Adaptive-Learning-Web-App-Using-Behaviour-Based-Difficulty-Adjustment

# Backend (Node/Express)
cd backend
npm install
npm start

# ML Service (FastAPI)
cd ../ml-service
pip install -r requirements.txt
uvicorn app:app --reload

# Frontend (React)
cd ../frontend
npm install
npm start
```

Set up a `.env` with your MongoDB URI, JWT secret, and Groq API key before running.

---

## 🗺️ Future Scope

- Deep learning (Transformer/LSTM) models to capture behavioural patterns across multiple sessions
- Expanded subject/curriculum coverage (CBSE, NCERT, university syllabi)
- Emotion/engagement detection via facial expression recognition
- Fully personalized learning paths, not just per-question difficulty
- Gamification — streaks, leaderboards, badges
- Offline-capable mobile app / PWA
- Multilingual support for questions and AI explanations
- Teacher-facing analytics dashboard
- Privacy-preserving federated learning for the adaptive model

---

## 👥 Team

| Name | Role |
|---|---|
| Priyangshu Saha | AI/ML |
| Saarnab Bishayee | AI/ML |
| Akashdeep Sengupta | Anti-Cheat |
| Trayambak Sarkar | Frontend |
| Shubham Chaudhary | Frontend |
| Shubhayan Bhattacharjee | Full Stack |

**Guided by:** Dr. Ajit Kumar Pasayat, School of Computer Engineering, KIIT Deemed to be University

Submitted in partial fulfilment of the requirements for the Bachelor's Degree in Computer Science & Engineering, KIIT, Bhubaneswar (2025–2026).
