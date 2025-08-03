# 🥤 Shake Maker ML - AI-Powered Beverage Game

*A retro-inspired shake crafting game powered by neural networks and wrapped in warm, pastel café aesthetics*

## Game Preview 
<img width="1600" height="900" alt="image" src="https://github.com/user-attachments/assets/1e230e92-2a73-48bf-a407-ac709cf99398" />
<img width="1600" height="900" alt="image" src="https://github.com/user-attachments/assets/49550d44-b1f8-4566-beab-105073653dff" />
<img width="1600" height="900" alt="image" src="https://github.com/user-attachments/assets/fa1d6001-2f52-4174-974a-736d9f6e94b7" />
<img width="1600" height="886" alt="image" src="https://github.com/user-attachments/assets/f6cb083f-17f8-46b4-aab2-2b71f3dfbf96" />



## 🎨 The Vision

Inspired by classic café aesthetics and the nostalgia of simple yet engaging browser games, **Shake Maker ML** combines the charm of retro gaming with cutting-edge machine learning. The game features warm, pastel pink-maroon tones that evoke the cozy atmosphere of a neighborhood café, where every shake creation feels like a personalized culinary adventure.

## 🌟 What Makes It Special

### 🧠 Dual Neural Network Architecture
- **Predictor Model**: Analyzes ingredient combinations and predicts taste quality scores
- **Generator Model**: Acts as an AI chef, suggesting optimal ingredient combinations based on user preferences
- **Smart Fallback**: Rule-based predictions when ML models are unavailable

### 🎮 Engaging Game Mechanics
- **Category-Based Ingredient Selection**: Fruits, vegetables, cakes, and syrups
- **Real-time ML Predictions**: Neural network analyzes your shake combinations
- **AI Chef Suggestions**: Let the generative model create personalized recommendations
- **Visual Feedback**: Animated bubbles, confetti celebrations, and emoji reactions

### 🎵 Immersive Experience
- **Sound Design**: Click sounds, bubble pops, and result notifications
- **Visual Animations**: Smooth transitions, bubble effects, and celebratory confetti
- **Responsive Design**: Fullscreen toggle for immersive gameplay


## 🏗️ Technical Architecture
- **Neural Network Predictor # 36 ingredients + 6 features = 41 input neurons**
- **Generative Model # 6 preference inputs → ingredient probabilities**
- **Rule-based Fallback System**
- **Comprehensive API Endpoints**
- **Auto-healing Model Management**

### Backend (FastAPI + TensorFlow)
**Key Features:**
- **Dynamic Vector Creation**: Handles ingredient feature encoding automatically
- **Model Compatibility Checks**: Prevents version mismatches
- **Emergency Recovery**: Auto-rebuilding when models fail
- **Debug Endpoints**: Real-time model diagnostics

### Frontend (Vanilla JavaScript + CSS3)
**Visual Highlights:**
- **Warm Color Palette**: Pink, maroon, and pastel tones
- **Smooth Animations**: CSS3 transitions and JavaScript-powered effects
- **Bubble Physics**: Dynamic floating animation system
- **Responsive Layout**: Works across different screen sizes

## 🧪 The Machine Learning Magic

### Predictor Model

- **Input Layer**: 42 neurons
- **Hidden Layers**: 128 → 64 → 32 (ReLU + Dropout)
- **Output Layer**: 1 neuron (Sigmoid for score prediction)
- **Loss Function**: Mean Squared Error
- **Optimizer**: Adam (learning rate = 0.001)

## Design Philosophy
Our design philosophy is rooted in creating a user experience that is both intuitive and rewarding. We've focused on four key principles:

- **Progressive Disclosure**: Believing in unlocking features progressively. This approach prevents users from feeling overwhelmed and allows them to master core functionalities before exploring more advanced options.
-  **Immediate Feedback**: Every interaction is met with an immediate visual or audio reaction. This constant feedback loop ensures users always know the result of their actions, making the experience feel responsive and satisfying.
-  **Guided Discovery**: My **AI Chef** acts as a guide, helping users explore new combinations and possibilities. This feature encourages experimentation and creative freedom without the frustration of not knowing where to start.
-  **Celebration**: High-scoring shakes are a cause for celebration! We reward user success with fun, celebratory animations like confetti and cheers, making achievements feel special and encouraging continued engagement.

