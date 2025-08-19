# Chess Opening Recommender AI

🏆 **UzChess AI Hackathon 2025 Project**

An AI-powered chess opening recommendation system that suggests the best opening moves based on your chess rating and playing style.

## 🚀 Features

- **Personalized Recommendations**: AI analyzes your rating to suggest optimal openings
- **Dual Color Support**: Get recommendations for both White and Black pieces  
- **Real Data**: Trained on thousands of real chess games
- **Interactive Web Interface**: Clean, responsive design
- **Instant Results**: Fast ML predictions in real-time

## 🛠 Tech Stack

- **Backend**: Flask (Python)
- **ML**: scikit-learn, Random Forest
- **Frontend**: HTML5, Tailwind CSS, Vanilla JS
- **Data**: Pandas for processing chess game data
- **Deployment**: Heroku/GitHub Pages

## 📊 Model Details

Our ML model uses:
- **Algorithm**: Random Forest Regressor
- **Features**: Player rating, opening popularity, historical win rates
- **Training Data**: 1,884 chess opening variations
- **Accuracy**: Optimized for rating-appropriate recommendations

## 🎯 How It Works

1. **Input**: Enter your chess rating (400-3000)
2. **Select**: Choose White or Black pieces
3. **AI Analysis**: Model processes your profile against opening database
4. **Results**: Get top-ranked openings with win rates and move sequences

## 🏃‍♂️ Quick Start

```bash
# Clone the repository
git clone https://github.com/yourusername/chess-opening-recommender.git
cd chess-opening-recommender

# Install dependencies
pip install -r requirements.txt

# Train the model (make sure openings.csv is in the directory)
python train_model.py

# Run the app
python app.py
```

Visit `http://localhost:5000` to use the app!

## 📈 Results Preview

The system provides:
- **Opening Name**: Full chess opening name
- **Win Rate**: Percentage success rate
- **Popularity**: Number of games in database  
- **Rating Match**: How well it fits your skill level
- **Move Sequence**: Actual chess moves to play

## 🎪 Demo for Judges

**Live Demo**: [Your deployed URL]

**Key Innovation**: Unlike static opening books, our AI personalizes recommendations based on individual player strength, making chess learning more effective.

## 👥 Team

Built for UzChess AI Hackathon 2025

---
*Ready to improve your chess game with AI? Try it now!* ♟️🤖