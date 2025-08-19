import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.metrics import mean_absolute_error, r2_score
import joblib
import json

class ChessOpeningRecommender:
    def __init__(self):
        self.model_white = None
        self.model_black = None
        self.opening_encoder = LabelEncoder()
        self.scaler = StandardScaler()
        self.opening_stats = {}
        
    def prepare_data(self, df):
        """Prepare data for training"""
        # Debug: Print data info
        print(f"📊 Original data shape: {df.shape}")
        
        # Fix column names - your data uses 'Colour' but with lowercase values
        if 'Colour' in df.columns:
            # Capitalize the color values: white -> White, black -> Black
            df['Colour'] = df['Colour'].str.capitalize()
            print("✅ Fixed color capitalization: white->White, black->Black")
        
        # Check required columns
        required_cols = ['Opening', 'Perf Rating', 'Avg Player', 'Player Win %', 'Colour', 'Num Games', 'Draw %']
        missing_cols = [col for col in required_cols if col not in df.columns]
        
        if missing_cols:
            print(f"❌ Missing columns: {missing_cols}")
            print(f"Available columns: {df.columns.tolist()}")
        
        # Clean and prepare features - handle missing values better
        df_clean = df.dropna(subset=['Opening', 'Perf Rating', 'Player Win %', 'Colour'])
        
        # Fill missing values for other columns
        df_clean['Num Games'] = df_clean['Num Games'].fillna(100)  # Default games
        df_clean['Avg Player'] = df_clean['Avg Player'].fillna(df_clean['Perf Rating'])  # Use Perf Rating as fallback
        df_clean['Draw %'] = df_clean['Draw %'].fillna(25.0)  # Default draw rate
        
        print(f"📊 After cleaning: {df_clean.shape}")
        
        # Create separate datasets for white and black
        white_data = df_clean[df_clean['Colour'] == 'White'].copy()
        black_data = df_clean[df_clean['Colour'] == 'Black'].copy()
        
        print(f"⚪ White data: {len(white_data)} games")
        print(f"⚫ Black data: {len(black_data)} games")
        
        if len(white_data) == 0:
            print("❌ No White games found! Check your 'Colour' column values")
            print("Unique values:", df_clean['Colour'].unique())
            
        if len(black_data) == 0:
            print("❌ No Black games found! Check your 'Colour' column values")
            print("Unique values:", df_clean['Colour'].unique())
        
        return white_data, black_data
    
    def engineer_features(self, data):
        """Create features for ML model"""
        features = pd.DataFrame()
        
        # Encode openings
        features['opening_encoded'] = self.opening_encoder.fit_transform(data['Opening'])
        features['perf_rating'] = data['Perf Rating']
        features['num_games'] = data['Num Games']
        features['avg_player_rating'] = data['Avg Player']
        features['draw_rate'] = data['Draw %']
        
        # Target variable
        target = data['Player Win %']
        
        return features, target
    
    def train_models(self, df):
        """Train separate models for white and black pieces"""
        white_data, black_data = self.prepare_data(df)
        
        # Train model for white pieces
        X_white, y_white = self.engineer_features(white_data)
        X_white_scaled = self.scaler.fit_transform(X_white)
        
        self.model_white = RandomForestRegressor(n_estimators=100, random_state=42)
        self.model_white.fit(X_white_scaled, y_white)
        
        # Train model for black pieces (refit encoders)
        X_black, y_black = self.engineer_features(black_data)
        X_black_scaled = self.scaler.fit_transform(X_black)
        
        self.model_black = RandomForestRegressor(n_estimators=100, random_state=42)
        self.model_black.fit(X_black_scaled, y_black)
        
        # Store opening statistics for recommendations
        self.create_opening_stats(df)
        
        print(f"Models trained!")
        print(f"White pieces dataset: {len(white_data)} games")
        print(f"Black pieces dataset: {len(black_data)} games")
        
        return self
    
    def create_opening_stats(self, df):
        """Create opening statistics for recommendations"""
        stats = df.groupby(['Opening', 'Colour']).agg({
            'Player Win %': 'mean',
            'Num Games': 'sum',
            'Perf Rating': 'mean',
            'Draw %': 'mean',
            'Moves': 'first'
        }).reset_index()
        
        # Convert to dictionary for easy lookup
        for _, row in stats.iterrows():
            key = f"{row['Opening']}_{row['Colour']}"
            self.opening_stats[key] = {
                'opening': row['Opening'],
                'colour': row['Colour'],
                'avg_win_rate': round(row['Player Win %'], 1),
                'total_games': int(row['Num Games']),
                'avg_rating': round(row['Perf Rating'], 0),
                'draw_rate': round(row['Draw %'], 1),
                'moves': row['Moves']
            }
    
    def recommend_openings(self, user_rating, colour='White', top_n=5):
        """Recommend best openings for a user"""
        recommendations = []
        
        # Filter openings by colour
        relevant_openings = {k: v for k, v in self.opening_stats.items() 
                           if v['colour'] == colour}
        
        # Score openings based on:
        # 1. Win rate
        # 2. Popularity (number of games)
        # 3. Rating compatibility
        
        for key, stats in relevant_openings.items():
            # Rating compatibility score (closer to user rating = better)
            rating_diff = abs(stats['avg_rating'] - user_rating)
            rating_score = max(0, 1 - rating_diff / 500)  # Normalize rating difference
            
            # Popularity score (more games = more reliable)
            popularity_score = min(1, stats['total_games'] / 1000)  # Cap at 1000 games
            
            # Win rate score
            win_rate_score = stats['avg_win_rate'] / 100
            
            # Combined score
            total_score = (win_rate_score * 0.4 + 
                          rating_score * 0.4 + 
                          popularity_score * 0.2)
            
            recommendations.append({
                'opening': stats['opening'],
                'win_rate': stats['avg_win_rate'],
                'total_games': stats['total_games'],
                'moves': stats['moves'],
                'score': total_score,
                'rating_match': round((1 - rating_diff / 500) * 100, 1)
            })
        
        # Sort by score and return top N
        recommendations.sort(key=lambda x: x['score'], reverse=True)
        return recommendations[:top_n]
    
    def save_model(self, filepath='chess_model'):
        """Save trained model and encoders"""
        model_data = {
            'model_white': self.model_white,
            'model_black': self.model_black,
            'opening_encoder': self.opening_encoder,
            'scaler': self.scaler,
            'opening_stats': self.opening_stats
        }
        joblib.dump(model_data, f'{filepath}.pkl')
        print(f"Model saved to {filepath}.pkl")
    
    def load_model(self, filepath='chess_model.pkl'):
        """Load trained model"""
        model_data = joblib.load(filepath)
        self.model_white = model_data['model_white']
        self.model_black = model_data['model_black']
        self.opening_encoder = model_data['opening_encoder']
        self.scaler = model_data['scaler']
        self.opening_stats = model_data['opening_stats']
        return self

# Training script
if __name__ == "__main__":
    # Load your data
    df = pd.read_csv("openings.csv")
    
    # Initialize and train model
    recommender = ChessOpeningRecommender()
    recommender.train_models(df)
    
    # Save the model
    recommender.save_model('chess_opening_model')
    
    # Test the recommender
    print("\n=== Testing Recommendations ===")
    print("For 1200 rating player (White):")
    white_recs = recommender.recommend_openings(1200, 'White', 3)
    for i, rec in enumerate(white_recs, 1):
        print(f"{i}. {rec['opening']}")
        print(f"   Win Rate: {rec['win_rate']}% | Games: {rec['total_games']} | Rating Match: {rec['rating_match']}%")
        print(f"   Moves: {rec['moves']}")
    
    print("\nFor 1200 rating player (Black):")
    black_recs = recommender.recommend_openings(1200, 'Black', 3)
    for i, rec in enumerate(black_recs, 1):
        print(f"{i}. {rec['opening']}")
        print(f"   Win Rate: {rec['win_rate']}% | Games: {rec['total_games']} | Rating Match: {rec['rating_match']}%")
        print(f"   Moves: {rec['moves']}")