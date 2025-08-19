from flask import Flask, request, jsonify, render_template
import joblib
import os
import json

app = Flask(__name__)

# Global variable to store the model
recommender = None

class ChessOpeningRecommender:
    def __init__(self):
        self.opening_stats = {}
        
    def load_model(self, filepath='chess_opening_model.pkl'):
        """Load trained model"""
        model_data = joblib.load(filepath)
        self.opening_stats = model_data['opening_stats']
        return self
    
    def recommend_openings(self, user_rating, colour='White', top_n=5):
        """Recommend best openings for a user"""
        recommendations = []
        
        # Filter openings by colour
        relevant_openings = {k: v for k, v in self.opening_stats.items() 
                           if v['colour'] == colour}
        
        for key, stats in relevant_openings.items():
            # Rating compatibility score
            rating_diff = abs(stats['avg_rating'] - user_rating)
            rating_score = max(0, 1 - rating_diff / 500)
            
            # Popularity score
            popularity_score = min(1, stats['total_games'] / 1000)
            
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
                'rating_match': round((1 - rating_diff / 500) * 100, 1),
                'difficulty': 'Beginner' if stats['avg_rating'] < 1400 else 'Intermediate' if stats['avg_rating'] < 1800 else 'Advanced'
            })
        
        # Sort by score and return top N
        recommendations.sort(key=lambda x: x['score'], reverse=True)
        return recommendations[:top_n]

def load_model():
    """Load the ML model on startup"""
    global recommender
    try:
        recommender = ChessOpeningRecommender()
        recommender.load_model('chess_opening_model.pkl')
        print("Model loaded successfully!")
        return True
    except Exception as e:
        print(f"Error loading model: {e}")
        return False

@app.route('/')
def home():
    """Main page"""
    return render_template('index.html')

@app.route('/api/recommend', methods=['POST'])
def get_recommendations():
    """API endpoint for getting opening recommendations"""
    try:
        data = request.get_json()
        
        # Validate input
        if not data or 'rating' not in data:
            return jsonify({'error': 'Rating is required'}), 400
        
        rating = int(data['rating'])
        colour = data.get('colour', 'White')
        count = min(int(data.get('count', 5)), 10)  # Max 10 recommendations
        
        # Validate rating range
        if rating < 400 or rating > 3000:
            return jsonify({'error': 'Rating must be between 400 and 3000'}), 400
        
        if colour not in ['White', 'Black']:
            return jsonify({'error': 'Colour must be White or Black'}), 400
        
        # Get recommendations
        if recommender is None:
            return jsonify({'error': 'Model not loaded'}), 500
        
        recommendations = recommender.recommend_openings(rating, colour, count)
        
        return jsonify({
            'success': True,
            'user_rating': rating,
            'colour': colour,
            'recommendations': recommendations
        })
        
    except ValueError as e:
        return jsonify({'error': 'Invalid rating value'}), 400
    except Exception as e:
        return jsonify({'error': f'Server error: {str(e)}'}), 500

@app.route('/api/opening-stats')
def get_opening_stats():
    """Get general opening statistics"""
    if recommender is None:
        return jsonify({'error': 'Model not loaded'}), 500
    
    try:
        # Get top openings by popularity
        all_openings = []
        for key, stats in recommender.opening_stats.items():
            all_openings.append(stats)
        
        # Sort by total games
        all_openings.sort(key=lambda x: x['total_games'], reverse=True)
        
        # Get top 20 most popular
        popular_openings = all_openings[:20]
        
        return jsonify({
            'success': True,
            'total_openings': len(all_openings),
            'popular_openings': popular_openings
        })
        
    except Exception as e:
        return jsonify({'error': f'Server error: {str(e)}'}), 500

@app.route('/health')
def health_check():
    """Health check endpoint"""
    return jsonify({
        'status': 'healthy',
        'model_loaded': recommender is not None
    })

@app.errorhandler(404)
def not_found(error):
    return jsonify({'error': 'Endpoint not found'}), 404

@app.errorhandler(500)
def internal_error(error):
    return jsonify({'error': 'Internal server error'}), 500

if __name__ == '__main__':
    # Load model on startup
    model_loaded = load_model()
    
    if not model_loaded:
        print("WARNING: Running without model - some features won't work")
    
    # Run the app
    port = int(os.environ.get('PORT', 5000))
    app.run(host='0.0.0.0', port=port, debug=True)