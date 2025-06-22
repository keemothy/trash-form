# RecycleBnB - AI-powered recycling assistant
from flask import Flask, request, jsonify, render_template
from flask_cors import CORS
import tensorflow as tf
import numpy as np
from PIL import Image
import io
import requests
import json
from datetime import datetime, timedelta
import os
import anthropic
from pathlib import Path

# Load environment variables from .env file
def load_env():
    env_path = Path('.env')
    if env_path.exists():
        with open(env_path) as f:
            for line in f:
                if '=' in line and not line.startswith('#'):
                    key, value = line.strip().split('=', 1)
                    os.environ[key] = value

load_env()

app = Flask(__name__)
CORS(app)

# Initialize Claude API client
claude_client = None
CLAUDE_API_KEY = os.getenv('CLAUDE_API_KEY')

if CLAUDE_API_KEY:
    claude_client = anthropic.Anthropic(api_key=CLAUDE_API_KEY)
else:
    print("Warning: CLAUDE_API_KEY not found. Claude features will be disabled.")

# Load your trained model
model = tf.keras.models.load_model('../best_garbage_model.h5')

# Enhanced material classification
MATERIAL_MAPPING = {
    'battery_training_set': {
        'display_name': 'Battery',
        'icon': '🔋',
        'color': '#FF5A5F',
        'category': 'hazardous',
        'refund_potential': 'low'
    },
    'biological_training_set': {
        'display_name': 'Organic Waste',
        'icon': '🥬',
        'color': '#00A699',
        'category': 'compostable',
        'refund_potential': 'none'
    },
    'cardboard_training_set': {
        'display_name': 'Cardboard',
        'icon': '📦',
        'color': '#FC642D',
        'category': 'recyclable',
        'refund_potential': 'medium'
    },
    'clothes_training_set': {
        'display_name': 'Textiles',
        'icon': '👕',
        'color': '#484848',
        'category': 'donation',
        'refund_potential': 'low'
    },
    'glass_training_set': {
        'display_name': 'Glass',
        'icon': '🍷',
        'color': '#00A699',
        'category': 'recyclable',
        'refund_potential': 'high'
    },
    'metal_training_set': {
        'display_name': 'Metal',
        'icon': '🥫',
        'color': '#484848',
        'category': 'recyclable',
        'refund_potential': 'high'
    },
    'paper_training_set': {
        'display_name': 'Paper',
        'icon': '📄',
        'color': '#FC642D',
        'category': 'recyclable',
        'refund_potential': 'low'
    },
    'plastic_training_set': {
        'display_name': 'Plastic',
        'icon': '🥤',
        'color': '#FF5A5F',
        'category': 'recyclable',
        'refund_potential': 'medium'
    },
    'shoes_training_set': {
        'display_name': 'Footwear',
        'icon': '👟',
        'color': '#484848',
        'category': 'donation',
        'refund_potential': 'none'
    },
    'trash_training_set': {
        'display_name': 'General Waste',
        'icon': '🗑️',
        'color': '#767676',
        'category': 'landfill',
        'refund_potential': 'none'
    }
}

CLASS_NAMES = list(MATERIAL_MAPPING.keys())

def get_location_from_ip():
    """Get user location from IP address"""
    try:
        response = requests.get('http://ipapi.co/json/', timeout=5)
        if response.status_code == 200:
            data = response.json()
            return {
                'city': data.get('city', 'Unknown'),
                'region': data.get('region', 'Unknown'),
                'country': data.get('country_name', 'Unknown'),
                'lat': data.get('latitude', 0),
                'lng': data.get('longitude', 0)
            }
    except:
        pass
    
    # Default to Davis, California for testing
    return {
        'city': 'Davis',
        'region': 'California',
        'country': 'United States',
        'lat': 38.5449,
        'lng': -121.7405
    }

def get_location_by_city(city_name):
    """Get location coordinates by city name"""
    # Predefined locations for testing
    locations = {
        'davis': {
            'city': 'Davis',
            'region': 'California',
            'country': 'United States',
            'lat': 38.5449,
            'lng': -121.7405
        },
        'san francisco': {
            'city': 'San Francisco',
            'region': 'California', 
            'country': 'United States',
            'lat': 37.7749,
            'lng': -122.4194
        },
        'new york': {
            'city': 'New York',
            'region': 'New York',
            'country': 'United States',
            'lat': 40.7128,
            'lng': -74.0060
        },
        'seattle': {
            'city': 'Seattle',
            'region': 'Washington',
            'country': 'United States',
            'lat': 47.6062,
            'lng': -122.3321
        },
        'portland': {
            'city': 'Portland',
            'region': 'Oregon',
            'country': 'United States',
            'lat': 45.5152,
            'lng': -122.6784
        },
        'austin': {
            'city': 'Austin',
            'region': 'Texas',
            'country': 'United States',
            'lat': 30.2672,
            'lng': -97.7431
        },
        'denver': {
            'city': 'Denver',
            'region': 'Colorado',
            'country': 'United States',
            'lat': 39.7392,
            'lng': -104.9903
        }
    }
    
    city_key = city_name.lower().strip()
    return locations.get(city_key, locations['davis'])  # Default to Davis

def get_recycling_centers_with_claude(material_type, location_info):
    """Find real recycling centers using Claude API"""
    if not claude_client:
        # Fallback to simulated data if Claude is not available
        return [{
            'id': 'fallback_center',
            'name': 'Local Recycling Center',
            'address': f"{location_info['city']}, {location_info['region']}",
            'lat': location_info['lat'],
            'lng': location_info['lng'],
            'materials': [material_type.replace('_training_set', '')],
            'refund_rates': {material_type.replace('_training_set', ''): 0.05},
            'hours': 'Mon-Fri: 9AM-5PM',
            'phone': 'Contact local authorities',
            'distance': 0.5
        }]
    
    material_name = MATERIAL_MAPPING[material_type]['display_name']
    city = location_info['city']
    region = location_info['region']
    
    prompt = f"""Find real recycling centers in {city}, {region} that accept {material_name}. 
    
    Please provide a JSON response with up to 3 actual recycling centers with this exact format:
    [
        {{
            "name": "Center Name",
            "address": "Full address",
            "phone": "Phone number if available",
            "hours": "Operating hours",
            "materials_accepted": ["{material_name.lower()}"],
            "estimated_refund_rate": 0.0,
            "notes": "Any special instructions"
        }}
    ]
    
    Only include real, existing facilities. If you're not certain about a facility, don't include it."""
    
    try:
        message = claude_client.messages.create(
            model="claude-3-5-sonnet-20241022",
            max_tokens=1000,
            temperature=0.7,  # Add some randomness for varied center suggestions
            messages=[{"role": "user", "content": prompt}]
        )
        
        # Parse Claude's response
        response_text = message.content[0].text
        
        # Extract JSON from response
        import re
        json_match = re.search(r'\[.*\]', response_text, re.DOTALL)
        if json_match:
            centers_data = json.loads(json_match.group())
            
            # Convert to our format
            centers = []
            for i, center in enumerate(centers_data):
                centers.append({
                    'id': f'claude_center_{i}',
                    'name': center.get('name', 'Unknown Center'),
                    'address': center.get('address', f"{city}, {region}"),
                    'lat': location_info['lat'] + (i * 0.01),  # Slight offset for mapping
                    'lng': location_info['lng'] + (i * 0.01),
                    'materials': center.get('materials_accepted', [material_name.lower()]),
                    'refund_rates': {material_name.lower(): center.get('estimated_refund_rate', 0.0)},
                    'hours': center.get('hours', 'Contact for hours'),
                    'phone': center.get('phone', 'Contact local authorities'),
                    'notes': center.get('notes', ''),
                    'distance': round(0.5 + (i * 0.3), 1)  # Estimated distance
                })
            
            return centers
            
    except Exception as e:
        print(f"Claude API error: {e}")
    
    # Fallback if Claude fails
    return [{
        'id': 'fallback_center',
        'name': 'Local Recycling Center',
        'address': f"{city}, {region}",
        'lat': location_info['lat'],
        'lng': location_info['lng'],
        'materials': [material_name.lower()],
        'refund_rates': {material_name.lower(): 0.05},
        'hours': 'Mon-Fri: 9AM-5PM',
        'phone': 'Contact local authorities',
        'distance': 0.5
    }]

def get_sustainability_tips_with_claude(material_type):
    """Generate sustainability tips using Claude API"""
    if not claude_client:
        # Fallback tips if Claude is not available
        fallback_tips = {
            'battery_training_set': {
                'tip': 'Batteries contain toxic metals that can contaminate water supplies.',
                'action': 'Never throw batteries in regular trash - find a battery drop-off location.',
                'best_practice': 'Store used batteries in a cool, dry place until disposal.'
            },
            'plastic_training_set': {
                'tip': 'Most plastic items can be recycled, but cleanliness matters.',
                'action': 'Rinse containers before recycling to prevent contamination.',
                'best_practice': 'Check the recycling number - some types are more valuable.'
            },
            'glass_training_set': {
                'tip': 'Glass can be recycled infinitely without losing quality.',
                'action': 'Remove lids and rinse containers before recycling.',
                'best_practice': 'Separate by color when possible for better recycling rates.'
            }
        }
        
        return fallback_tips.get(material_type, {
            'tip': 'Every small action counts toward a more sustainable future.',
            'action': 'Check local regulations for proper disposal methods.',
            'best_practice': 'When in doubt, ask your local waste management facility.'
        })
    
    material_name = MATERIAL_MAPPING[material_type]['display_name']
    
    # Add randomness to get different tips each time
    import random
    perspectives = [
        "environmental impact", "economic benefits", "surprising facts", 
        "global statistics", "local impact", "innovation and technology"
    ]
    perspective = random.choice(perspectives)
    
    prompt = f"""Generate unique sustainability tips for {material_name} recycling with a focus on {perspective}. 
    
    Please provide a JSON response with this exact format:
    {{
        "tip": "A fascinating and specific 'Did you know?' fact about {material_name} that most people don't know",
        "action": "A concrete, actionable step the user should take before recycling this {material_name}",
        "best_practice": "An expert-level best practice tip for maximum environmental or economic impact"
    }}
    
    Make each tip unique, specific, and surprising. Avoid generic advice. Include specific numbers, statistics, or little-known facts when possible. Each response should feel fresh and educational."""
    
    try:
        message = claude_client.messages.create(
            model="claude-3-5-sonnet-20241022",
            max_tokens=500,
            temperature=0.8,  # Higher temperature for more creative/varied responses
            messages=[{"role": "user", "content": prompt}]
        )
        
        response_text = message.content[0].text
        
        # Extract JSON from response
        import re
        json_match = re.search(r'\{.*\}', response_text, re.DOTALL)
        if json_match:
            tips_data = json.loads(json_match.group())
            return tips_data
            
    except Exception as e:
        print(f"Claude API error for tips: {e}")
    
    # Fallback if Claude fails
    return {
        'tip': f'{material_name} recycling helps reduce environmental impact.',
        'action': 'Prepare the item according to local recycling guidelines.',
        'best_practice': 'Check with local authorities for the best disposal method.'
    }

@app.route('/')
def home():
    return render_template('index.html')

@app.route('/predict', methods=['POST'])
def predict():
    """Enhanced prediction endpoint with location-based recycling centers"""
    if 'image' not in request.files:
        return jsonify({'error': 'No image uploaded'}), 400
    
    file = request.files['image']
    
    try:
        # Image processing and prediction
        img = Image.open(file.stream).convert('RGB')
        img = img.resize((224, 224))
        img_array = np.array(img) / 255.0
        img_array = np.expand_dims(img_array, axis=0)
        preds = model.predict(img_array)
        
        # Get prediction results
        pred_class = CLASS_NAMES[np.argmax(preds[0])]
        confidence = float(np.max(preds[0]))
        
        # Get material info
        material_info = MATERIAL_MAPPING[pred_class]
        
        # Get user location
        selected_location = request.form.get('location', '').strip()
        if selected_location:
            location_info = get_location_by_city(selected_location)
        else:
            location_info = get_location_from_ip()
        
        # Get nearby recycling centers using Claude API
        centers = get_recycling_centers_with_claude(pred_class, location_info)
        
        # Get sustainability tips using Claude API
        tips = get_sustainability_tips_with_claude(pred_class)
        
        # Response structure
        response = {
            # 1. Identify & Confirm
            'identification': {
                'material': material_info['display_name'],
                'icon': material_info['icon'],
                'confidence': f"{confidence:.1%}",
                'category': material_info['category'],
                'color': material_info['color'],
                'message': f"Great! I've identified this as {material_info['display_name']} with {confidence:.1%} confidence."
            },
            
            # 2. Locate & Display
            'locations': {
                'nearest_centers': centers[:3],  # Top 3 nearest centers
                'map_data': {
                    'center_lat': centers[0]['lat'] if centers else location_info['lat'],
                    'center_lng': centers[0]['lng'] if centers else location_info['lng'],
                    'zoom': 13
                },
                'summary': f"Found {len(centers)} nearby centers accepting {material_info['display_name'].lower()}",
                'user_location': f"{location_info['city']}, {location_info['region']}"
            },
            
            # 3. Sustainability Insights
            'insights': {
                'tip': tips['tip'],
                'action_required': tips['action'],
                'best_practice': tips['best_practice']
            },
            
            # Additional features
            'features': {
                'refund_potential': material_info['refund_potential'],
                'can_schedule_pickup': material_info['category'] in ['hazardous', 'donation'],
                'weather_sensitive': material_info['category'] == 'recyclable'
            }
        }
        
        return jsonify(response)
        
    except Exception as e:
        return jsonify({'error': f'Prediction failed: {str(e)}'}), 500

@app.route('/centers/<material_type>')
def get_centers(material_type):
    """Get recycling centers for specific material type"""
    location_info = get_location_from_ip()
    centers = get_recycling_centers_with_claude(material_type, location_info)
    return jsonify({
        'centers': centers,
        'count': len(centers),
        'material': material_type,
        'location': f"{location_info['city']}, {location_info['region']}"
    })

@app.route('/schedule-pickup', methods=['POST'])
def schedule_pickup():
    """Schedule pickup for bulky items (future feature)"""
    data = request.get_json()
    
    # Simulate scheduling logic
    pickup_date = datetime.now() + timedelta(days=3)
    
    return jsonify({
        'scheduled': True,
        'pickup_date': pickup_date.strftime('%Y-%m-%d'),
        'confirmation_number': 'RBB-' + str(hash(data.get('address', '')))[0:6],
        'message': 'Pickup scheduled! You\'ll receive a confirmation email shortly.'
    })

@app.route('/rates/<material_type>')
def get_current_rates(material_type):
    """Get current refund rates for material type"""
    location_info = get_location_from_ip()
    centers = get_recycling_centers_with_claude(material_type, location_info)
    
    if not centers:
        return jsonify({'error': 'No centers found for this material'}), 404
    
    rates = []
    for center in centers:
        material_key = material_type.replace('_training_set', '').lower()
        if material_key in center['refund_rates'] and center['refund_rates'][material_key] > 0:
            rates.append(center['refund_rates'][material_key])
    
    return jsonify({
        'material': material_type,
        'average_rate': sum(rates) / len(rates) if rates else 0,
        'highest_rate': max(rates) if rates else 0,
        'centers_paying': len(rates),
        'rate_trend': 'stable',  # This would be calculated from historical data
        'location': f"{location_info['city']}, {location_info['region']}"
    })

@app.route('/health')
def health_check():
    """Health check endpoint"""
    return jsonify({
        'status': 'healthy',
        'model_loaded': model is not None,
        'timestamp': datetime.now().isoformat()
    })

if __name__ == '__main__':
    app.run(debug=True, port=9000)
