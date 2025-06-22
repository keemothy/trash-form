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

# Enhanced material classification with realistic carbon footprint data
# Carbon data sources: EPA, Ellen MacArthur Foundation, Carbon Trust studies
MATERIAL_MAPPING = {
    'battery_training_set': {
        'display_name': 'Battery',
        'icon': '🔋',
        'color': '#FF5A5F',
        'category': 'hazardous',
        'refund_potential': 'low',
        'carbon_per_kg': {'saved': 6.2, 'landfill': 18.5},  # Li-ion recycling vs mining
        'typical_weight_kg': 0.05,  # typical AA battery weight
        'size_multipliers': {'small': 0.3, 'medium': 1.0, 'large': 3.0}  # phone vs car battery
    },
    'biological_training_set': {
        'display_name': 'Organic Waste',
        'icon': '🥬',
        'color': '#00A699',
        'category': 'compostable',
        'refund_potential': 'none',
        'carbon_per_kg': {'saved': 0.5, 'landfill': 1.8},  # Composting vs methane emissions
        'typical_weight_kg': 0.2,  # typical food scrap
        'size_multipliers': {'small': 0.5, 'medium': 1.0, 'large': 2.5}
    },
    'cardboard_training_set': {
        'display_name': 'Cardboard',
        'icon': '📦',
        'color': '#FC642D',
        'category': 'recyclable',
        'refund_potential': 'medium',
        'carbon_per_kg': {'saved': 1.1, 'landfill': 0.3},  # Recycled vs virgin paper
        'typical_weight_kg': 0.15,  # typical cardboard box
        'size_multipliers': {'small': 0.3, 'medium': 1.0, 'large': 4.0}
    },
    'clothes_training_set': {
        'display_name': 'Textiles',
        'icon': '👕',
        'color': '#484848',
        'category': 'donation',
        'refund_potential': 'low',
        'carbon_per_kg': {'saved': 3.6, 'landfill': 2.1},  # Reuse vs new textile production
        'typical_weight_kg': 0.4,  # typical t-shirt
        'size_multipliers': {'small': 0.4, 'medium': 1.0, 'large': 2.2}
    },  
    'glass_training_set': {
        'display_name': 'Glass',
        'icon': '🍷',
        'color': '#00A699',
        'category': 'recyclable',
        'refund_potential': 'high',
        'carbon_per_kg': {'saved': 0.31, 'landfill': 0.02},  # Glass recycling energy savings
        'typical_weight_kg': 0.35,  # typical glass bottle
        'size_multipliers': {'small': 0.3, 'medium': 1.0, 'large': 2.8}
    },
    'metal_training_set': {
        'display_name': 'Metal',
        'icon': '🥫',
        'color': '#484848',
        'category': 'recyclable',
        'refund_potential': 'high',
        'carbon_per_kg': {'saved': 1.47, 'landfill': 0.12},  # Aluminum: 95% energy savings
        'typical_weight_kg': 0.015,  # typical aluminum can (15g)
        'size_multipliers': {'small': 0.2, 'medium': 1.0, 'large': 5.0}
    },
    'paper_training_set': {
        'display_name': 'Paper',
        'icon': '📄',
        'color': '#FC642D',
        'category': 'recyclable',
        'refund_potential': 'low',
        'carbon_per_kg': {'saved': 1.38, 'landfill': 0.98},  # Paper recycling vs virgin pulp
        'typical_weight_kg': 0.08,  # typical stack of papers
        'size_multipliers': {'small': 0.3, 'medium': 1.0, 'large': 3.0}
    },
    'plastic_training_set': {
        'display_name': 'Plastic',
        'icon': '🥤',
        'color': '#FF5A5F',
        'category': 'recyclable',
        'refund_potential': 'medium',
        'carbon_per_kg': {'saved': 1.76, 'landfill': 3.14},  # PET recycling vs new production
        'typical_weight_kg': 0.03,  # typical plastic bottle (30g)
        'size_multipliers': {'small': 0.2, 'medium': 1.0, 'large': 4.0}
    },
    'shoes_training_set': {
        'display_name': 'Footwear',
        'icon': '👟',
        'color': '#484848',
        'category': 'donation',
        'refund_potential': 'none',
        'carbon_per_kg': {'saved': 2.8, 'landfill': 1.2},  # Shoe reuse vs new production
        'typical_weight_kg': 0.6,  # typical pair of shoes
        'size_multipliers': {'small': 0.6, 'medium': 1.0, 'large': 1.8}
    },
    'trash_training_set': {
        'display_name': 'General Waste',
        'icon': '🗑️',
        'color': '#767676',
        'category': 'landfill',
        'refund_potential': 'none',
        'carbon_per_kg': {'saved': 0.0, 'landfill': 0.67},  # Landfill methane emissions
        'typical_weight_kg': 0.1,  # various small items
        'size_multipliers': {'small': 0.3, 'medium': 1.0, 'large': 3.0}
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

# Simple analytics storage (in production, use a database)
import json
from datetime import datetime

ANALYTICS_FILE = 'recycling_analytics.json'

def load_analytics():
    """Load analytics data from file"""
    try:
        if os.path.exists(ANALYTICS_FILE):
            with open(ANALYTICS_FILE, 'r') as f:
                return json.load(f)
    except:
        pass
    return {
        'total_items': 0,
        'total_carbon_saved': 0.0,
        'total_carbon_avoided': 0.0,
        'items_by_category': {},
        'items_by_month': {},
        'sessions': []
    }

def save_analytics(data):
    """Save analytics data to file"""
    try:
        with open(ANALYTICS_FILE, 'w') as f:
            json.dump(data, f, indent=2)
    except Exception as e:
        print(f"Error saving analytics: {e}")

def update_analytics(material_type, location, estimated_size='medium'):
    """Update analytics with new recycling action including size estimation"""
    analytics = load_analytics()
    material_info = MATERIAL_MAPPING.get(material_type, {})
    
    # Calculate carbon impact with size consideration
    carbon_impact = calculate_carbon_impact_with_size(material_info, estimated_size)
    carbon_saved = carbon_impact['co2_saved_kg']
    carbon_avoided = carbon_impact['co2_avoided_kg']
    
    # Update totals
    analytics['total_items'] += 1
    analytics['total_carbon_saved'] = round(analytics['total_carbon_saved'] + carbon_saved, 2)
    analytics['total_carbon_avoided'] = round(analytics['total_carbon_avoided'] + carbon_avoided, 2)
    
    # Update by category
    category = material_info.get('category', 'unknown')
    if category not in analytics['items_by_category']:
        analytics['items_by_category'][category] = 0
    analytics['items_by_category'][category] += 1
    
    # Update by month
    current_month = datetime.now().strftime('%Y-%m')
    if current_month not in analytics['items_by_month']:
        analytics['items_by_month'][current_month] = 0
    analytics['items_by_month'][current_month] += 1
    
    # Add session data
    session = {
        'timestamp': datetime.now().isoformat(),
        'material': material_info.get('display_name', 'Unknown'),
        'category': category,
        'estimated_size': estimated_size,
        'estimated_weight_kg': carbon_impact['estimated_weight_kg'],
        'carbon_saved': round(carbon_saved, 2),
        'carbon_avoided': round(carbon_avoided, 2),
        'location': location
    }
    analytics['sessions'].append(session)
    
    # Keep only last 1000 sessions
    if len(analytics['sessions']) > 1000:
        analytics['sessions'] = analytics['sessions'][-1000:]
    
    save_analytics(analytics)
    return analytics

def calculate_carbon_impact(material_info):
    """Calculate carbon footprint impact for a material"""
    carbon_saved = material_info.get('carbon_saved_kg', 0)
    carbon_avoided = material_info.get('carbon_cost_landfill', 0)
    
    # Calculate equivalent comparisons
    car_miles_equivalent = (carbon_saved + carbon_avoided) * 2.31  # miles
    trees_equivalent = (carbon_saved + carbon_avoided) / 21.77  # trees per year
    
    return {
        'co2_saved_kg': round(carbon_saved, 2),
        'co2_avoided_kg': round(carbon_avoided, 2),
        'total_impact_kg': round(carbon_saved + carbon_avoided, 2),
        'car_miles_equivalent': round(car_miles_equivalent, 1),
        'trees_planted_equivalent': round(trees_equivalent, 2)
    }

def estimate_item_size_from_image(image_array, material_type):
    """
    Estimate item size category from image dimensions and material type
    This is a simplified estimation - in production, you'd use more sophisticated CV techniques
    """
    height, width = image_array.shape[:2]
    total_pixels = height * width
    
    # Get material-specific size thresholds
    material_info = MATERIAL_MAPPING.get(material_type, {})
    
    # Base size estimation on image resolution and material type
    # These thresholds are rough estimates and would need calibration with real data
    if material_type in ['battery_training_set', 'paper_training_set']:
        # Small items by nature
        if total_pixels < 50000:
            return 'small'
        elif total_pixels < 200000:
            return 'medium'
        else:
            return 'large'
    elif material_type in ['cardboard_training_set', 'plastic_training_set']:
        # Variable size items
        if total_pixels < 80000:
            return 'small'
        elif total_pixels < 300000:
            return 'medium'
        else:
            return 'large'
    else:
        # Default for other materials
        if total_pixels < 100000:
            return 'small'
        elif total_pixels < 250000:
            return 'medium'
        else:
            return 'large'

def calculate_carbon_impact_with_size(material_info, estimated_size='medium'):
    """Calculate realistic carbon footprint impact for a material with size consideration"""
    carbon_per_kg = material_info.get('carbon_per_kg', {'saved': 0, 'landfill': 0})
    typical_weight = material_info.get('typical_weight_kg', 0.1)
    size_multipliers = material_info.get('size_multipliers', {'small': 0.5, 'medium': 1.0, 'large': 2.0})
    
    # Calculate actual estimated weight based on size
    size_multiplier = size_multipliers.get(estimated_size, 1.0)
    estimated_weight = typical_weight * size_multiplier
    
    # Calculate carbon impact based on estimated weight
    carbon_saved = carbon_per_kg['saved'] * estimated_weight
    carbon_avoided = carbon_per_kg['landfill'] * estimated_weight
    total_impact = carbon_saved + carbon_avoided
    
    # More realistic equivalent calculations based on research data
    # Source: EPA, Carbon Trust, various environmental studies
    
    # Car miles: Average car emits 0.404 kg CO2 per mile (EPA 2023)
    car_miles_equivalent = total_impact / 0.404
    
    # Trees: A mature tree absorbs about 22 kg CO2 per year
    # So we calculate what fraction of a tree's annual absorption this represents
    tree_days_equivalent = (total_impact / 22) * 365  # days worth of tree absorption
    
    # Material-specific environmental benefits
    material_type = material_info.get('display_name', '').lower()
    
    # Energy savings (kWh) - based on actual recycling energy savings
    energy_multipliers = {
        'aluminum': 95,   # Recycling aluminum saves 95% of energy vs new production
        'metal': 74,      # Steel recycling saves 74% energy
        'plastic': 88,    # Plastic recycling saves 88% energy  
        'glass': 30,      # Glass recycling saves 30% energy
        'paper': 60,      # Paper recycling saves 60% energy
        'cardboard': 24,  # Cardboard recycling saves 24% energy
    }
    
    # Water savings (liters) - based on production water usage
    water_multipliers = {
        'aluminum': 1800,  # L per kg - aluminum production is very water intensive
        'metal': 280,      # L per kg - steel production water usage
        'plastic': 185,    # L per kg - plastic production water usage
        'glass': 120,      # L per kg - glass production water usage
        'paper': 60,       # L per kg - paper production water usage
        'cardboard': 50,   # L per kg - cardboard production water usage
    }
    
    # Get material-specific multipliers
    energy_factor = energy_multipliers.get(material_type, 40)  # default 40% energy savings
    water_factor = water_multipliers.get(material_type, 100)   # default 100L per kg
    
    energy_saved_kwh = estimated_weight * (energy_factor / 100) * 8.5  # 8.5 kWh avg per kg material production
    water_saved_liters = estimated_weight * water_factor
    
    # Format the results more meaningfully
    def format_tree_impact(days):
        if days >= 365:
            years = days / 365
            return f"{years:.1f} tree-years"
        elif days >= 30:
            months = days / 30
            return f"{months:.1f} tree-months"
        elif days >= 7:
            weeks = days / 7
            return f"{weeks:.1f} tree-weeks"
        else:
            return f"{days:.0f} tree-days"
    
    # Create a more informative calculation method
    impact_level = "significant" if total_impact > 1.0 else "moderate" if total_impact > 0.1 else "small but meaningful"
    
    return {
        'co2_saved_kg': round(carbon_saved, 3),
        'co2_avoided_kg': round(carbon_avoided, 3),
        'total_impact_kg': round(total_impact, 3),
        'estimated_weight_kg': round(estimated_weight, 3),
        'estimated_size': estimated_size,
        'car_miles_equivalent': round(car_miles_equivalent, 2),
        'trees_planted_equivalent': format_tree_impact(tree_days_equivalent),
        'energy_saved_kwh': round(energy_saved_kwh, 2),
        'water_saved_liters': round(water_saved_liters, 0),
        'calculation_method': f"Realistic impact assessment: {impact_level} environmental benefit from {estimated_size} {material_type} ({int(estimated_weight*1000)}g)"
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
        original_size = img.size  # Store original dimensions
        img = img.resize((224, 224))
        img_array = np.array(img) / 255.0
        img_array = np.expand_dims(img_array, axis=0)
        preds = model.predict(img_array)
        
        # Get prediction results
        pred_class = CLASS_NAMES[np.argmax(preds[0])]
        confidence = float(np.max(preds[0]))
        
        # Estimate item size from original image dimensions
        estimated_size = estimate_item_size_from_image(np.array(Image.open(file.stream).convert('RGB')), pred_class)
        
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
        
        # Update analytics with this action (including size estimation)
        update_analytics(pred_class, f"{location_info['city']}, {location_info['region']}", estimated_size)
        
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
            
            # 4. Carbon Footprint Impact (with size consideration)
            'carbon_impact': calculate_carbon_impact_with_size(material_info, estimated_size),
            
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

@app.route('/analytics', methods=['GET'])
def get_analytics():
    """Get recycling analytics and carbon footprint data"""
    try:
        analytics = load_analytics()
        
        # Calculate additional metrics
        total_impact = analytics['total_carbon_saved'] + analytics['total_carbon_avoided']
        car_miles = total_impact * 2.31
        trees_planted = total_impact / 21.77
        
        # Get recent activity (last 30 days)
        recent_sessions = []
        thirty_days_ago = (datetime.now() - timedelta(days=30)).isoformat()
        
        for session in analytics['sessions']:
            if session['timestamp'] > thirty_days_ago:
                recent_sessions.append(session)
        
        # Top categories
        top_categories = sorted(
            analytics['items_by_category'].items(), 
            key=lambda x: x[1], 
            reverse=True
        )[:5]
        
        response = {
            'summary': {
                'total_items_recycled': analytics['total_items'],
                'total_co2_saved_kg': round(analytics['total_carbon_saved'], 2),
                'total_co2_avoided_kg': round(analytics['total_carbon_avoided'], 2),
                'total_environmental_impact_kg': round(total_impact, 2),
                'equivalent_car_miles': round(car_miles, 1),
                'equivalent_trees_planted': round(trees_planted, 2)
            },
            'recent_activity': {
                'last_30_days': len(recent_sessions),
                'sessions': recent_sessions[-10:]  # Last 10 sessions
            },
            'breakdowns': {
                'by_category': dict(top_categories),
                'by_month': analytics['items_by_month']
            },
            'achievements': generate_achievements(analytics),
            'projections': calculate_projections(analytics)
        }
        
        return jsonify(response)
        
    except Exception as e:
        return jsonify({'error': str(e)}), 500

def generate_achievements(analytics):
    """Generate achievement badges based on recycling activity"""
    achievements = []
    
    total_items = analytics['total_items']
    total_impact = analytics['total_carbon_saved'] + analytics['total_carbon_avoided']
    
    # Item-based achievements
    if total_items >= 1:
        achievements.append({'name': 'First Step', 'icon': '🌱', 'description': 'Recycled your first item!'})
    if total_items >= 10:
        achievements.append({'name': 'Eco Warrior', 'icon': '♻️', 'description': 'Recycled 10+ items'})
    if total_items >= 50:
        achievements.append({'name': 'Green Champion', 'icon': '🏆', 'description': 'Recycled 50+ items'})
    if total_items >= 100:
        achievements.append({'name': 'Sustainability Master', 'icon': '🌍', 'description': 'Recycled 100+ items'})
    
    # Carbon impact achievements
    if total_impact >= 10:
        achievements.append({'name': 'Carbon Saver', 'icon': '🌬️', 'description': 'Saved 10+ kg CO2'})
    if total_impact >= 50:
        achievements.append({'name': 'Climate Hero', 'icon': '🦸', 'description': 'Saved 50+ kg CO2'})
    
    # Category diversity
    if len(analytics['items_by_category']) >= 5:
        achievements.append({'name': 'Recycling Expert', 'icon': '🎯', 'description': 'Recycled 5+ different material types'})
    
    return achievements

def calculate_projections(analytics):
    """Calculate yearly projections based on current activity"""
    if not analytics['sessions']:
        return {'yearly_items': 0, 'yearly_co2_impact': 0}
    
    # Calculate average per month from recent activity
    recent_months = list(analytics['items_by_month'].keys())[-3:]  # Last 3 months
    if not recent_months:
        return {'yearly_items': 0, 'yearly_co2_impact': 0}
    
    avg_monthly_items = sum(analytics['items_by_month'].get(month, 0) for month in recent_months) / len(recent_months)
    avg_monthly_co2 = (analytics['total_carbon_saved'] + analytics['total_carbon_avoided']) / len(recent_months)
    
    return {
        'yearly_items': round(avg_monthly_items * 12),
        'yearly_co2_impact': round(avg_monthly_co2 * 12, 2),
        'message': f"At your current pace, you'll recycle {round(avg_monthly_items * 12)} items and save {round(avg_monthly_co2 * 12, 2)} kg CO2 this year!"
    }

if __name__ == '__main__':
    app.run(debug=True, port=9000)
