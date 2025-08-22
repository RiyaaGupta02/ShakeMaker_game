from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from scipy.spatial.distance import cosine
import tensorflow as tf
import numpy as np
import pandas as pd
import json
import random
import os
from typing import List, Dict, Optional
import logging
import traceback


# Configure logging
# This sets up logging to print informational messages. The logger will be used to print logs for debugging or status updates.
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Initialize FastAPI app title shake maker ml api ---> fastapi makes api dev using python
app = FastAPI(title="Shake Maker ML API", version="1.0.0")

# Enable CORS for frontend connection
# This adds CORS (Cross-Origin Resource Sharing) middleware to allow your frontend (here Javascript one running on localhost:3000 or similar) to make API requests to the FastAPI backend.
# alllow_methods allows GET and POST requests no other methods
app.add_middleware(
    CORSMiddleware,
    allow_origins=["http://localhost:3000", "http://127.0.0.1:3000", "http://localhost:8080", "file://", "*"],  # Add your frontend URL
    allow_credentials=True,
    allow_methods=["GET", "POST"],
    allow_headers=["*"],
)

# Pydantic models for API
class ShakeRequest(BaseModel):
    ingredients: List[str]

class ShakeResponse(BaseModel):
    quality_score: float
    feedback: str
    emoji: str
    detailed_analysis: Dict

class IngredientInfo(BaseModel):
    name: str
    features: List[float]
    category: str

# ==== 2. ADD NEW PYDANTIC MODELS ====
class GenerateRequest(BaseModel):
    target_score: Optional[float] = None  # Target quality score (0.0-1.0)
    preferences: Optional[Dict] = None    # {"sweetness": 0.8, "creaminess": 0.6}
    categories: Optional[List[str]] = None # ["fruits", "syrups"]
    num_ingredients: Optional[int] = 3     # Number of ingredients to suggest
    avoid_ingredients: Optional[List[str]] = None  # Ingredients to avoid

class GenerateResponse(BaseModel):
    generated_shake: List[str]
    predicted_score: float
    confidence: float
    reasoning: str
    alternatives: List[Dict]  # Alternative suggestions

# ML Model Class
class ShakeMakerML:
    def __init__(self):
        self.model = None
        self.is_trained = False
        self.ingredient_features = self._define_ingredient_features()
        self.feature_names = ['sweetness', 'acidity', 'bitterness', 'creaminess', 'nutrition']
        self.model_path = "models/shake_predictor.keras"  # Change from .h5  --> as was causing issues with keras version compatibility & Fast api loading
        self.generator_model = None # new: generator neural network model
        self.is_generator_trained = False
        self.generator_path = "models/shake_generator.keras"  # Change from .h5

        # New line to check if set fix_vector is set to True
        self.fixed_ingredient_order = None  # This will be set to a fixed order of ingredients if needed


    # 3. ADD NEW DEBUG METHODS (place after _define_ingredient_features method)
    def debug_model_issues(self):
        """Comprehensive debugging method to identify predictor model issues"""
        print("🔍 DEBUGGING PREDICTOR MODEL ISSUES")
        print("=" * 50)
        
        # 1. Check ingredient consistency
        print(f"📊 Current ingredient count: {len(self.ingredient_features)}")
        print(f"📋 Ingredients: {list(self.ingredient_features.keys())}")
        
        # 2. Check vector size
        sample_vector = self._create_input_vector(['apple'])
        print(f"🔢 Current input vector size: {sample_vector.shape}")
        print(f"📏 Vector length: {sample_vector.shape[1]}")
        
        # 3. Check model file existence
        print(f"📁 Model file exists: {os.path.exists(self.model_path)}")
        print(f"📂 Model path: {self.model_path}")
        
        # 4. Try loading model and check compatibility
        if os.path.exists(self.model_path):
            try:
                temp_model = tf.keras.models.load_model(self.model_path)
                print(f"✅ Model loads successfully")
                print(f"🏗️  Model input shape: {temp_model.input_shape}")
                print(f"🧮 Expected input size: {temp_model.input_shape[1]}")
                print(f"⚖️  Size match: {temp_model.input_shape[1] == sample_vector.shape[1]}")
                
                # Try a prediction
                try:
                    test_pred = temp_model.predict(sample_vector, verbose=0)
                    print(f"✅ Test prediction successful: {test_pred[0][0]}")
                except Exception as e:
                    print(f"❌ Test prediction failed: {e}")
                    
            except Exception as e:
                print(f"❌ Model loading failed: {e}")
        else:
            print("📭 No existing model found - will need training")
        
        # 5. Check TensorFlow version
        print(f"🔧 TensorFlow version: {tf.__version__}")
        
        return {
            'ingredient_count': len(self.ingredient_features),
            'vector_size': sample_vector.shape[1],
            'model_exists': os.path.exists(self.model_path),
            'tf_version': tf.__version__
        }


    def fix_vector_size_consistency(self):
        """Fix input vector size consistency issues"""
        print("🔧 FIXING VECTOR SIZE CONSISTENCY")
        
        # Define fixed ingredient order (alphabetical for consistency)
        self.fixed_ingredient_order = sorted(list(self.ingredient_features.keys()))
        print(f"📝 Fixed ingredient order (first 10): {self.fixed_ingredient_order[:10]}")
        
        # Create consistent vector
        sample = self._create_input_vector_fixed(['apple', 'banana'])
        print(f"✅ Fixed vector size: {sample.shape[1]}")
        
        return sample.shape[1]
    
    

    def _define_ingredient_features(self) -> Dict:
        """Define nutritional and taste features for each ingredient"""
        ingredients = {
            # Fruits - [sweetness, acidity, bitterness, creaminess, nutrition]
            'apple': [0.8, 0.3, 0.1, 0.2, 0.8],        # More sweet than before
            'banana': [0.9, 0.1, 0.0, 0.8, 0.9],       # Very creamy and sweet
            'strawberry': [0.8, 0.5, 0.0, 0.2, 0.8],   # Much sweeter, more acidic
            'mango': [0.95, 0.2, 0.0, 0.6, 0.9],       # Very sweet and creamy
            'blueberry': [0.7, 0.4, 0.1, 0.2, 0.95],   # Sweeter, super nutritious
            'pineapple': [0.8, 0.7, 0.1, 0.3, 0.8],    # Sweet but very acidic
            'watermelon': [0.9, 0.1, 0.0, 0.4, 0.7],   # Very sweet, watery
            'peach': [0.9, 0.3, 0.0, 0.5, 0.8],        # Very sweet and soft
            'cherry': [0.8, 0.4, 0.1, 0.2, 0.7],       # Sweet with slight tartness
            'coconut': [0.6, 0.1, 0.1, 0.95, 0.8],     # Very creamy, moderately sweet
            'avocado': [0.2, 0.1, 0.1, 0.95, 0.95],    # Very creamy, super nutritious

            # Vegetables - [sweetness, acidity, bitterness, creaminess, nutrition]
            'spinach': [0.1, 0.2, 0.2, 0.2, 0.95],     # Mild, very nutritious
            'carrot': [0.6, 0.1, 0.1, 0.3, 0.85],      # Naturally sweet
            'beetroot': [0.7, 0.1, 0.2, 0.3, 0.9],     # Sweet earthy vegetable
            'kale': [0.1, 0.2, 0.6, 0.2, 0.95],        # Quite bitter but nutritious
            'celery': [0.1, 0.2, 0.4, 0.3, 0.7],       # Mild bitter, watery
            'cauliflower': [0.2, 0.1, 0.2, 0.4, 0.8],  # Mild, creamy when blended
            'sweet_potato': [0.8, 0.1, 0.1, 0.6, 0.9], # Very sweet and creamy
            'pumpkin': [0.7, 0.1, 0.1, 0.7, 0.85],     # Sweet and creamy
            'parsley': [0.1, 0.3, 0.5, 0.2, 0.8],      # Fresh but bitter
            'mint': [0.2, 0.3, 0.3, 0.2, 0.7],         # Fresh, slightly bitter

            # Cakes/Desserts - [sweetness, acidity, bitterness, creaminess, nutrition]
            'chocolate': [0.8, 0.0, 0.3, 0.9, 0.4],    # Sweet but bitter, very creamy
            'vanilla': [0.9, 0.0, 0.0, 0.85, 0.3],     # Very sweet and creamy
            'red_velvet': [0.95, 0.1, 0.1, 0.95, 0.2], # Very sweet and rich
            'cheesecake': [0.7, 0.2, 0.0, 0.95, 0.3],  # Rich and creamy
            'lemon': [0.3, 0.95, 0.1, 0.2, 0.6],       # Very acidic, not sweet
            'strawberry_cake': [0.95, 0.2, 0.0, 0.8, 0.3], # Very sweet
            'tiramisu': [0.7, 0.1, 0.4, 0.9, 0.3],     # Sweet but coffee bitter
            'black_forest': [0.8, 0.2, 0.3, 0.85, 0.3], # Sweet with chocolate bitter
            'coconut_cake': [0.9, 0.0, 0.0, 0.9, 0.4],  # Very sweet and creamy

            # Syrups - [sweetness, acidity, bitterness, creaminess, nutrition]
            'honey': [0.95, 0.0, 0.0, 0.3, 0.7],       # Very sweet, some nutrition
            'maple': [0.9, 0.0, 0.0, 0.4, 0.6],        # Sweet with slight complexity
            'chocolate_syrup': [0.9, 0.0, 0.2, 0.5, 0.2], # Very sweet, slight bitter
            'caramel': [0.95, 0.0, 0.0, 0.6, 0.3],     # Very sweet and creamy
            'vanilla_syrup': [0.95, 0.0, 0.0, 0.4, 0.2], # Very sweet
            'strawberry_syrup': [0.95, 0.1, 0.0, 0.3, 0.3] # Very sweet with fruit notes
        }
        
        print(f"✅ Total ingredients defined: {len(ingredients)}")
        return ingredients
    


    def _categorize_ingredients(self, ingredients: List[str]) -> Dict:
        """Categorize ingredients by type"""
        categories = {'fruits': [], 'vegetables': [], 'cakes': [], 'syrups': []}
        
        category_map = {
            # Fruits
            'apple': 'fruits', 'banana': 'fruits', 'strawberry': 'fruits',
            'mango': 'fruits', 'blueberry': 'fruits', 'pineapple': 'fruits',
            'watermelon': 'fruits', 'peach': 'fruits', 'cherry': 'fruits',
            'coconut': 'fruits', 'avocado': 'fruits',

            # Vegetables
            'spinach': 'vegetables', 'carrot': 'vegetables', 'beetroot': 'vegetables',
            'kale': 'vegetables', 'celery': 'vegetables', 'cauliflower': 'vegetables',
            'sweet_potato': 'vegetables', 'pumpkin': 'vegetables', 'parsley': 'vegetables',
            'mint': 'vegetables',

            # Cakes
            'chocolate': 'cakes', 'vanilla': 'cakes', 'red_velvet': 'cakes',
            'cheesecake': 'cakes', 'lemon': 'cakes', 'strawberry_cake': 'cakes',
            'tiramisu': 'cakes', 'black_forest': 'cakes', 'coconut_cake': 'cakes',
    
            # Syrups
            'honey': 'syrups', 'maple': 'syrups', 'chocolate_syrup': 'syrups',
            'caramel': 'syrups', 'vanilla_syrup': 'syrups', 'strawberry_syrup': 'syrups'
        }
        
        for ingredient in ingredients:
            category = category_map.get(ingredient)
            if category and category in categories:
                categories[category].append(ingredient)
                
        return categories
    
    # changed method to _create_input_vector to include 36 ingredients and 6 features
    def _create_input_vector(self, ingredients: List[str]) -> np.ndarray:
        """Fixed version of input vector creation with consistent ordering"""
        if not hasattr(self, 'fixed_ingredient_order') or self.fixed_ingredient_order is None:
            self.fixed_ingredient_order = sorted(list(self.ingredient_features.keys()))
        
        vector = []

        # print(f"Total ingredients available: {len(all_ingredients)}")
        # print(f"Ingredients list: {all_ingredients}")

        # One-hot encoding for ingredients (49 ingredients)
        for ingredient in self.fixed_ingredient_order:
            vector.append(1 if ingredient in ingredients else 0)

        # print(f"After one-hot encoding: {len(vector)} features")
        # 2. Aggregate features (5 features)
        if ingredients:
            valid_ingredients = [ing for ing in ingredients if ing in self.ingredient_features]
            if valid_ingredients:
                features_matrix = np.array([
                    self.ingredient_features[ing] for ing in valid_ingredients
                ])
                avg_features = np.mean(features_matrix, axis=0)
                vector.extend(avg_features.tolist())
            else:
                vector.extend([0.0] * 5)
        else:
            vector.extend([0.0] * 5)
        
        # 3. Ingredient count feature (1 feature)
        vector.append(len(ingredients) / 6.0)
        return np.array(vector).reshape(1, -1)
    
    
    def _generate_training_data_fixed(self, num_samples: int = 2000) -> tuple:
        """Generate training data with fixed vector creation"""
        X_train = []
        y_train = []
        
        all_ingredients = self.fixed_ingredient_order
        
        # Generate random combinations
        for _ in range(num_samples):
            num_ingredients = np.random.randint(2, 6)
            ingredients = np.random.choice(all_ingredients, num_ingredients, replace=False).tolist()
            
            X_train.append(self._create_input_vector_fixed(ingredients).flatten())
            y_train.append(self._calculate_quality_score(ingredients))
        
        # Add expert combinations
        expert_combinations = [
            # Excellent combinations (0.85-0.95)
            (['banana', 'strawberry', 'honey'], 0.92),
            (['mango', 'pineapple', 'coconut'], 0.91),
            (['chocolate', 'vanilla', 'chocolate_syrup'], 0.93),
            (['strawberry', 'vanilla_syrup', 'strawberry_cake'], 0.90),
            (['coconut', 'coconut_cake', 'vanilla_syrup'], 0.91),
            (['cherry', 'black_forest', 'chocolate_syrup'], 0.90),
            (['peach', 'honey', 'vanilla'], 0.89),
            (['tiramisu', 'chocolate_syrup'], 0.90),
            (['banana', 'strawberry', 'vanilla_syrup'], 0.91),
            (['banana', 'strawberry', 'caramel_syrup'], 0.90),
            (['banana', 'honey', 'vanilla_syrup'], 0.90),
            (['strawberry', 'banana', 'yogurt'], 0.89),
            (['banana', 'caramel', 'vanilla'], 0.88),
            (['strawberry', 'caramel', 'banana'], 0.89),
            (['strawberry', 'vanilla', 'banana'], 0.90),
        
            # Very good combinations (0.75-0.84)
            (['apple', 'caramel'], 0.84),
            (['blueberry', 'banana', 'honey'], 0.85),
            (['watermelon', 'mint'], 0.82),
            (['sweet_potato', 'maple'], 0.83),
            (['pumpkin', 'maple', 'vanilla'], 0.84),
            (['coconut_cake', 'pineapple'], 0.85),
            (['avocado', 'banana', 'honey'], 0.82),
            (['strawberry_cake', 'strawberry_syrup'], 0.86),
            (['carrot', 'apple', 'honey'], 0.81),
            (['spinach', 'banana', 'mango'], 0.80),
            (['beetroot', 'apple', 'honey'], 0.81),
            (['pineapple', 'mint'], 0.80),
            (['red_velvet', 'vanilla_syrup'], 0.83),
            (['cheesecake', 'strawberry'], 0.82),
            (['lemon', 'honey'], 0.80),


            # Good combinations (0.70–0.79)
            (['apple', 'honey', 'yogurt'], 0.78),
            (['mango', 'banana', 'carrot'], 0.74),
            (['pumpkin', 'banana', 'maple'], 0.77),
            (['blueberry', 'coconut', 'vanilla'], 0.76),
            (['peach', 'banana', 'caramel'], 0.75),
            (['yogurt', 'banana', 'honey'], 0.79),

            # Poor combinations (0.15–0.25)
            (['spinach', 'chocolate', 'lemon'], 0.18),
            (['kale', 'red_velvet', 'cauliflower'], 0.12),
            (['celery', 'strawberry_syrup', 'beetroot'], 0.22),
            (['parsley', 'tiramisu', 'mint'], 0.15),
            (['cauliflower', 'chocolate_syrup', 'spinach'], 0.14),
            (['kale', 'caramel', 'celery'], 0.16),

            
            # Very poor combinations (0.08-0.14)
            (['spinach', 'black_forest', 'parsley'], 0.10),
            (['celery', 'coconut_cake', 'kale'], 0.11),
            (['beetroot', 'tiramisu', 'cauliflower'], 0.09),
            (['mint', 'red_velvet', 'spinach'], 0.13),
            (['parsley', 'chocolate_syrup', 'cauliflower'], 0.08),
        ]
        
        for ingredients, quality in expert_combinations:
            X_train.append(self._create_input_vector_fixed(ingredients).flatten())
            y_train.append(quality)
        
        return np.array(X_train), np.array(y_train)
    

    def rebuild_predictor_model_safe(self):
        """Rebuild predictor model with error handling and validation"""
        print("🏗️  REBUILDING PREDICTOR MODEL SAFELY")
        
        try:
            # 1. Ensure consistent ingredient ordering
            self.fix_vector_size_consistency()
            
            # 2. Generate training data with fixed vectors
            print("📊 Generating training data...")
            X_train, y_train = self._generate_training_data_fixed()
            print(f"✅ Training data shape: X={X_train.shape}, y={y_train.shape}")
            
            # 3. Build model with exact input size
            actual_size = X_train.shape[1]
            print(f"🎯 Building model for input size: {actual_size}")
            
            self.model = tf.keras.Sequential([
                tf.keras.layers.Dense(128, activation='relu', input_shape=(actual_size,),
                                    kernel_regularizer=tf.keras.regularizers.l2(0.001)),
                tf.keras.layers.BatchNormalization(),
                tf.keras.layers.Dropout(0.3),
                
                tf.keras.layers.Dense(64, activation='relu',
                                    kernel_regularizer=tf.keras.regularizers.l2(0.001)),
                tf.keras.layers.BatchNormalization(),
                tf.keras.layers.Dropout(0.2),
                
                tf.keras.layers.Dense(32, activation='relu'),
                tf.keras.layers.Dropout(0.1),
                
                tf.keras.layers.Dense(1, activation='sigmoid')
            ])
            
            # 4. Compile model
            self.model.compile(
                optimizer=tf.keras.optimizers.Adam(learning_rate=0.001),
                loss='mean_squared_error',
                metrics=['mean_absolute_error']
            )
            
            print("✅ Model built successfully")
            
            # 5. Train with validation
            print("🎓 Training model...")
            history = self.model.fit(
                X_train, y_train,
                epochs=100,
                batch_size=32,
                validation_split=0.2,
                verbose=1
            )
            
            # 6. Save model
            os.makedirs("models", exist_ok=True)
            self.model.save(self.model_path)
            print(f"💾 Model saved to: {self.model_path}")
            
            # 7. Test model
            test_ingredients = ['apple', 'banana', 'honey']
            test_vector = self._create_input_vector_fixed(test_ingredients)
            test_pred = self.model.predict(test_vector, verbose=0)
            print(f"🧪 Test prediction: {test_pred[0][0]:.3f}")
            
            self.is_trained = True
            return True
            
        except Exception as e:
            print(f"❌ Rebuild failed: {e}")
            traceback.print_exc()
            return False
        
    def emergency_fix_predictor(self):
        """Emergency fix for predictor model"""
        print("🚨 EMERGENCY PREDICTOR FIX")
        
        # 1. Remove incompatible model
        if os.path.exists(self.model_path):
            os.remove(self.model_path)
            print("🗑️  Removed incompatible model")
        
        # 2. Reset training state
        self.is_trained = False
        self.model = None
        # 3. Rebuild from scratch
        success = self.rebuild_predictor_model_safe()
        if success:
            print("✅ Emergency fix successful!")
        else:
            print("❌ Emergency fix failed - using rule-based fallback")
        return success
    

    def _calculate_quality_score(self, ingredients: List[str]) -> float:
        """Rule-based quality calculation for training data"""
        if not ingredients:
            return 0.0
            
        score = 0.5  # Base score
        categories = self._categorize_ingredients(ingredients)
        
        # Category compatibility rules
        if len(categories['fruits']) >= 2:
            score += 0.25
        
        # Penalty for mixing vegetables with desserts
        if len(categories['vegetables']) > 0 and (len(categories['cakes']) > 0 or len(categories['syrups']) > 0):
            score -= 0.35
        
        # Bonus for fruit + syrup combinations
        if len(categories['fruits']) > 0 and len(categories['syrups']) > 0:
            score += 0.2
        
        # Penalty for too many dessert items
        if len(categories['cakes']) > 1 and len(categories['syrups']) > 1:
            score -= 0.15
        
        # Bonus for healthy combinations
        if len(categories['fruits']) > 0 and len(categories['vegetables']) == 1:
            score += 0.15
        
        # Penalty for too many ingredients
        if len(ingredients) > 4:
            score -= 0.1
        
        # Feature-based adjustments
        valid_ingredients = [ing for ing in ingredients if ing in self.ingredient_features]
        if valid_ingredients:
            features = np.array([self.ingredient_features[ing] for ing in valid_ingredients])
            avg_features = np.mean(features, axis=0)
            
            # Penalty for high bitterness
            if avg_features[2] > 0.4:
                score -= 0.25
            
            # Bonus for balanced sweet-acid
            if 0.5 < avg_features[0] < 0.9 and 0.1 < avg_features[1] < 0.6:
                score += 0.15
        
        return np.clip(score, 0.0, 1.0)
    
    # new feature on model compatibility check --> added
    
    def _check_model_compatibility(self) -> bool:
        """Check if saved model matches current vector size"""
        if not os.path.exists(self.model_path):
            return False
            
        try:
            sample_vector = self._create_input_vector(['apple'])
            current_size = sample_vector.shape[1]
            
            temp_model = tf.keras.models.load_model(self.model_path)
            expected_size = temp_model.input_shape[1]
            
            if expected_size != current_size:
                logger.warning(f"Model incompatible: expects {expected_size}, got {current_size}")
                return False
                
            return True
        except Exception as e:
            logger.error(f"Model compatibility check failed: {e}")
            return False
        


    
    def build_model(self) -> tf.keras.Model:
        """Build neural network architecture"""

        # dynamically determine actual input vector size based on ingredient features
        sample_vector = self._create_input_vector(['apple'])  # Use any valid ingredient
        actual_size = sample_vector.shape[1]  # This will reflect 41 input features

        print(f"Model inpute size determined: {actual_size}")      # --> debug line adding


        model = tf.keras.Sequential([
            tf.keras.layers.Dense(128, activation='relu', input_shape=(actual_size,),    
                                kernel_regularizer=tf.keras.regularizers.l2(0.001)),
            tf.keras.layers.BatchNormalization(),
            tf.keras.layers.Dropout(0.3),
            
            tf.keras.layers.Dense(64, activation='relu',
                                kernel_regularizer=tf.keras.regularizers.l2(0.001)),
            tf.keras.layers.BatchNormalization(),
            tf.keras.layers.Dropout(0.2),
            
            tf.keras.layers.Dense(32, activation='relu'),
            tf.keras.layers.Dropout(0.1),
            
            tf.keras.layers.Dense(1, activation='sigmoid')
        ])
        
        model.compile(
            optimizer=tf.keras.optimizers.Adam(learning_rate=0.001),
            loss='mean_squared_error',  # Using MSE for regression-like output
            metrics=['mean_absolute_error'] # changed to MAE for better interpretability --> sometimes shortcut doesnt work so mse & mae write full 
        )
        return model
    
    def train_model(self) -> bool:
        """Train the neural network with fixed vector creation"""
        try:
            logger.info("Training model with fixed vector approach...")
            
            # Ensure fixed ingredient order
            if self.fixed_ingredient_order is None:
                self.fixed_ingredient_order = sorted(list(self.ingredient_features.keys()))
            
            # Use fixed training data generation
            logger.info("Generating training data...")
            X_train, y_train = self._generate_training_data_fixed()
            
            logger.info("Building model...")
            actual_size = X_train.shape[1]
            self.model = tf.keras.Sequential([
                tf.keras.layers.Dense(128, activation='relu', input_shape=(actual_size,),
                                    kernel_regularizer=tf.keras.regularizers.l2(0.001)),
                tf.keras.layers.BatchNormalization(),
                tf.keras.layers.Dropout(0.3),
                
                tf.keras.layers.Dense(64, activation='relu',
                                    kernel_regularizer=tf.keras.regularizers.l2(0.001)),
                tf.keras.layers.BatchNormalization(),
                tf.keras.layers.Dropout(0.2),
                
                tf.keras.layers.Dense(32, activation='relu'),
                tf.keras.layers.Dropout(0.1),
                
                tf.keras.layers.Dense(1, activation='sigmoid')
            ])
            
            self.model.compile(
                optimizer=tf.keras.optimizers.Adam(learning_rate=0.001),
                loss='mean_squared_error',
                metrics=['mean_absolute_error']
            )
            
            # Create callbacks
            early_stopping = tf.keras.callbacks.EarlyStopping(
                monitor='val_loss', patience=15, restore_best_weights=True
            )
            
            reduce_lr = tf.keras.callbacks.ReduceLROnPlateau(
                monitor='val_loss', factor=0.5, patience=10, min_lr=0.0001
            )
            
            logger.info("Training model...")
            history = self.model.fit(
                X_train, y_train,
                epochs=150,
                batch_size=32,
                validation_split=0.2,
                callbacks=[early_stopping, reduce_lr],
                verbose=1
            )
            
            # Save model
            os.makedirs("models", exist_ok=True)
            self.model.save(self.model_path)
            
            self.is_trained = True
            logger.info("Model training completed successfully!")

            # Log final metrics
            final_loss = history.history.get('val_loss', [None])[-1]
            if final_loss is not None:
                logger.info(f"Final validation loss: {final_loss:.4f}")
            
            final_mae = history.history.get('val_mean_absolute_error', [None])[-1]
            if final_mae is not None:
                logger.info(f"Final validation MAE: {final_mae:.4f}")
                    
            return True
                
        except Exception as e:
            logger.error(f"Error during training: {str(e)}")
            return False
        
    
    def load_model(self) -> bool:
        """Load pre-trained model with enhanced compatibility check"""
        try:
            if self._check_model_compatibility():
                self.model = tf.keras.models.load_model(self.model_path)
                # Ensure fixed ingredient order is set
                if self.fixed_ingredient_order is None:
                    self.fixed_ingredient_order = sorted(list(self.ingredient_features.keys()))
                self.is_trained = True
                logger.info("Model loaded successfully!")
                return True
            else:
                logger.info("Model incompatible or missing - will need retraining")
                return False
        except Exception as e:
            logger.error(f"Error loading model: {str(e)}")
            return False
    
    # 4. REPLACE EXISTING predict_quality METHOD (find this method and replace it completely)
    def predict_quality(self, ingredients: List[str]) -> Dict:
        """Predict shake quality and provide analysis with safe fallback"""
        if not self.is_trained or self.model is None:
            # Fallback to rule-based prediction
            score = self._calculate_quality_score(ingredients)
            logger.warning("Using rule-based prediction (model not ready)")
        else:
            try:
                # Use fixed vector creation if available
                if hasattr(self, '_create_input_vector_fixed') and self.fixed_ingredient_order is not None:
                    input_vector = self._create_input_vector_fixed(ingredients)
                else:
                    input_vector = self._create_input_vector(ingredients)
                    
                prediction = self.model.predict(input_vector, verbose=0)
                score = float(prediction[0][0])
                logger.info(f"Neural prediction successful: {score:.3f}")
                
            except Exception as e:
                logger.error(f"Prediction error: {str(e)}")
                logger.warning("Falling back to rule-based prediction")
                score = self._calculate_quality_score(ingredients)
        
        # Generate feedback and analysis
        feedback = self._generate_feedback(ingredients, score)
        emoji = self._get_emoji(score)
        analysis = self._detailed_analysis(ingredients, score)
        
        return {
            'quality_score': score,
            'feedback': feedback,
            'emoji': emoji,
            'detailed_analysis': analysis
        }
    
    
    def _generate_feedback(self, ingredients: List[str], score: float) -> str:
        """Generate textual feedback"""
        if score > 0.8:
            base_feedback = "Excellent combination! This shake tastes amazing! 🌟"
        elif score > 0.6:
            base_feedback = "Good shake! Nice flavor balance. 😊"
        elif score > 0.4:
            base_feedback = "Interesting combination, could use some tweaking. 🤔"
        elif score > 0.2:
            base_feedback = "Hmm, this combination might not work well together. 😐"
        else:
            base_feedback = "This combination doesn't taste good at all! 🤢"
        
        # Add specific tips
        categories = self._categorize_ingredients(ingredients)
        tips = []
        
        if len(categories['vegetables']) > 0 and len(categories['cakes']) > 0:
            tips.append("Vegetables and cakes don't usually mix well in shakes.")
        
        if len(categories['fruits']) >= 2:
            tips.append("Great choice mixing fruits together!")
        
        if len(ingredients) > 4:
            tips.append("Try using fewer ingredients for better balance.")
        
        if tips:
            base_feedback += " Tips: " + " ".join(tips)
        
        return base_feedback
    
    def _get_emoji(self, score: float) -> str:
        """Get emoji based on score"""
        if score > 0.8:
            return "😋"
        elif score > 0.6:
            return "🙂"
        elif score > 0.4:
            return "😐"
        elif score > 0.2:
            return "😕"
        else:
            return "🤢"
    
    def _detailed_analysis(self, ingredients: List[str], score: float) -> Dict:
        """Provide detailed nutritional and taste analysis"""
        categories = self._categorize_ingredients(ingredients)
        
        # Calculate average features
        valid_ingredients = [ing for ing in ingredients if ing in self.ingredient_features]
        if valid_ingredients:
            features = np.array([self.ingredient_features[ing] for ing in valid_ingredients])
            avg_features = np.mean(features, axis=0)
        else:
            avg_features = np.zeros(5)
        
        return {
            'score_breakdown': {
                'overall_score': round(score * 100, 1),
                'taste_balance': round(avg_features[0] * 0.4 + avg_features[1] * 0.3 + (1-avg_features[2]) * 0.3, 2),
                'nutritional_value': round(avg_features[4], 2),
                'texture_score': round(avg_features[3], 2)
            },
            'ingredient_categories': categories,
            'feature_analysis': {
                'sweetness': round(avg_features[0], 2),
                'acidity': round(avg_features[1], 2),
                'bitterness': round(avg_features[2], 2),
                'creaminess': round(avg_features[3], 2),
                'nutrition': round(avg_features[4], 2)
            },
            'recommendations': self._get_recommendations(ingredients, score, categories)
        }
    
    # Made changes & updates in recommendations method to include more --> better fun game kind of way recommendations --> to make it more fun and engaging
    def _get_recommendations(self, ingredients: List[str], score: float, categories: Dict) -> List[str]:
        """Generate fun, cafe-style recommendations without being judgmental"""
        recommendations = []
        
        # Cafe-style encouraging phrases
        cafe_phrases = [
            "What a creative mix! 🎨",
            "Interesting choice! ☕",
            "That's adventurous! 🌟",
            "Nice experiment! 🧪",
            "Ooh, creative! ✨"
        ]
        
        # Always start positive
        recommendations.append(random.choice(cafe_phrases))
        
        # Simple, encouraging suggestions based on what they have
        if len(categories.get('fruits', [])) >= 2:
            recommendations.append("🍓 Love the fruit combo - very refreshing!")
        
        if len(categories.get('vegetables', [])) > 0:
            recommendations.append("🥬 Adding veggies = extra nutrition points!")
        
        if len(categories.get('syrups', [])) > 0:
            recommendations.append("🍯 Sweet touch - that'll taste yummy!")
        
        # Fun cafe-style suggestions (not corrections)
        fun_suggestions = [
            "☕ Café tip: Try adding a banana next time for creaminess!",
            "🍓 Berry idea: Strawberries go with almost everything!",
            "🥭 Tropical twist: Mango makes everything taste like vacation!",
            "🍯 Sweet secret: A little honey never hurt anyone!",
            "🥬 Green goddess: Spinach is sneaky - you won't even taste it!",
            "🍫 Chocolate fix: Everything's better with a little chocolate!"
        ]
        
        # Add one random fun suggestion
        recommendations.append(random.choice(fun_suggestions))
        return recommendations[:3]  # Keep it short - max 3 recommendations
    
    def _analyze_combination_type(self, categories: Dict) -> str:
        """Simplified analysis - no judgment, just categorization"""
        if len(categories.get('fruits', [])) >= 2:
            return "fruity"
        elif len(categories.get('vegetables', [])) > 0:
            return "healthy"
        elif len(categories.get('cakes', [])) > 0:
            return "treat"
        else:
            return "classic"


    # adding functions now for GEN shake recommendations 
    # that is generative model to suggest new shake combinations
    def build_generator_model(self) -> tf.keras.Model:
        """Build generator neural for creating shake suggestions"""
        model = tf.keras.Sequential([
            # input: prefernces + target score(6 features )
            tf.keras.layers.Dense(128, activation='relu', input_shape=(6,)),
            tf.keras.layers.Dropout(0.4),

            tf.keras.layers.Dense(256, activation='relu'),
            tf.keras.layers.BatchNormalization(),
            tf.keras.layers.Dropout(0.4),

            tf.keras.layers.Dense(128, activation='relu'),
            tf.keras.layers.Dropout(0.4),

            # Output: probablity for each ingredient (24 ingredients)
            tf.keras.layers.Dense(len(self.ingredient_features), activation='sigmoid')
        ])
        
        model.compile(
            optimizer=tf.keras.optimizers.Adam(learning_rate=0.001),
            loss='binary_crossentropy',
            metrics=['accuracy']
        )
        
        return model
    
    def _generate_generator_training_data(self, num_samples: int = 1500) -> tuple:
        """Generate training data for generator model using 1500 samples not 3000 halved down for small scale project not an issue """
        X_gen = []       # [target_score, sweetness, creaminess, acidity, bitterness, nutrition]
        y_gen = []       # one-hot encoded ingredient combinations
        
        all_ingredients = list(self.ingredient_features.keys())
        
        for _ in range(num_samples):
            # Randomly select target score and preferences
            # Random target preferences
            target_score = np.random.uniform(0.2, 0.95)
            target_sweetness = np.random.uniform(0.0, 1.0)
            target_acidity = np.random.uniform(0.0, 0.8)
            target_bitterness = np.random.uniform(0.0, 0.5)
            target_creaminess = np.random.uniform(0.0, 1.0)
            target_nutrition = np.random.uniform(0.3, 1.0)
            
            # Generate ingredients that would match these preferences
            num_ingredients = np.random.randint(2, 5)
            selected_ingredients = self._select_ingredients_for_preferences({
                'target_score': target_score,
                'sweetness': target_sweetness,
                'acidity': target_acidity,
                'bitterness': target_bitterness,
                'creaminess': target_creaminess,
                'nutrition': target_nutrition
            }, num_ingredients)
            
            # Create input vector
            input_vector = [target_score, target_sweetness, target_acidity, 
                          target_bitterness, target_creaminess, target_nutrition]
            X_gen.append(input_vector)
            
            # Create output vector (one-hot encoded ingredients)
            output_vector = [1 if ing in selected_ingredients else 0 
                           for ing in all_ingredients]
            y_gen.append(output_vector)
        
        return np.array(X_gen), np.array(y_gen)
    

    def _select_ingredients_for_preferences(self, prefs: Dict, num_ingredients: int) -> List[str]:
        """Helper method to select ingredients that match preferences"""
        all_ingredients = list(self.ingredient_features.keys())
        scored_ingredients = []
        
        for ingredient in all_ingredients:
            features = self.ingredient_features[ingredient]
            # Calculate how well this ingredient matches preferences
            match_score = (
                1.0 - abs(features[0] - prefs['sweetness']) * 0.3 +
                1.0 - abs(features[1] - prefs['acidity']) * 0.2 +
                1.0 - abs(features[2] - prefs['bitterness']) * 0.2 +
                1.0 - abs(features[3] - prefs['creaminess']) * 0.2 +
                1.0 - abs(features[4] - prefs['nutrition']) * 0.1
            )
            scored_ingredients.append((ingredient, match_score))
        
        # Sort by match score and add some randomness
        scored_ingredients.sort(key=lambda x: x[1] + np.random.normal(0, 0.1), reverse=True)
        
        return [ing for ing, _ in scored_ingredients[:num_ingredients]]

    def train_generator_model(self) -> bool:
        """Train the generator neural network"""
        try:
            logger.info("Generating training data for generator...")
            X_gen, y_gen = self._generate_generator_training_data()
            
            logger.info("Building generator model...")
            self.generator_model = self.build_generator_model()
            
            # Create callbacks
            early_stopping = tf.keras.callbacks.EarlyStopping(
                monitor='val_loss', patience=10, restore_best_weights=True
            )
            
            logger.info("Training generator model...")
            history = self.generator_model.fit(
                X_gen, y_gen,
                epochs=120,   # increased from 100 to 120 (more epochs for better training as less data )
                batch_size=16,   # reduced batch size for better convergence
                validation_split=0.15,     # reduced validation split to 0.15 from 0.2 for more training data
                callbacks=[early_stopping],
                verbose=1
            )
            
            # Save generator model
            os.makedirs("models", exist_ok=True)
            self.generator_model.save(self.generator_path)
            
            self.is_generator_trained = True
            logger.info("Generator model training completed successfully!")
            
            return True
            
        except Exception as e:
            logger.error(f"Error during generator training: {str(e)}")
            return False

    def load_generator_model(self) -> bool:
        """Load pre-trained generator model"""
        try:
            if os.path.exists(self.generator_path):
                self.generator_model = tf.keras.models.load_model(self.generator_path)
                self.is_generator_trained = True
                logger.info("Generator model loaded successfully!")
                return True
            return False
        except Exception as e:
            logger.error(f"Error loading generator model: {str(e)}")
            return False

    def generate_shake_suggestion(self, request: GenerateRequest) -> Dict:
        """Generate shake suggestions based on user preferences"""
        try:
            # Set default values
            target_score = request.target_score or 0.8
            num_ingredients = request.num_ingredients or 3
            avoid_ingredients = request.avoid_ingredients or []
            
            # If we have a trained generator model, use it
            if self.is_generator_trained and self.generator_model:
                return self._neural_generate(request, target_score, num_ingredients, avoid_ingredients)
            else:
                # Fallback to rule-based generation
                return self._rule_based_generate(request, target_score, num_ingredients, avoid_ingredients)
                
        except Exception as e:
            logger.error(f"Error generating shake: {str(e)}")
            return self._fallback_generation()

    def _neural_generate(self, request: GenerateRequest, target_score: float, 
                        num_ingredients: int, avoid_ingredients: List[str]) -> Dict:
        """Generate using neural network"""
        # Prepare input for generator
        prefs = request.preferences or {}
        input_vector = np.array([[
            target_score,
            prefs.get('sweetness', 0.7),
            prefs.get('acidity', 0.3),
            prefs.get('bitterness', 0.1),
            prefs.get('creaminess', 0.5),
            prefs.get('nutrition', 0.8)
        ]])
        
        # Get predictions from generator
        predictions = self.generator_model.predict(input_vector, verbose=0)[0]
        
        # Convert predictions to ingredient selection
        all_ingredients = list(self.ingredient_features.keys())
        ingredient_scores = list(zip(all_ingredients, predictions))
        
        # Filter out avoided ingredients
        ingredient_scores = [(ing, score) for ing, score in ingredient_scores 
                           if ing not in avoid_ingredients]
        
        # Filter by categories if specified
        if request.categories:
            category_map = self._get_category_map()
            ingredient_scores = [(ing, score) for ing, score in ingredient_scores 
                               if category_map.get(ing) in request.categories]
        
        # Sort by prediction score and add some randomness for variety
        ingredient_scores.sort(key=lambda x: x[1] + np.random.normal(0, 0.05), reverse=True)
        
        # Select top ingredients
        selected_ingredients = [ing for ing, _ in ingredient_scores[:num_ingredients]]
        
        # Predict quality of generated shake
        predicted_quality = self.predict_quality(selected_ingredients)
        
        # Generate alternatives
        alternatives = self._generate_alternatives(ingredient_scores, num_ingredients, avoid_ingredients)
        
        return {
            'generated_shake': selected_ingredients,
            'predicted_score': predicted_quality['quality_score'],
            'confidence': min(0.95, sum([score for _, score in ingredient_scores[:num_ingredients]]) / num_ingredients),
            'reasoning': self._generate_reasoning(selected_ingredients, request),
            'alternatives': alternatives
        }

    def _rule_based_generate(self, request: GenerateRequest, target_score: float, 
                            num_ingredients: int, avoid_ingredients: List[str]) -> Dict:
        """Fallback rule-based generation when neural model isn't available"""
        all_ingredients = list(self.ingredient_features.keys())
        available_ingredients = [ing for ing in all_ingredients if ing not in avoid_ingredients]
        
        # Filter by categories if specified
        if request.categories:
            category_map = self._get_category_map()
            available_ingredients = [ing for ing in available_ingredients 
                                   if category_map.get(ing) in request.categories]
        
        # Score ingredients based on how well they match preferences
        prefs = request.preferences or {}
        scored_ingredients = []
        
        for ingredient in available_ingredients:
            features = self.ingredient_features[ingredient]
            score = self._calculate_preference_match(features, prefs, target_score)
            scored_ingredients.append((ingredient, score))
        
        # Sort and select best ingredients
        scored_ingredients.sort(key=lambda x: x[1] + np.random.normal(0, 0.1), reverse=True)
        selected_ingredients = [ing for ing, _ in scored_ingredients[:num_ingredients]]
        
        # Predict quality
        predicted_quality = self.predict_quality(selected_ingredients)
        
        # Generate alternatives
        alternatives = self._generate_alternatives(scored_ingredients, num_ingredients, avoid_ingredients)
        
        return {
            'generated_shake': selected_ingredients,
            'predicted_score': predicted_quality['quality_score'],
            'confidence': 0.75,  # Lower confidence for rule-based
            'reasoning': self._generate_reasoning(selected_ingredients, request),
            'alternatives': alternatives
        }

    def _calculate_preference_match(self, features: List[float], prefs: Dict, target_score: float) -> float:
        """Calculate how well ingredient matches user preferences"""
        score = 0.5  # Base score
        
        # Match against specific preferences
        if 'sweetness' in prefs:
            score += (1.0 - abs(features[0] - prefs['sweetness'])) * 0.3
        if 'acidity' in prefs:
            score += (1.0 - abs(features[1] - prefs['acidity'])) * 0.2
        if 'creaminess' in prefs:
            score += (1.0 - abs(features[3] - prefs['creaminess'])) * 0.3
        if 'nutrition' in prefs:
            score += (1.0 - abs(features[4] - prefs['nutrition'])) * 0.2
        
        # Boost score based on target quality
        if target_score > 0.8:
            # For high-quality shakes, prefer fruits and balanced ingredients
            if features[0] > 0.6 and features[2] < 0.3:  # Sweet and not bitter
                score += 0.2
        
        return np.clip(score, 0.0, 1.0)

    def _generate_alternatives(self, scored_ingredients: List, num_ingredients: int, 
                              avoid_ingredients: List[str]) -> List[Dict]:
        """Generate alternative shake suggestions"""
        alternatives = []
        
        for i in range(3):  # Generate 3 alternatives
            start_idx = (i + 1) * 2  # Start from different positions
            alt_ingredients = [ing for ing, _ in scored_ingredients[start_idx:start_idx + num_ingredients]]
            
            if len(alt_ingredients) >= 2:  # Ensure minimum ingredients
                alt_quality = self.predict_quality(alt_ingredients)
                alternatives.append({
                    'ingredients': alt_ingredients,
                    'predicted_score': alt_quality['quality_score'],
                    'description': f"Alternative {i+1}: {', '.join(alt_ingredients)}"
                })
        
        return alternatives

    def _generate_reasoning(self, ingredients: List[str], request: GenerateRequest) -> str:
        """Generate explanation for why these ingredients were chosen"""
        categories = self._categorize_ingredients(ingredients)
        reasoning_parts = []
        
        # Explain ingredient selection
        if len(categories['fruits']) >= 2:
            reasoning_parts.append("Selected multiple fruits for natural sweetness and flavor balance")
        
        if len(categories['syrups']) > 0:
            reasoning_parts.append("Added syrup for enhanced sweetness")
        
        if request.preferences:
            if request.preferences.get('creaminess', 0) > 0.6:
                creamy_ingredients = [ing for ing in ingredients 
                                    if self.ingredient_features[ing][3] > 0.5]
                if creamy_ingredients:
                    reasoning_parts.append(f"Included {', '.join(creamy_ingredients)} for creaminess")
        
        if request.target_score and request.target_score > 0.8:
            reasoning_parts.append("Chose high-quality ingredients to meet your target score")
        
        return ". ".join(reasoning_parts) if reasoning_parts else "Selected based on flavor compatibility"

    def _get_category_map(self) -> Dict:
        """Get ingredient to category mapping"""
        return {
            # Fruits
            'apple': 'fruits', 'banana': 'fruits', 'strawberry': 'fruits',
            'mango': 'fruits', 'blueberry': 'fruits', 'pineapple': 'fruits',
            'watermelon': 'fruits', 'peach': 'fruits', 'cherry': 'fruits',
            'coconut': 'fruits', 'avocado': 'fruits',
            # Vegetables
            'spinach': 'vegetables', 'carrot': 'vegetables', 'beetroot': 'vegetables',
            'kale': 'vegetables', 'celery': 'vegetables', 'cauliflower': 'vegetables',
            'sweet_potato': 'vegetables', 'pumpkin': 'vegetables', 'parsley': 'vegetables',
            'mint': 'vegetables',
            # Cakes
            'chocolate': 'cakes', 'vanilla': 'cakes', 'red_velvet': 'cakes',
            'cheesecake': 'cakes', 'lemon': 'cakes', 'strawberry_cake': 'cakes',
            'tiramisu': 'cakes', 'black_forest': 'cakes', 'coconut_cake': 'cakes',
            # Syrups
            'honey': 'syrups', 'maple': 'syrups', 'chocolate_syrup': 'syrups',
            'caramel': 'syrups', 'vanilla_syrup': 'syrups', 'strawberry_syrup': 'syrups'
        }

    def _fallback_generation(self) -> Dict:
        """Emergency fallback generation"""
        safe_combinations = [
            ['banana', 'strawberry', 'honey'],
            ['mango', 'pineapple', 'coconut'],
            ['apple', 'caramel'],
            ['chocolate', 'vanilla', 'chocolate_syrup']
        ]
        
        selected = random.choice(safe_combinations)
        quality = self.predict_quality(selected)
        
        return {
            'generated_shake': selected,
            'predicted_score': quality['quality_score'],
            'confidence': 0.6,
            'reasoning': "Generated using safe ingredient combinations",
            'alternatives': []
        }

# Initialize ML model
ml_model = ShakeMakerML()

@app.on_event("startup")
async def startup_event():
    """Initialize model on startup with debugging and auto-fixing"""
    logger.info("Starting Shake Maker ML API...")
    
    # Debug predictor model issues first
    logger.info("Running diagnostic check...")
    debug_info = ml_model.debug_model_issues()
    
    # Try to load main model
    if not ml_model.load_model():
        logger.info("Main model failed to load - attempting emergency fix...")
        success = ml_model.emergency_fix_predictor()
        if not success:
            logger.error("Emergency fix failed - API will use rule-based predictions")
    else:
        logger.info("✅ Predictor model loaded successfully!")

    # Try to load generator model (this should work fine)
    if not ml_model.load_generator_model():
        logger.info("Training new generator model...")
        success = ml_model.train_generator_model()
        if not success:
            logger.error("Failed to train generator model!")
    else:
        logger.info("✅ Generator model loaded successfully!")

    logger.info("Shake Maker ML API ready!")

# API Endpoints
@app.get("/")
async def root():
    return {"message": "Shake Maker ML API", "status": "running"}


@app.post("/predict", response_model=ShakeResponse)
async def predict_shake_quality(request: ShakeRequest):
    """Predict shake quality based on ingredients"""
    try:
        if not request.ingredients:
            raise HTTPException(status_code=400, detail="No ingredients provided")
        
        # Validate ingredients
        valid_ingredients = [ing for ing in request.ingredients 
                           if ing in ml_model.ingredient_features]
        
        if not valid_ingredients:
            raise HTTPException(status_code=400, detail="No valid ingredients provided")
        
        # Get prediction
        result = ml_model.predict_quality(valid_ingredients)
        
        return ShakeResponse(
            quality_score=result['quality_score'],
            feedback=result['feedback'],
            emoji=result['emoji'],
            detailed_analysis=result['detailed_analysis']
        )
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Prediction error: {str(e)}")
        raise HTTPException(status_code=500, detail="Internal server error")

@app.get("/ingredients")
async def get_ingredients():
    """Get all available ingredients with their features"""
    ingredients_info = {}
    
    category_map = {
            # Fruits
            'apple': 'fruits', 'banana': 'fruits', 'strawberry': 'fruits',
            'mango': 'fruits', 'blueberry': 'fruits', 'pineapple': 'fruits',
            'watermelon': 'fruits', 'peach': 'fruits', 'cherry': 'fruits',
            'coconut': 'fruits', 'avocado': 'fruits',

            # Vegetables
            'spinach': 'vegetables', 'carrot': 'vegetables', 'beetroot': 'vegetables',
            'kale': 'vegetables', 'celery': 'vegetables', 'cauliflower': 'vegetables',
            'sweet_potato': 'vegetables', 'pumpkin': 'vegetables', 'parsley': 'vegetables',
            'mint': 'vegetables',

            # Cakes
            'chocolate': 'cakes', 'vanilla': 'cakes', 'red_velvet': 'cakes',
            'cheesecake': 'cakes', 'lemon': 'cakes', 'strawberry_cake': 'cakes',
            'tiramisu': 'cakes', 'black_forest': 'cakes', 'coconut_cake': 'cakes',
    
            # Syrups
            'honey': 'syrups', 'maple': 'syrups', 'chocolate_syrup': 'syrups',
            'caramel': 'syrups', 'vanilla_syrup': 'syrups', 'strawberry_syrup': 'syrups'
        }
    
    for ingredient, features in ml_model.ingredient_features.items():
        ingredients_info[ingredient] = {
            'features': dict(zip(ml_model.feature_names, features)),
            'category': category_map.get(ingredient, 'unknown')
        }
    
    return ingredients_info

@app.post("/retrain")
async def retrain_model():
    """Retrain the model (admin endpoint)"""
    try:
        logger.info("Retraining model...")
        success = ml_model.train_model()
        
        if success:
            return {"message": "Model retrained successfully", "status": "success"}
        else:
            raise HTTPException(status_code=500, detail="Model retraining failed")
            
    except Exception as e:
        logger.error(f"Retraining error: {str(e)}")
        raise HTTPException(status_code=500, detail="Internal server error")
    

# ==== 6. ADD NEW API ENDPOINT ====
@app.post("/generate", response_model=GenerateResponse)
async def generate_shake(request: GenerateRequest):
    """Generate new shake suggestions based on preferences"""
    try:
        # Validate target score
        if request.target_score and (request.target_score < 0.0 or request.target_score > 1.0):
            raise HTTPException(status_code=400, detail="Target score must be between 0.0 and 1.0")
        
        # Validate categories
        valid_categories = ['fruits', 'vegetables', 'cakes', 'syrups']
        if request.categories:
            invalid_cats = [cat for cat in request.categories if cat not in valid_categories]
            if invalid_cats:
                raise HTTPException(status_code=400, detail=f"Invalid categories: {invalid_cats}")
        
        # Validate number of ingredients
        if request.num_ingredients and (request.num_ingredients < 1 or request.num_ingredients > 6):
            raise HTTPException(status_code=400, detail="Number of ingredients must be between 1 and 6")
        
        # Generate shake suggestion
        result = ml_model.generate_shake_suggestion(request)
        
        return GenerateResponse(
            generated_shake=result['generated_shake'],
            predicted_score=result['predicted_score'],
            confidence=result['confidence'],
            reasoning=result['reasoning'],
            alternatives=result['alternatives']
        )
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Generation error: {str(e)}")
        raise HTTPException(status_code=500, detail="Internal server error")

# ==== 7. ADD GENERATOR STATUS TO HEALTH CHECK ====
# Modify the existing health endpoint:
# added later but is more completely fulfilling the requirements ---> of both the models i.e is generator & predictor model
# have one health endpoint that shows the status of both your predictor model and generator model, which is exactly what want for monitoring the API.
@app.get("/health")
async def health_check():
    logger.info("Health check requested")
    return {
        "status": "healthy",
        "model_ready": ml_model.is_trained,
        "generator_ready": ml_model.is_generator_trained,  # Add this line
        "version": "1.0.0"
    }

# ==== 8. ADD NEW ENDPOINT FOR GENERATOR RETRAINING ====
@app.post("/retrain-generator")
async def retrain_generator():
    """Retrain the generator model (admin endpoint)"""
    try:
        logger.info("Retraining generator model...")
        success = ml_model.train_generator_model()
        
        if success:
            return {"message": "Generator model retrained successfully", "status": "success"}
        else:
            raise HTTPException(status_code=500, detail="Generator model retraining failed")
            
    except Exception as e:
        logger.error(f"Generator retraining error: {str(e)}")
        raise HTTPException(status_code=500, detail="Internal server error")


@app.post("/debug-predictor")
async def debug_predictor_model():
    """Debug endpoint to diagnose predictor model issues"""
    try:
        debug_info = ml_model.debug_model_issues()
        return {
            "status": "debug_complete",
            "debug_info": debug_info,
            "predictor_ready": ml_model.is_trained,
            "generator_ready": ml_model.is_generator_trained
        }
    except Exception as e:
        logger.error(f"Debug error: {str(e)}")
        raise HTTPException(status_code=500, detail="Debug failed")

@app.post("/emergency-fix")
async def emergency_fix_predictor():
    """Emergency fix endpoint for predictor model"""
    try:
        logger.info("Emergency fix requested...")
        success = ml_model.emergency_fix_predictor()
        
        if success:
            return {"message": "Emergency fix successful", "status": "success", "predictor_ready": True}
        else:
            return {"message": "Emergency fix failed", "status": "failed", "predictor_ready": False}
            
    except Exception as e:
        logger.error(f"Emergency fix error: {str(e)}")
        raise HTTPException(status_code=500, detail="Emergency fix failed")
    


if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)
