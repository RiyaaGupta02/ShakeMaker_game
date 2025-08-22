# test_generator.py
# Import your main class from your API file
# This script is for testing the AI Chef Generator functionality in debug mode.
# R generator_model is also working fine in separate file, but here we are testing the API logic --> so simply based on rule based logic seeing how things are working.
# It simulates requests to the generator and prints the results. --> N that way here we aren't trying to train again generator_model


from main import ShakeMakerML  # File & the class name for which testing --> separate file made so that API logic not interfered with

print("\n" + "="*60)
print("🧪 TESTING AI CHEF GENERATOR - DEBUG MODE")
print("="*60)

# Initialize your shake predictor
shake_predictor = ShakeMakerML()  #  actual class name

try:
    # Test 1: Simple generation request
    print("\n📋 TEST 1: Basic Generation")
    print("-" * 30)
    
    # Create test request (adjust based on your GenerateRequest structure)
    class TestRequest:
        def __init__(self):
            self.target_score = 0.8
            self.num_ingredients = 3
            self.preferences = {
                'sweetness': 0.8,
                'creaminess': 0.6,
                'nutrition': 0.7
            }
            self.avoid_ingredients = []
            self.categories = ['fruits', 'syrups']
    
    test_req = TestRequest()
    result = shake_predictor.generate_shake_suggestion(test_req)
    
    print(f"🥤 Generated Ingredients: {result['generated_shake']}")
    print(f"📊 Raw Score: {result['predicted_score']:.4f}")
    print(f"📈 Percentage: {int(result['predicted_score'] * 100)}%")
    
    # Format ingredients nicely
    ingredients_display = " + ".join([ing.replace('_', ' ').title() for ing in result['generated_shake']])
    print(f"✨ Display Format: {ingredients_display}")
    
    # Test multiple generations
    print("\n📋 TEST 2: Multiple Generations")
    print("-" * 30)
    
    for i in range(5):
        result = shake_predictor.generate_shake_suggestion(test_req)
        ingredients_nice = " + ".join([ing.replace('_', ' ').title() for ing in result['generated_shake']])
        score_pct = int(result['predicted_score'] * 100)
        print(f"Generation {i+1}: {ingredients_nice} → {score_pct}%")

except Exception as e:
    print(f"❌ Error during testing: {str(e)}")
    import traceback
    traceback.print_exc()