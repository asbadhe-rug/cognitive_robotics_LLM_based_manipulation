"""
Comprehensive test suite for LLM coordinate generation.
Tests prompt interpretation, object selection, and coordinate quality.
"""

from metrics import CoordinateMetricsTracker, print_metrics_summary
from datetime import datetime
from collections import Counter

# =============================================================================
# TEST CONFIGURATION
# =============================================================================

# Level 1: Basic geometric shapes with object count variations
LEVEL_1_TESTS = {
    "Make a circle with all objects": [
        {
            "objects": ["Strawberry", "Pear", "Banana", "TennisBall", "Strawberry", "Pear", "Banana"],
            "expected_count": 7,
            "repeat": 5
        },
        {
            "objects": ["Strawberry", "Pear", "Banana", "TennisBall", "Strawberry", "Pear", "Banana", "Hammer"],
            "expected_count": 8,
            "repeat": 5
        }
    ],
    "Arrange all objects to create a square": [
        {
            "objects": ["Strawberry", "Pear", "Banana", "TennisBall", "Strawberry", "Pear"],
            "expected_count": 6,
            "repeat": 5
        },
        {
            "objects": ["Strawberry", "Pear", "Banana", "TennisBall", "Strawberry", "Pear", "MustardBottle"],
            "expected_count": 7,
            "repeat": 5
        }
    ],
    "Form a triangle with all objects": [
        {
            "objects": ["Strawberry", "Pear", "Banana", "TennisBall", "Strawberry"],
            "expected_count": 5,
            "repeat": 5
        },
        {
            "objects": ["Strawberry", "Pear", "Banana", "TennisBall", "Strawberry", "Pear"],
            "expected_count": 6,
            "repeat": 5
        }
    ]
}

# Level 2: Object type selection (fruits, tennis balls, yellow objects)
LEVEL_2_TESTS = {
    "Make a circle with all fruits": [
        {
            "objects": ["Strawberry", "Pear", "Banana", "Strawberry", "Pear", "TennisBall", "Hammer"],
            "expected_objects": ["Strawberry", "Pear", "Banana", "Strawberry", "Pear"],
            "expected_count": 5,
            "repeat": 3
        },
        {
            "objects": ["Strawberry", "Pear", "Banana", "Strawberry", "Pear", "Banana", "Strawberry", "TennisBall", "MustardBottle"],
            "expected_objects": ["Strawberry", "Pear", "Banana", "Strawberry", "Pear", "Banana", "Strawberry"],
            "expected_count": 7,
            "repeat": 3
        }
    ],
    "Make a frame-like structure with all tennis balls": [
        {
            "objects": ["TennisBall", "TennisBall", "TennisBall", "TennisBall", "TennisBall", "TennisBall", "Strawberry", "Pear"],
            "expected_objects": ["TennisBall", "TennisBall", "TennisBall", "TennisBall", "TennisBall", "TennisBall"],
            "expected_count": 6,
            "repeat": 3
        },
        {
            "objects": ["TennisBall", "TennisBall", "TennisBall", "TennisBall", "TennisBall", "TennisBall", "TennisBall", "TennisBall", "Strawberry", "Pear"],
            "expected_objects": ["TennisBall", "TennisBall", "TennisBall", "TennisBall", "TennisBall", "TennisBall", "TennisBall", "TennisBall"],
            "expected_count": 8,
            "repeat": 3
        }
    ],
    "Make a triangle with all yellow objects": [
        {
            "objects": ["Banana", "MustardBottle", "Banana", "MustardBottle", "Strawberry", "Pear", "TennisBall"],
            "expected_objects": ["Banana", "MustardBottle", "Banana", "MustardBottle"],
            "expected_count": 4,
            "repeat": 3
        },
        {
            "objects": ["Banana", "MustardBottle", "Banana", "MustardBottle", "Banana", "Strawberry", "Pear"],
            "expected_objects": ["Banana", "MustardBottle", "Banana", "MustardBottle", "Banana"],
            "expected_count": 5,
            "repeat": 3
        }
    ]
}

# Level 3: Complex constraints (color + type, position, size)
LEVEL_3_TESTS = {
    "Make a square with all green fruits": [
        {
            "objects": ["Pear", "Pear", "Pear", "Pear", "Pear", "Strawberry", "Strawberry", "Strawberry", "TennisBall", "TomatoSoupCan"],
            "expected_objects": ["Pear", "Pear", "Pear", "Pear", "Pear"],
            "expected_count": 5,
            "repeat": 5
        }
    ],
    "Make a line with 4 objects on the top side of the table": [
        {
            "objects": ["Strawberry", "Pear", "Banana", "TennisBall", "TennisBall", "TennisBall", "TennisBall"],
            "expected_count": 4,  # Any 4 objects
            "repeat": 5
        }
    ],
    "Form a ring with 6 objects smaller than a banana": [
        {
            "objects": ["Strawberry", "Strawberry", "Strawberry", "Pear", "Pear", "Pear", "Pear", "Pear", "Banana", "Hammer"],
            "expected_objects": ["Strawberry", "Strawberry", "Strawberry", "Pear", "Pear", "Pear"],
            "expected_count": 6,
            "repeat": 5
        }
    ]
}

# Models to test
MODELS = [
    #"meta-llama/Meta-Llama-3-8B-Instruct",  # Uncomment when ready
    "gpt-3.5-turbo",
    # "mistralai/Mistral-7B-Instruct-v0.2"  # Uncomment when ready
]


# =============================================================================
# EVALUATION HELPERS
# =============================================================================

def evaluate_object_selection(selected_objects, expected_objects=None, expected_count=None):
    """
    Evaluate if the correct objects were selected.

    Returns:
        dict with:
        - correct: bool
        - score: 0.0 to 1.0
        - details: explanation
    """

    # ------------------------------------------------------------------
    # Case 1: Explicit expected objects (with duplicates allowed)
    # ------------------------------------------------------------------
    if expected_objects is not None:
        selected_counter = Counter(selected_objects)
        expected_counter = Counter(expected_objects)

        # Exact multiset match
        if selected_counter == expected_counter:
            return {
                "correct": True,
                "score": 1.0,
                "details": f"Correctly selected {dict(selected_counter)}"
            }

        # Partial credit: count-aware overlap
        correct_items = sum(
            min(selected_counter[obj], expected_counter[obj])
            for obj in expected_counter
        )
        total_expected = sum(expected_counter.values())

        score = correct_items / total_expected if total_expected > 0 else 0.0

        # Diagnostics
        missing = {
            obj: expected_counter[obj] - selected_counter.get(obj, 0)
            for obj in expected_counter
            if selected_counter.get(obj, 0) < expected_counter[obj]
        }
        extra = {
            obj: selected_counter[obj] - expected_counter.get(obj, 0)
            for obj in selected_counter
            if selected_counter[obj] > expected_counter.get(obj, 0)
        }

        details = (
            f"Expected: {dict(expected_counter)}, "
            f"Selected: {dict(selected_counter)}."
        )
        if missing:
            details += f" Missing: {missing}."
        if extra:
            details += f" Extra: {extra}."

        return {
            "correct": False,
            "score": score,
            "details": details
        }

    # ------------------------------------------------------------------
    # Case 2: Only expected count is specified
    # ------------------------------------------------------------------
    elif expected_count is not None:
        selected_count = len(selected_objects)

        if selected_count == expected_count:
            return {
                "correct": True,
                "score": 1.0,
                "details": f"Correctly selected {expected_count} objects"
            }

        # Linear penalty for count mismatch
        score = 1.0 - abs(selected_count - expected_count) / expected_count
        score = max(0.0, score)

        return {
            "correct": False,
            "score": score,
            "details": f"Selected {selected_count}, expected {expected_count}"
        }

    # ------------------------------------------------------------------
    # Case 3: No explicit expectation (fallback)
    # ------------------------------------------------------------------
    return {
        "correct": len(selected_objects) > 0,
        "score": 1.0 if len(selected_objects) > 0 else 0.0,
        "details": f"Selected {len(selected_objects)} objects"
    }


# =============================================================================
# MAIN TEST RUNNER
# =============================================================================

def run_comprehensive_tests():
    """Run all test levels with detailed tracking."""
    
    tracker = CoordinateMetricsTracker("comprehensive_results.json")
    tracker.load_existing_results()
    
    print("="*80)
    print("COMPREHENSIVE LLM COORDINATE GENERATION TEST SUITE")
    print("="*80)
    print(f"Models: {len(MODELS)}")
    print(f"Test Levels: 3")
    print(f"Total unique tests: {sum(len(tests) for tests in [LEVEL_1_TESTS, LEVEL_2_TESTS, LEVEL_3_TESTS] for tests in tests.values())}")
    print("="*80)
    
    from llm_utils import llm_generate_coordinates
    
    all_test_levels = [
    ("Level 1: recognize shape", LEVEL_3_TESTS)
    ]
    
    for model in MODELS:
        model_start_time = datetime.now()
        run_id = f"{model.split('/')[-1]}_{model_start_time.strftime('%Y%m%d_%H%M%S')}"
        
        print(f"\n{'='*80}")
        print(f"TESTING MODEL: {model}")
        print(f"Run ID: {run_id}")
        print(f"{'='*80}\n")
        
        for level_name, test_suite in all_test_levels:
            print(f"\n{'-'*80}")
            print(f"{level_name}")
            print(f"{'-'*80}\n")
            
            for instruction, test_configs in test_suite.items():
                print(f"\nInstruction: '{instruction}'")
                
                for config_idx, config in enumerate(test_configs, 1):
                    objects = config["objects"]
                    print("AVAILABLE OBJECTS: ", objects)
                    expected_objects = config.get("expected_objects")
                    expected_count = config.get("expected_count")
                    repeat = config.get("repeat", 1)
                    
                    print(f"\n  Config {config_idx}: {len(objects)} objects")
                    if expected_objects:
                        print(f"  Expected selection: {expected_objects}")
                    elif expected_count:
                        print(f"  Expected count: {expected_count}")
                    
                    for rep in range(repeat):
                        if repeat > 1:
                            print(f"\n    Repetition {rep + 1}/{repeat}")
                        
                        # Generate coordinates
                        coords, decision = llm_generate_coordinates(
                            objects, instruction, model
                        )
                        
                        if not decision or not coords:
                            print("    ✗ Failed to generate")
                            continue
                        
                        # Evaluate object selection
                        selected_objects = decision.get("selected_objects", [])
                        selection_eval = evaluate_object_selection(
                            selected_objects,
                            expected_objects=expected_objects,
                            expected_count=expected_count
                        )
                        
                        # Evaluate coordinates (standard metrics)
                        metrics = tracker.evaluate_generation(
                            generated_coords=coords,
                            schema=decision["schema"],
                            instruction=instruction,
                            model_name=model,
                            objects=objects,
                            run_id=run_id
                        )
                        
                        # Add object selection evaluation to metrics
                        metrics["object_selection"] = selection_eval
                        metrics["expected_objects"] = expected_objects
                        metrics["expected_count"] = expected_count
                        
                        # Update success criteria to include object selection
                        original_success = metrics["success"]
                        metrics["success"] = (original_success and 
                                            selection_eval["correct"])
                        
                        tracker.save_metrics(metrics)
                        
                        # Print concise results
                        status = "✓" if metrics["success"] else "✗"
                        print(f"    {status} ", end="")
                        print(f"Prompt: {'✓' if metrics['prompt_interpretation']['correct'] else '✗'}, ", end="")
                        print(f"Selection: {'✓' if selection_eval['correct'] else '✗'} ({selection_eval['score']:.2f}), ", end="")
                        print(f"Coords: {'✓' if metrics['coordinate_validity']['all_valid'] else '✗'}, ", end="")
                        print(f"Shape: {metrics['shape_quality']['score']:.2f}")
                        
                        if not selection_eval['correct']:
                            print(f"       {selection_eval['details']}")
        
        # Generate aggregate for this model
        print(f"\n{'='*60}")
        print(f"AGGREGATE STATS FOR {model}")
        print(f"{'='*60}")
        
        aggregate = tracker.save_aggregate_stats(run_id=run_id, output_file="comprehensive_aggregates.json")
        
        print(f"Total tests: {aggregate['total_experiments']}")
        print(f"Success rate: {100*aggregate['success_rate']:.1f}%")
        print(f"Prompt interpretation: {100*aggregate['prompt_interpretation_accuracy']:.1f}%")
        
        # Object selection stats (if available)
        if aggregate.get('object_selection_accuracy') is not None:
            print(f"Object selection: {100*aggregate['object_selection_accuracy']:.1f}% ← THIS IS THE PROBLEM!")
            print(f"Avg selection score: {aggregate['avg_object_selection_score']:.3f}")
        
        print(f"Coordinate validity: {100*aggregate['coordinate_validity_rate']:.1f}%")
        print(f"Avg shape quality: {aggregate['avg_shape_quality']:.3f}")
        
        model_duration = (datetime.now() - model_start_time).total_seconds()
        print(f"Time taken: {model_duration:.1f}s")
    
    # Overall summary
    print(f"\n{'='*80}")
    print("OVERALL SUMMARY")
    print(f"{'='*80}")
    
    tracker.generate_summary_report("comprehensive_summary.txt")
    overall = tracker.save_aggregate_stats(run_id=None, output_file="comprehensive_overall.json")
    
    print(f"Total experiments: {overall['total_experiments']}")
    print(f"Overall success rate: {100*overall['success_rate']:.1f}%")
    print(f"Overall avg shape quality: {overall['avg_shape_quality']:.3f}")
    
    print("\n✓ Testing complete!")
    print(f"✓ Results: comprehensive_results.json")
    print(f"✓ Aggregates: comprehensive_aggregates.json")
    print(f"✓ Overall: comprehensive_overall.json")
    print(f"✓ Summary: comprehensive_summary.txt")


if __name__ == "__main__":
    run_comprehensive_tests()