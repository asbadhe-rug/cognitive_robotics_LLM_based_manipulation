import numpy as np
import json
import math
from datetime import datetime
from typing import List, Dict, Tuple


def convert_to_json_serializable(obj):
    """Convert numpy types to native Python types for JSON serialization."""
    if isinstance(obj, np.bool_):
        return bool(obj)
    elif isinstance(obj, np.integer):
        return int(obj)
    elif isinstance(obj, np.floating):
        return float(obj)
    elif isinstance(obj, np.ndarray):
        return obj.tolist()
    elif isinstance(obj, dict):
        return {k: convert_to_json_serializable(v) for k, v in obj.items()}
    elif isinstance(obj, list):
        return [convert_to_json_serializable(item) for item in obj]
    return obj


class CoordinateMetricsTracker:
    """
    Simplified metrics tracker focusing on:
    1. Prompt interpretation accuracy (did LLM choose correct schema?)
    2. Coordinate generation accuracy (are coordinates valid and reasonable?)
    """
    
    def __init__(self, results_file="coordinate_metrics.json"):
        self.results_file = results_file
        self.all_results = []
        
    def evaluate_prompt_interpretation(self, instruction: str, chosen_schema: str) -> Dict:
        """
        Check if the LLM correctly interpreted the user's instruction.
        Returns: {
            "correct": bool,
            "expected": str,
            "chosen": str,
            "score": 1.0 or 0.0
        }
        """
        instruction_lower = instruction.lower()
        
        # Define expected schemas for common instructions
        expected_schema = None
        if "circle" in instruction_lower:
            expected_schema = "circle"
        elif "square" in instruction_lower:
            expected_schema = "square"
        elif "triangle" in instruction_lower:
            expected_schema = "triangle"
        elif "line" in instruction_lower or "row" in instruction_lower:
            expected_schema = "line"
        
        correct = (chosen_schema == expected_schema) if expected_schema else True
        
        return {
            "correct": correct,
            "expected": expected_schema,
            "chosen": chosen_schema,
            "score": 1.0 if correct else 0.0
        }
    
    def calculate_shape_quality(self, coords: List[Dict], schema: str) -> Dict:
        """
        Calculate shape quality score based on how well coordinates match the intended shape.
        
        Returns a score from 0.0 to 1.0 where:
        - 1.0 = perfect shape
        - 0.0 = completely wrong shape
        """
        if not coords or len(coords) < 2:
            return {"score": 0.0, "details": "Insufficient coordinates"}
        
        coord_list = [(p['x'], p['y']) for p in coords]
        n = len(coord_list)
        
        if schema == "circle":
            # Calculate center
            cx = np.mean([p[0] for p in coord_list])
            cy = np.mean([p[1] for p in coord_list])
            
            # Calculate radii from center
            radii = [math.sqrt((x - cx)**2 + (y - cy)**2) for x, y in coord_list]
            mean_radius = np.mean(radii)
            
            # Quality = how consistent the radii are
            # Perfect circle = all radii equal
            if mean_radius == 0:
                return {"score": 0.0, "details": "All points at center"}
            
            radius_variance = np.var(radii)
            circularity_score = 1.0 / (1.0 + 10 * radius_variance)  # Scale variance
            
            return {
                "score": circularity_score,
                "details": f"Radius variance: {radius_variance:.4f}"
            }
            
        elif schema in ["square", "triangle"]:
            # Calculate edge lengths
            edge_lengths = []
            for i in range(n):
                x1, y1 = coord_list[i]
                x2, y2 = coord_list[(i + 1) % n]
                length = math.sqrt((x2 - x1)**2 + (y2 - y1)**2)
                edge_lengths.append(length)
            
            # Quality = how consistent the edge lengths are
            # Perfect shape = all edges equal
            mean_length = np.mean(edge_lengths)
            if mean_length == 0:
                return {"score": 0.0, "details": "All points at same location"}
            
            edge_variance = np.var(edge_lengths)
            regularity_score = 1.0 / (1.0 + 10 * edge_variance)
            
            return {
                "score": regularity_score,
                "details": f"Edge variance: {edge_variance:.4f}"
            }
            
        elif schema == "line":
            # Fit a line and measure deviation
            xs = np.array([p[0] for p in coord_list])
            ys = np.array([p[1] for p in coord_list])
            
            # Fit line: y = mx + c
            A = np.vstack([xs, np.ones(len(xs))]).T
            m, c = np.linalg.lstsq(A, ys, rcond=None)[0]
            
            # Calculate deviations
            deviations = []
            for x, y in coord_list:
                y_pred = m * x + c
                deviation = abs(y - y_pred)
                deviations.append(deviation)
            
            mean_deviation = np.mean(deviations)
            linearity_score = 1.0 / (1.0 + 10 * mean_deviation)
            
            return {
                "score": linearity_score,
                "details": f"Mean deviation: {mean_deviation:.4f}"
            }
        
        return {"score": 0.5, "details": "Unknown schema"}
    
    def evaluate_coordinate_validity(self, coords: List[Dict]) -> Dict:
        """
        Check if coordinates are valid and within workspace bounds.
        Workspace: X[-0.5, 0.5], Y[-1, 0]
        
        Returns: {
            "all_valid": bool,
            "valid_count": int,
            "total_count": int,
            "validity_rate": float (0.0 to 1.0),
            "invalid_coords": list of problems
        }
        """
        x_min, x_max = -0.5, 0.5
        y_min, y_max = -1, -0
        
        if not coords:
            return {
                "all_valid": False,
                "valid_count": 0,
                "total_count": 0,
                "validity_rate": 0.0,
                "invalid_coords": ["Empty coordinate list"]
            }
        
        invalid = []
        valid_count = 0
        
        for i, coord in enumerate(coords):
            x, y = coord.get('x', 0), coord.get('y', 0)
            is_valid = True
            
            if not (x_min <= x <= x_max):
                invalid.append(f"Point {i}: X={x:.3f} out of bounds")
                is_valid = False
            if not (y_min <= y <= y_max):
                invalid.append(f"Point {i}: Y={y:.3f} out of bounds")
                is_valid = False
            
            if is_valid:
                valid_count += 1
        
        return {
            "all_valid": len(invalid) == 0,
            "valid_count": valid_count,
            "total_count": len(coords),
            "validity_rate": valid_count / len(coords),
            "invalid_coords": invalid
        }
    
    def evaluate_generation(self, 
                           generated_coords: List[Dict],
                           schema: str,
                           instruction: str,
                           model_name: str,
                           objects: List[str],
                           run_id: str = None) -> Dict:
        """
        Complete evaluation with prompt interpretation, coordinate validity, and shape quality.
        run_id: Optional identifier for grouping results (e.g., "run_1", "config_A")
        """
        # 1. Prompt interpretation
        interpretation = self.evaluate_prompt_interpretation(instruction, schema)
        
        # 2. Coordinate validity
        validity = self.evaluate_coordinate_validity(generated_coords)
        
        # 3. Shape quality
        shape_quality = self.calculate_shape_quality(generated_coords, schema)
        
        # 4. Overall success (all three must be good)
        success = (interpretation["correct"] and 
                  validity["all_valid"] and 
                  shape_quality["score"] > 0.7)  # Quality threshold
        
        # Create metrics report
        metrics = {
            "timestamp": datetime.now().isoformat(),
            "run_id": run_id,  # For grouping results
            "model": model_name,
            "instruction": instruction,
            "n_objects": len(generated_coords),
            "objects": objects,
            
            # Main metrics
            "prompt_interpretation": interpretation,
            "coordinate_validity": validity,
            "shape_quality": shape_quality,
            
            # Simple pass/fail
            "success": success,
            "score": 1.0 if success else 0.0,
            
            # Raw data
            "generated_coords": generated_coords,
            "schema": schema
        }
        
        return metrics
    
    def save_metrics(self, metrics: Dict):
        """Save metrics to results file."""
        # Convert numpy types to native Python types
        metrics = convert_to_json_serializable(metrics)
        self.all_results.append(metrics)
        
        with open(self.results_file, 'w') as f:
            json.dump(self.all_results, f, indent=2)
    
    def get_aggregate_stats(self, run_id: str = None) -> Dict:
        """
        Calculate aggregate statistics for a subset of results.
        
        Args:
            run_id: If provided, only calculate stats for results with this run_id.
                   If None, calculate stats for all results.
        
        Returns:
            Dictionary with aggregate statistics
        """
        # Filter results
        if run_id:
            results = [r for r in self.all_results if r.get("run_id") == run_id]
        else:
            results = self.all_results
        
        if not results:
            return {"error": "No results found"}
        
        # Calculate aggregates
        total = len(results)
        successful = sum(1 for r in results if r.get("success", False))
        prompt_correct = sum(1 for r in results if r["prompt_interpretation"]["correct"])
        coords_valid = sum(1 for r in results if r["coordinate_validity"]["all_valid"])
        
        # Object selection stats
        obj_selection_correct = sum(
            1 for r in results if r["object_selection"]["correct"]
        )

        obj_selection_scores = [
            r["object_selection"]["score"] for r in results
        ]

        object_selection_accuracy = obj_selection_correct / total
        
        # Shape quality scores
        shape_scores = [r["shape_quality"]["score"] for r in results 
                       if "shape_quality" in r]
        
        # By schema breakdown
        by_schema = {}
        for result in results:
            schema = result["schema"]
            if schema not in by_schema:
                by_schema[schema] = {
                    "count": 0,
                    "successful": 0,
                    "shape_scores": []
                }
            by_schema[schema]["count"] += 1
            if result.get("success", False):
                by_schema[schema]["successful"] += 1
            if "shape_quality" in result:
                by_schema[schema]["shape_scores"].append(result["shape_quality"]["score"])
        
        # Calculate schema averages
        for schema in by_schema:
            scores = by_schema[schema]["shape_scores"]
            by_schema[schema]["avg_shape_score"] = np.mean(scores) if scores else 0.0
            by_schema[schema]["success_rate"] = (
                by_schema[schema]["successful"] / by_schema[schema]["count"]
            )
        
        aggregate = {
            "run_id": run_id,
            "total_experiments": total,
            "successful": successful,
            "success_rate": successful / total,
            "prompt_interpretation_accuracy": prompt_correct / total,
            "coordinate_validity_rate": coords_valid / total,
            "avg_shape_quality": np.mean(shape_scores) if shape_scores else 0.0,
            "min_shape_quality": np.min(shape_scores) if shape_scores else 0.0,
            "max_shape_quality": np.max(shape_scores) if shape_scores else 0.0,
            "by_schema": by_schema
        }
        
        return aggregate
    
    def save_aggregate_stats(self, run_id: str = None, output_file: str = None):
        """
        Save aggregate statistics to a separate file.
        
        Args:
            run_id: Identifier for this run (e.g., "run_1", "experiment_A")
            output_file: Where to save (default: aggregate_stats.json)
        """
        if output_file is None:
            output_file = "aggregate_stats.json"
        
        aggregate = self.get_aggregate_stats(run_id)
        
        # Convert numpy types
        aggregate = convert_to_json_serializable(aggregate)
        
        # Load existing aggregates
        try:
            with open(output_file, 'r') as f:
                all_aggregates = json.load(f)
        except FileNotFoundError:
            all_aggregates = []
        
        # Add timestamp
        aggregate["timestamp"] = datetime.now().isoformat()
        
        # Append this aggregate
        all_aggregates.append(aggregate)
        
        # Save
        with open(output_file, 'w') as f:
            json.dump(all_aggregates, f, indent=2)
        
        print(f"✓ Aggregate stats saved to {output_file}")
        return aggregate
    
    def load_existing_results(self):
        """Load existing results from file."""
        try:
            with open(self.results_file, 'r') as f:
                self.all_results = json.load(f)
            print(f"✓ Loaded {len(self.all_results)} existing results")
        except FileNotFoundError:
            self.all_results = []
            print(f"No existing results file found, starting fresh")
        except json.JSONDecodeError as e:
            print(f"⚠️  Warning: Existing results file is corrupted ({e})")
            print(f"   Creating backup and starting fresh...")
            
            # Backup corrupted file
            import shutil
            backup_file = f"{self.results_file}.backup_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
            try:
                shutil.copy(self.results_file, backup_file)
                print(f"   Corrupted file backed up to: {backup_file}")
            except Exception as backup_error:
                print(f"   Could not create backup: {backup_error}")
            
            # Start fresh
            self.all_results = []
    
    def generate_summary_report(self, output_file="metrics_summary.txt"):
        """Generate simple summary report."""
        if not self.all_results:
            return
        
        # Group by model
        by_model = {}
        for result in self.all_results:
            model = result["model"]
            if model not in by_model:
                by_model[model] = []
            by_model[model].append(result)
        
        with open(output_file, 'a') as f:
            f.write("=" * 80 + "\n")
            f.write("COORDINATE GENERATION ACCURACY REPORT\n")
            f.write("=" * 80 + "\n\n")
            
            for model, results in by_model.items():
                f.write(f"\n{'='*80}\n")
                f.write(f"MODEL: {model}\n")
                f.write(f"{'='*80}\n\n")
                
                # Overall statistics
                total = len(results)
                successful = sum(1 for r in results if r.get("success", False))
                prompt_correct = sum(1 for r in results if r["prompt_interpretation"]["correct"])
                coords_valid = sum(1 for r in results if r["coordinate_validity"]["all_valid"])
                
                f.write(f"Total experiments: {total}\n")
                f.write(f"Fully successful: {successful} ({100*successful/total:.1f}%)\n")
                f.write(f"Correct prompt interpretation: {prompt_correct} ({100*prompt_correct/total:.1f}%)\n")
                f.write(f"Valid coordinates: {coords_valid} ({100*coords_valid/total:.1f}%)\n")
                
                # Shape quality stats
                shape_scores = [r["shape_quality"]["score"] for r in results 
                               if "shape_quality" in r]
                if shape_scores:
                    f.write(f"Avg shape quality: {np.mean(shape_scores):.3f}\n")
                    f.write(f"Min shape quality: {np.min(shape_scores):.3f}\n")
                    f.write(f"Max shape quality: {np.max(shape_scores):.3f}\n")
                
                # Object selection stats (if available)
                selection_scores = [r["object_selection"]["score"] for r in results 
                                   if "object_selection" in r]
                if selection_scores:
                    selection_correct = sum(1 for r in results 
                                          if r.get("object_selection", {}).get("correct", False))
                    f.write(f"Object selection accuracy (all tests): "
                    f"{selection_correct}/{total} "
                    f"({100*selection_correct/total:.1f}%)\n")
                    f.write(f"Avg selection score: {np.mean(selection_scores):.3f}\n")
                
                f.write("\n")
                
                # By schema
                f.write("Performance by Schema:\n")
                f.write("-" * 40 + "\n")
                
                schemas = set(r["schema"] for r in results)
                for schema in schemas:
                    schema_results = [r for r in results if r["schema"] == schema]
                    schema_success = sum(1 for r in schema_results if r.get("success", False))
                    
                    f.write(f"\n  {schema.upper()}:\n")
                    f.write(f"    Count: {len(schema_results)}\n")
                    f.write(f"    Success rate: {100*schema_success/len(schema_results):.1f}%\n")
                
                f.write("\n" + "-" * 80 + "\n")
                
                # Individual results
                f.write("\nDetailed Results:\n")
                f.write("-" * 80 + "\n\n")
                
                for i, result in enumerate(results, 1):
                    status = "✓ PASS" if result["success"] else "✗ FAIL"
                    f.write(f"{i}. {status} - {result['instruction']}\n")
                    f.write(f"   Schema: {result['schema']} ")
                    
                    if result["prompt_interpretation"]["correct"]:
                        f.write("✓\n")
                    else:
                        f.write(f"✗ (expected: {result['prompt_interpretation']['expected']})\n")
                    
                    # Object selection (if available)
                    if "object_selection" in result:
                        sel = result["object_selection"]
                        f.write(f"   Object selection: ")
                        if sel["correct"]:
                            f.write(f"✓ {sel['details']}\n")
                        else:
                            f.write(f"✗ {sel['details']}\n")
                    
                    f.write(f"   Coordinates: {result['coordinate_validity']['valid_count']}/{result['coordinate_validity']['total_count']} valid ")
                    
                    if result["coordinate_validity"]["all_valid"]:
                        f.write("✓\n")
                    else:
                        f.write("✗\n")
                        for problem in result["coordinate_validity"]["invalid_coords"][:3]:
                            f.write(f"     - {problem}\n")
                    
                    # Add shape quality
                    if "shape_quality" in result:
                        score = result["shape_quality"]["score"]
                        f.write(f"   Shape quality: {score:.3f} ")
                        if score > 0.9:
                            f.write("✓ (excellent)\n")
                        elif score > 0.7:
                            f.write("✓ (good)\n")
                        else:
                            f.write("✗ (poor)\n")
                    
                    f.write("\n")
        
        print(f"✓ Summary report saved to {output_file}")


def print_metrics_summary(metrics: Dict):
    """Print simple summary to console."""
    print("\n" + "="*80)
    print("COORDINATE GENERATION METRICS")
    print("="*80)
    print(f"Model: {metrics['model']}")
    print(f"Instruction: {metrics['instruction']}")
    print(f"Schema: {metrics['schema']}")
    print(f"Objects: {metrics['n_objects']}")
    
    # Overall result
    if metrics["success"]:
        print(f"\n✓ SUCCESS")
    else:
        print(f"\n✗ FAILED")
    
    # Prompt interpretation
    interp = metrics["prompt_interpretation"]
    print(f"\nPrompt Interpretation:")
    if interp["correct"]:
        print(f"  ✓ Correct (chose {interp['chosen']})")
    else:
        print(f"  ✗ Incorrect (chose {interp['chosen']}, expected {interp['expected']})")
    
    # Coordinate validity
    validity = metrics["coordinate_validity"]
    print(f"\nCoordinate Validity:")
    print(f"  {validity['valid_count']}/{validity['total_count']} coordinates valid", end="")
    
    if validity["all_valid"]:
        print(" ✓")
    else:
        print(" ✗")
        if validity["invalid_coords"]:
            print("  Problems:")
            for problem in validity["invalid_coords"][:3]:
                print(f"    - {problem}")
            if len(validity["invalid_coords"]) > 3:
                print(f"    ... and {len(validity['invalid_coords'])-3} more")
    
    # Shape quality
    if "shape_quality" in metrics:
        quality = metrics["shape_quality"]
        print(f"\nShape Quality:")
        print(f"  Score: {quality['score']:.3f}", end="")
        if quality['score'] > 0.9:
            print(" ✓ (excellent)")
        elif quality['score'] > 0.7:
            print(" ✓ (good)")
        elif quality['score'] > 0.5:
            print(" ⚠ (fair)")
        else:
            print(" ✗ (poor)")
        print(f"  {quality['details']}")
    
    print("="*80 + "\n")