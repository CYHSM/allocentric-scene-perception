import open_clip
import torch
import inspect
import pkg_resources

def check_openclip_intermediates_support():
    """
    Check which OpenCLIP models support forward_intermediates
    without downloading pretrained weights.
    """
    # First, check OpenCLIP version
    try:
        version = pkg_resources.get_distribution("open_clip_torch").version
        print(f"OpenCLIP version: {version}")
    except:
        print("Could not determine OpenCLIP version")
    
    # Check if CLIP class has forward_intermediates method
    try:
        from open_clip.model import CLIP
        clip_has_method = hasattr(CLIP, 'forward_intermediates')
        print(f"CLIP base class has forward_intermediates: {clip_has_method}")
    except:
        clip_has_method = False
        print("Could not check CLIP base class")
    
    # List all available models
    models = open_clip.list_models()
    print(f"Found {len(models)} available models")
    
    # Test a representative model with random weights
    repr_model = "ViT-B-32"  # Small model that should be available in all versions
    print(f"\nTesting representative model: {repr_model}")
    try:
        model, _, _ = open_clip.create_model_and_transforms(
            repr_model, 
            pretrained=False  # Don't download weights
        )
        
        model_has_method = hasattr(model, 'forward_intermediates')
        print(f"Model has forward_intermediates: {model_has_method}")
        
        if model_has_method:
            # Check if the method is implemented by looking at its source
            try:
                source = inspect.getsource(model.forward_intermediates)
                is_stub = "pass" in source and len(source.strip().split('\n')) <= 3
                print(f"Method appears to be a stub: {is_stub}")
                
                if not is_stub:
                    # Extract parameters to check expected behavior
                    params = inspect.signature(model.forward_intermediates).parameters
                    has_expected_params = ('image' in params and 
                                         'image_indices' in params)
                    print(f"Method has expected parameters: {has_expected_params}")
            except:
                print("Could not inspect method source")
    except Exception as e:
        print(f"Error creating test model: {e}")
        model_has_method = False
    
    # Conclusion
    if clip_has_method or model_has_method:
        print("\nConclusion: Your OpenCLIP version appears to support forward_intermediates.")
        print("Most models should be able to extract intermediate features.")
        
        # If both checks are positive, we're very confident
        if clip_has_method and model_has_method:
            print("Support level: High confidence")
        else:
            print("Support level: Moderate confidence")
        
        # List architecture types that are likely to work
        print("\nThese architecture types likely support intermediate features:")
        print("- ViT models (ViT-B-32, ViT-L-14, etc.)")
        print("- EVA models")
        print("- CLIP models")
        # Less certain about ResNet-based models
        print("\nLess certain about:")
        print("- ResNet-based models (RN50, etc.)")
        
    else:
        print("\nConclusion: Your OpenCLIP version likely does NOT support forward_intermediates.")
        print("You may need to update to a newer version or use a different approach.")
    
    return {
        "clip_has_method": clip_has_method,
        "model_has_method": model_has_method,
        "available_models": models
    }

def check_specific_architectures():
    """
    Test specific model architectures for forward_intermediates support
    without downloading weights.
    """
    # List of model architectures to test
    # These are representative of different architecture families
    architectures_to_test = [
        #"ViT-B-32",    # Basic ViT
        "EVA02-E-14",  # EVA model
    ]
    
    results = {}
    
    print("Testing specific OpenCLIP architectures...")
    
    for arch in architectures_to_test:
        print(f"\nTesting {arch}...")
        
        try:
            # Create the model without downloading weights
            model, _, _ = open_clip.create_model_and_transforms(
                arch, 
                pretrained=False  # No weights download
            )
            
            # Check for method
            has_method = hasattr(model, 'forward_intermediates')
            
            results[arch] = {
                "supports_intermediates": has_method
            }
            
            print(f"{'✓' if has_method else '✗'} {arch} {'supports' if has_method else 'does NOT support'} forward_intermediates")
            
            # Clean up to save memory
            del model
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
            
        except Exception as e:
            results[arch] = {
                "error": str(e)
            }
            print(f"! Error testing {arch}: {e}")
    
    # Summarize findings
    print("\nSummary:")
    supports_count = sum(1 for r in results.values() if r.get("supports_intermediates", False))
    print(f"- {supports_count}/{len(architectures_to_test)} tested architectures support forward_intermediates")
    
    if supports_count > 0:
        print("\nThe following architectures support intermediate features:")
        for arch, info in results.items():
            if info.get("supports_intermediates", False):
                print(f"- {arch}")
    
    return results

if __name__ == "__main__":
    print("Checking OpenCLIP forward_intermediates support...\n")
    check_openclip_intermediates_support()
    
    print("\n" + "="*50 + "\n")
    
    print("Testing specific model architectures...\n")
    check_specific_architectures()
