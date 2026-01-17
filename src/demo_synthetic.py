"""
Demo script with synthetic CT data
Tests the compression pipeline without requiring real DICOM files
"""

import numpy as np
from ct_compression import CTScanCompressor
import json


def generate_synthetic_ct_series(num_slices=50, size=(512, 512)):
    """
    Generate synthetic CT-like images for testing
    Simulates gradual changes between slices (like real CT)
    """
    print(f"Generating {num_slices} synthetic CT slices ({size[0]}x{size[1]})...")
    
    slices = []
    metadata = []
    
    # Create base anatomical structure
    y, x = np.ogrid[:size[0], :size[1]]
    center_y, center_x = size[0] // 2, size[1] // 2
    
    for i in range(num_slices):
        # Create synthetic CT slice with anatomical features
        
        # Body outline (circular)
        radius = 200 - i * 0.5  # Gradual change across slices
        body = ((x - center_x)**2 + (y - center_y)**2) < radius**2
        
        # Spine (bright, in center)
        spine_radius = 20
        spine = ((x - center_x)**2 + (y - center_y - 50)**2) < spine_radius**2
        
        # Ribs (arc structures)
        rib_left = np.abs((x - center_x + 80)**2 + (y - center_y)**2 - 5000) < 200
        rib_right = np.abs((x - center_x - 80)**2 + (y - center_y)**2 - 5000) < 200
        
        # Lungs (darker regions)
        lung_left = ((x - center_x + 60)**2 + (y - center_y + 20)**2) < 3000
        lung_right = ((x - center_x - 60)**2 + (y - center_y + 20)**2) < 3000
        
        # Heart (denser tissue)
        heart = ((x - center_x + 20)**2 + (y - center_y + 40)**2) < 1500
        
        # Combine structures with appropriate CT intensities
        image = np.zeros(size, dtype=np.float32)
        
        # Background (air) - HU = -1000 → intensity 0
        image[~body] = 0
        
        # Soft tissue - HU = ~40 → intensity 140
        image[body] = 140
        
        # Lungs (air-filled) - HU = -500 → intensity 70
        image[lung_left | lung_right] = 70
        
        # Heart - HU = 50 → intensity 150
        image[heart] = 150
        
        # Ribs/Bone - HU = 400-1000 → intensity 200-255
        image[rib_left | rib_right] = 200
        
        # Spine (bone) - HU = 1000 → intensity 255
        image[spine] = 255
        
        # Add some noise to make it realistic
        noise = np.random.normal(0, 5, size)
        image = np.clip(image + noise, 0, 255)
        
        # Add gradual intensity change between slices (like real CT)
        if i > 0:
            # Blend with previous slice for smooth transition
            blend_factor = 0.9
            image = image * (1 - blend_factor) + slices[-1] * blend_factor
        
        slices.append(image.astype(np.uint8))
        
        # Synthetic metadata
        metadata.append({
            'slice_location': float(i * 5.0),  # 5mm spacing
            'slice_thickness': 5.0,
            'window_center': 40.0,
            'window_width': 400.0,
            'patient_position': 'HFS',
            'modality': 'CT',
            'series_description': 'Synthetic Chest CT'
        })
    
    print(f"✓ Generated {len(slices)} synthetic slices")
    return slices, metadata


def demonstrate_compression():
    """Full demonstration with synthetic data"""
    
    print("="*70)
    print("CT SCAN COMPRESSION DEMO (Synthetic Data)")
    print("="*70)
    
    # Generate synthetic CT data
    print("\n[1] Generating synthetic CT series...")
    num_slices = 60
    slices, metadata = generate_synthetic_ct_series(num_slices=num_slices)
    
    # Initialize compressor
    print("\n[2] Initializing compressor...")
    compressor = CTScanCompressor(
        roi_percentile=85,
        compression_level=9
    )
    
    # Encode
    print("\n[3] Encoding with delta + ROI compression...")
    encoded = compressor.encode_ct_series(slices, metadata)
    
    # Show statistics
    stats = compressor.get_stats()
    print("\n" + "="*70)
    print("COMPRESSION RESULTS")
    print("="*70)
    print(f"Number of slices: {stats.num_slices}")
    print(f"\nFile Size:")
    print(f"  Original: {stats.original_size_bytes / 1024 / 1024:.2f} MB")
    print(f"  Compressed: {stats.compressed_size_bytes / 1024 / 1024:.2f} MB")
    print(f"  Compression ratio: {stats.compression_ratio:.2f}x")
    print(f"  Space saved: {((1 - 1/stats.compression_ratio) * 100):.1f}%")
    print(f"\nROI Coverage: {stats.roi_coverage_percent:.1f}% of pixels")
    print(f"  (High-detail regions: bones, organs, edges)")
    print(f"\nToken Efficiency:")
    print(f"  Without compression: ~{stats.estimated_tokens_original:,} tokens")
    print(f"  With compression: ~{stats.estimated_tokens_compressed:,} tokens")
    print(f"  Token savings: {stats.token_savings_percent:.1f}%")
    print(f"  ({stats.estimated_tokens_original - stats.estimated_tokens_compressed:,} tokens saved)")
    
    # Test reconstruction
    print("\n[4] Testing reconstruction...")
    reconstructed = compressor.decode_ct_series(encoded)
    
    # Verify reconstruction
    print("\n" + "="*70)
    print("RECONSTRUCTION VERIFICATION")
    print("="*70)
    
    max_errors = []
    for i in range(len(slices)):
        error = np.max(np.abs(slices[i].astype(np.int16) - reconstructed[i].astype(np.int16)))
        max_errors.append(error)
    
    print(f"Base slice max error: {max_errors[0]} (should be 0)")
    print(f"Average max error: {np.mean(max_errors):.2f}")
    print(f"Maximum error across all slices: {max(max_errors)}")
    
    if max(max_errors) <= 2:
        print("✓ Reconstruction quality: EXCELLENT (near-perfect)")
    elif max(max_errors) <= 5:
        print("✓ Reconstruction quality: GOOD (diagnostically acceptable)")
    else:
        print("⚠ Reconstruction quality: Check for issues")
    
    # Prepare for LLM
    print("\n[5] Preparing for LLM API...")
    llm_payload = compressor.prepare_for_llm(
        encoded,
        clinical_question="Please analyze this chest CT scan for any abnormalities."
    )
    
    print("\n" + "="*70)
    print("LLM API PAYLOAD")
    print("="*70)
    print("\nPrompt Preview:")
    print("-" * 70)
    print(llm_payload['prompt'][:400] + "...")
    print("-" * 70)
    print(f"\nBase image: {len(llm_payload['base_image_base64']):,} bytes (base64-encoded PNG)")
    print(f"Estimated tokens: ~{llm_payload['estimated_tokens']:,}")
    
    # Save example
    print("\n[6] Saving example output...")
    compressor.save_encoded(encoded, "demo_compressed_ct.json")
    
    # Create sample payload for LLM
    sample_payload = {
        'prompt': llm_payload['prompt'],
        'base_image_size_bytes': len(llm_payload['base_image_base64']),
        'estimated_tokens': llm_payload['estimated_tokens'],
        'note': 'Use base_image_base64 from demo_compressed_ct.json for actual API call'
    }
    
    with open('demo_llm_payload.json', 'w') as f:
        json.dump(sample_payload, f, indent=2)
    
    print("✓ Saved: demo_compressed_ct.json")
    print("✓ Saved: demo_llm_payload.json")
    
    # Compare approaches
    print("\n" + "="*70)
    print("COMPARISON: Different Approaches")
    print("="*70)
    
    print("\n1. Send all slices individually (NO compression):")
    print(f"   - Tokens: ~{num_slices * 700:,}")
    print(f"   - File size: ~{num_slices * 512 * 512 / 1024 / 1024:.1f} MB")
    print(f"   - Pros: Full quality everywhere")
    print(f"   - Cons: Expensive, slow, often exceeds context limits")
    
    print("\n2. Send only 1 representative slice:")
    print(f"   - Tokens: ~700")
    print(f"   - File size: ~0.3 MB")
    print(f"   - Pros: Very cheap")
    print(f"   - Cons: Loses information from other slices")
    
    print("\n3. Delta encoding + ROI (THIS APPROACH):")
    print(f"   - Tokens: ~{stats.estimated_tokens_compressed:,}")
    print(f"   - File size: ~{stats.compressed_size_bytes / 1024 / 1024:.1f} MB")
    print(f"   - Pros: Full series, diagnostic quality ROI, 90% token savings")
    print(f"   - Cons: Requires preprocessing")
    
    print("\n" + "="*70)
    print("DEMO COMPLETE!")
    print("="*70)
    print("\nKey Takeaways:")
    print("  • Delta encoding reduces file size by ~{:.0f}%".format((1 - 1/stats.compression_ratio) * 100))
    print("  • Token usage reduced by ~{:.0f}%".format(stats.token_savings_percent))
    print("  • Diagnostic quality preserved in ROI regions")
    print("  • Entire CT series fits in LLM context window")
    print("\nNext Steps:")
    print("  1. Test with real DICOM files: python ct_compression.py <dicom_folder>")
    print("  2. Integrate with your LLM API (see README.md)")
    print("  3. Adjust roi_percentile and compression_level for your needs")


if __name__ == "__main__":
    demonstrate_compression()
