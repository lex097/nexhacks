"""
Example usage of CT scan compression for LLM APIs
"""

from ct_compression import CTScanCompressor
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path


def visualize_compression(compressor, slices, encoded, num_slices_to_show=3):
    """
    Visualize original vs reconstructed slices with ROI overlay
    """
    # Reconstruct slices
    reconstructed = compressor.decode_ct_series(encoded)
    
    # Select evenly spaced slices to visualize
    indices = np.linspace(0, len(slices)-1, num_slices_to_show, dtype=int)
    
    fig, axes = plt.subplots(num_slices_to_show, 3, figsize=(15, 5*num_slices_to_show))
    
    if num_slices_to_show == 1:
        axes = axes.reshape(1, -1)
    
    for i, idx in enumerate(indices):
        # Original
        axes[i, 0].imshow(slices[idx], cmap='gray')
        axes[i, 0].set_title(f'Original Slice {idx}')
        axes[i, 0].axis('off')
        
        # Reconstructed
        axes[i, 1].imshow(reconstructed[idx], cmap='gray')
        axes[i, 1].set_title(f'Reconstructed Slice {idx}')
        axes[i, 1].axis('off')
        
        # ROI overlay
        roi_mask = np.array(encoded['base_slice']['roi_mask'] if idx == 0 
                           else encoded['deltas'][idx-1]['roi_mask'])
        
        # Create RGB image with ROI highlighted
        rgb_img = np.stack([slices[idx]]*3, axis=-1)
        rgb_img[roi_mask, 1] = np.minimum(255, rgb_img[roi_mask, 1] + 100)  # Green tint for ROI
        
        axes[i, 2].imshow(rgb_img.astype(np.uint8))
        axes[i, 2].set_title(f'ROI Overlay (green = important regions)')
        axes[i, 2].axis('off')
    
    plt.tight_layout()
    plt.savefig('compression_visualization.png', dpi=150, bbox_inches='tight')
    print("\n✓ Visualization saved to compression_visualization.png")
    plt.close()


def demonstrate_compression_stats(compressor):
    """Display compression statistics"""
    stats = compressor.get_stats()
    
    if not stats:
        print("No statistics available")
        return
    
    print("\n" + "="*60)
    print("COMPRESSION STATISTICS")
    print("="*60)
    print(f"Number of slices: {stats.num_slices}")
    print(f"\nFile Size:")
    print(f"  Original: {stats.original_size_bytes / 1024 / 1024:.2f} MB")
    print(f"  Compressed: {stats.compressed_size_bytes / 1024 / 1024:.2f} MB")
    print(f"  Compression ratio: {stats.compression_ratio:.2f}x")
    print(f"  Space saved: {(1 - 1/stats.compression_ratio) * 100:.1f}%")
    print(f"\nROI Coverage: {stats.roi_coverage_percent:.1f}% of pixels")
    print(f"\nToken Estimate:")
    print(f"  Without compression: ~{stats.estimated_tokens_original:,} tokens")
    print(f"  With compression: ~{stats.estimated_tokens_compressed:,} tokens")
    print(f"  Token savings: {stats.token_savings_percent:.1f}%")
    print("="*60)


def example_llm_integration(llm_payload):
    """
    Example of how to use the compressed data with an LLM API
    (This is pseudo-code - adapt to your specific LLM API)
    """
    print("\n" + "="*60)
    print("EXAMPLE LLM API INTEGRATION")
    print("="*60)
    
    # Example for OpenAI/Anthropic-style APIs
    example_code = '''
# Example with Anthropic Claude API
import anthropic
import base64

client = anthropic.Anthropic(api_key="your-api-key")

# Decode base64 image
image_data = base64.b64decode(llm_payload['base_image_base64'])

message = client.messages.create(
    model="claude-3-5-sonnet-20241022",
    max_tokens=1024,
    messages=[
        {
            "role": "user",
            "content": [
                {
                    "type": "image",
                    "source": {
                        "type": "base64",
                        "media_type": "image/png",
                        "data": llm_payload['base_image_base64'],
                    },
                },
                {
                    "type": "text",
                    "text": llm_payload['prompt']
                }
            ],
        }
    ],
)

print(message.content)
'''
    
    print(example_code)
    print("\nNote: The compressed format reduces tokens by ~90% compared to")
    print("sending all slices individually!")
    print("="*60)


def main():
    """
    Complete example workflow
    """
    import sys
    
    if len(sys.argv) < 2:
        print("CT Scan Compression Example")
        print("="*60)
        print("\nUsage: python example_usage.py <dicom_folder>")
        print("\nThis script will:")
        print("  1. Load CT DICOM series from folder")
        print("  2. Apply ROI-preserving delta compression")
        print("  3. Display compression statistics")
        print("  4. Visualize results")
        print("  5. Show LLM API integration example")
        print("\nExample:")
        print("  python example_usage.py ./ct_scans/")
        return
    
    dicom_folder = sys.argv[1]
    
    # Validate folder exists
    if not Path(dicom_folder).exists():
        print(f"Error: Folder not found: {dicom_folder}")
        return
    
    print("\n" + "="*60)
    print("CT SCAN COMPRESSION EXAMPLE")
    print("="*60)
    
    # Step 1: Initialize compressor
    print("\n[Step 1] Initializing compressor...")
    compressor = CTScanCompressor(
        roi_percentile=85,  # Detect top 15% of high-gradient/variance regions
        compression_level=9  # Maximum compression
    )
    
    # Step 2: Load CT series
    print("\n[Step 2] Loading CT DICOM series...")
    try:
        slices, metadata = compressor.load_ct_series(dicom_folder)
    except Exception as e:
        print(f"Error loading DICOM files: {e}")
        return
    
    # Step 3: Compress with delta encoding + ROI
    print("\n[Step 3] Applying compression...")
    encoded = compressor.encode_ct_series(slices, metadata)
    
    # Step 4: Display statistics
    demonstrate_compression_stats(compressor)
    
    # Step 5: Prepare for LLM
    print("\n[Step 4] Preparing for LLM API...")
    llm_payload = compressor.prepare_for_llm(
        encoded,
        clinical_question="Please analyze this chest CT scan for any abnormalities, particularly looking for signs of pneumonia, nodules, or other pathology."
    )
    
    # Step 6: Show LLM integration
    example_llm_integration(llm_payload)
    
    # Step 7: Visualize (optional)
    print("\n[Step 5] Creating visualization...")
    try:
        visualize_compression(compressor, slices, encoded, num_slices_to_show=3)
    except Exception as e:
        print(f"Visualization skipped (matplotlib might not be available): {e}")
    
    # Step 8: Save compressed data
    print("\n[Step 6] Saving compressed data...")
    output_file = "compressed_ct_data.json"
    compressor.save_encoded(encoded, output_file)
    
    print("\n" + "="*60)
    print("✓ EXAMPLE COMPLETE!")
    print("="*60)
    print("\nFiles created:")
    print(f"  - {output_file} (compressed CT data)")
    print(f"  - compression_visualization.png (if matplotlib available)")
    print("\nNext steps:")
    print("  1. Examine the visualization to see ROI detection")
    print("  2. Use the LLM payload to integrate with your API")
    print("  3. Adjust roi_percentile (85-95) for more/less ROI coverage")
    print("  4. Adjust compression_level (1-9) for speed vs size tradeoff")


if __name__ == "__main__":
    main()
