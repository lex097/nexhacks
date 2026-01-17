"""
CT Scan Compression for LLM APIs
Implements ROI preservation and DICOM-aware delta encoding
"""

import pydicom
import numpy as np
import blosc
import cv2
from PIL import Image
import io
import base64
from pathlib import Path
from typing import List, Dict, Tuple, Optional
import json
from dataclasses import dataclass, asdict


@dataclass
class CompressionStats:
    """Statistics from compression process"""
    num_slices: int
    original_size_bytes: int
    compressed_size_bytes: int
    compression_ratio: float
    roi_coverage_percent: float
    estimated_tokens_original: int
    estimated_tokens_compressed: int
    token_savings_percent: float


class CTScanCompressor:
    """
    CT Scan compression for LLM APIs
    Uses delta encoding between slices + ROI preservation for diagnostic quality
    """
    
    def __init__(self, roi_percentile: int = 85, compression_level: int = 9):
        """
        Args:
            roi_percentile: Gradient percentile threshold for ROI detection (85-95 recommended)
            compression_level: Blosc compression level 1-9 (9 = maximum compression)
        """
        self.roi_percentile = roi_percentile
        self.compression_level = compression_level
        self.stats = None
    
    def load_ct_series(self, dicom_folder: str) -> Tuple[List[np.ndarray], List[Dict]]:
        """
        Load CT DICOM series from folder
        
        Args:
            dicom_folder: Path to folder containing DICOM files
            
        Returns:
            images: List of normalized image arrays (uint8)
            metadata: List of dictionaries with DICOM metadata
        """
        dicom_files = sorted(Path(dicom_folder).glob('*.dcm'))
        
        if not dicom_files:
            raise ValueError(f"No DICOM files found in {dicom_folder}")
        
        slices = []
        metadata = []
        
        print(f"Loading {len(dicom_files)} DICOM files...")
        
        for dcm_path in dicom_files:
            dcm = pydicom.dcmread(str(dcm_path))
            
            # Extract pixel data
            pixel_array = dcm.pixel_array.astype(np.float32)
            
            # Apply CT windowing if available
            if hasattr(dcm, 'WindowCenter') and hasattr(dcm, 'WindowWidth'):
                # Handle multiple window values
                wc = dcm.WindowCenter
                ww = dcm.WindowWidth
                if isinstance(wc, (list, pydicom.multival.MultiValue)):
                    wc = wc[0]
                if isinstance(ww, (list, pydicom.multival.MultiValue)):
                    ww = ww[0]
                pixel_array = self._apply_ct_window(pixel_array, float(wc), float(ww))
            else:
                # Default windowing for soft tissue
                pixel_array = self._apply_ct_window(pixel_array, 40, 400)
            
            slices.append(pixel_array)
            
            # Store important metadata
            metadata.append({
                'slice_location': float(getattr(dcm, 'SliceLocation', 0)),
                'slice_thickness': float(getattr(dcm, 'SliceThickness', 1.0)),
                'window_center': self._get_numeric_value(dcm, 'WindowCenter', 40),
                'window_width': self._get_numeric_value(dcm, 'WindowWidth', 400),
                'patient_position': str(getattr(dcm, 'PatientPosition', 'Unknown')),
                'modality': str(getattr(dcm, 'Modality', 'CT')),
                'series_description': str(getattr(dcm, 'SeriesDescription', '')),
            })
        
        # Sort by slice location
        sorted_indices = sorted(range(len(metadata)), 
                              key=lambda i: metadata[i]['slice_location'])
        slices = [slices[i] for i in sorted_indices]
        metadata = [metadata[i] for i in sorted_indices]
        
        print(f"Loaded {len(slices)} slices, sorted by position")
        
        return slices, metadata
    
    def _get_numeric_value(self, dcm, attr: str, default: float) -> float:
        """Safely extract numeric value from DICOM, handling MultiValue"""
        if not hasattr(dcm, attr):
            return default
        val = getattr(dcm, attr)
        if isinstance(val, (list, pydicom.multival.MultiValue)):
            return float(val[0])
        return float(val)
    
    def _apply_ct_window(self, pixel_array: np.ndarray, center: float, width: float) -> np.ndarray:
        """
        Apply CT windowing (window/level adjustment)
        Critical for proper CT visualization
        """
        img_min = center - width / 2
        img_max = center + width / 2
        
        windowed = np.clip(pixel_array, img_min, img_max)
        windowed = ((windowed - img_min) / (img_max - img_min) * 255.0)
        
        return windowed.astype(np.uint8)
    
    def detect_roi(self, image: np.ndarray) -> np.ndarray:
        """
        Detect regions of interest using gradient and variance analysis
        High gradients/variance = edges, lesions, abnormalities
        
        Args:
            image: Input image (uint8)
            
        Returns:
            roi_mask: Boolean array where True = important region
        """
        # Convert to float for gradient computation
        img_float = image.astype(np.float32)
        
        # Compute gradients (detect edges, lesions)
        grad_x = cv2.Sobel(img_float, cv2.CV_32F, 1, 0, ksize=3)
        grad_y = cv2.Sobel(img_float, cv2.CV_32F, 0, 1, ksize=3)
        gradient_magnitude = np.sqrt(grad_x**2 + grad_y**2)
        
        # Compute local variance (detect texture, abnormalities)
        kernel_size = 7
        mean = cv2.blur(img_float, (kernel_size, kernel_size))
        mean_sq = cv2.blur(img_float**2, (kernel_size, kernel_size))
        variance = mean_sq - mean**2
        
        # Combine gradient and variance for importance map
        importance_map = gradient_magnitude + variance * 0.3
        
        # Threshold to create ROI mask
        threshold = np.percentile(importance_map, self.roi_percentile)
        roi_mask = importance_map > threshold
        
        # Morphological operations to clean up and expand ROI
        kernel = np.ones((5, 5), np.uint8)
        roi_mask = cv2.morphologyEx(roi_mask.astype(np.uint8), cv2.MORPH_CLOSE, kernel)
        roi_mask = cv2.dilate(roi_mask, kernel, iterations=2)
        
        return roi_mask.astype(bool)
    
    def compress_delta(self, delta_array: np.ndarray) -> bytes:
        """
        Compress delta array using blosc
        Blosc is optimized for numerical arrays
        
        Args:
            delta_array: Difference array (int16)
            
        Returns:
            compressed: Compressed bytes
        """
        compressed = blosc.compress(
            delta_array.tobytes(),
            typesize=delta_array.itemsize,
            cname='zstd',  # zstd offers best compression
            clevel=self.compression_level,
            shuffle=blosc.BITSHUFFLE  # Bit shuffle works well for medical images
        )
        return compressed
    
    def decompress_delta(self, compressed: bytes, shape: Tuple[int, int], dtype=np.int16) -> np.ndarray:
        """
        Decompress delta array
        
        Args:
            compressed: Compressed bytes
            shape: Original array shape
            dtype: Data type of array
            
        Returns:
            delta_array: Decompressed delta array
        """
        decompressed = blosc.decompress(compressed)
        delta_array = np.frombuffer(decompressed, dtype=dtype)
        return delta_array.reshape(shape)
    
    def encode_ct_series(self, slices: List[np.ndarray], metadata: List[Dict]) -> Dict:
        """
        Encode CT series using delta encoding with ROI preservation
        
        Args:
            slices: List of CT slice images (uint8)
            metadata: List of metadata dictionaries
            
        Returns:
            encoded: Dictionary containing compressed data structure
        """
        if len(slices) == 0:
            raise ValueError("No slices to encode")
        
        print(f"\nEncoding {len(slices)} slices...")
        
        # Storage for encoded data
        encoded = {
            'base_slice': None,
            'deltas': [],
            'roi_masks': [],
            'metadata': metadata,
            'shape': slices[0].shape,
            'num_slices': len(slices)
        }
        
        # Track statistics
        original_size = 0
        compressed_size = 0
        total_roi_pixels = 0
        total_pixels = 0
        
        # First slice: store as base (with ROI)
        base_slice = slices[0]
        base_roi = self.detect_roi(base_slice)
        
        # Convert base slice to PNG for efficient storage
        base_img = Image.fromarray(base_slice)
        buffer = io.BytesIO()
        base_img.save(buffer, format='PNG', compress_level=9)
        base_png = buffer.getvalue()
        
        encoded['base_slice'] = {
            'data': base64.b64encode(base_png).decode('utf-8'),
            'format': 'png_base64',
            'roi_mask': base_roi.tolist()
        }
        
        original_size += base_slice.nbytes
        compressed_size += len(base_png)
        total_roi_pixels += np.sum(base_roi)
        total_pixels += base_roi.size
        
        print(f"Base slice: {base_slice.nbytes / 1024:.1f} KB → {len(base_png) / 1024:.1f} KB")
        
        # Subsequent slices: store as deltas
        for i in range(1, len(slices)):
            current_slice = slices[i]
            previous_slice = slices[i-1]
            
            # Compute delta (difference from previous slice)
            delta = current_slice.astype(np.int16) - previous_slice.astype(np.int16)
            
            # Detect ROI in current slice
            current_roi = self.detect_roi(current_slice)
            
            # Separate delta into ROI and background
            roi_delta = delta.copy()
            roi_delta[~current_roi] = 0
            
            bg_delta = delta.copy()
            bg_delta[current_roi] = 0
            
            # Compress ROI delta (lossless, high quality)
            roi_compressed = self.compress_delta(roi_delta)
            
            # Compress background delta (can be more aggressive)
            bg_compressed = self.compress_delta(bg_delta)
            
            encoded['deltas'].append({
                'slice_index': i,
                'roi_delta': base64.b64encode(roi_compressed).decode('utf-8'),
                'bg_delta': base64.b64encode(bg_compressed).decode('utf-8'),
                'roi_mask': current_roi.tolist()
            })
            
            original_size += current_slice.nbytes
            compressed_size += len(roi_compressed) + len(bg_compressed)
            total_roi_pixels += np.sum(current_roi)
            total_pixels += current_roi.size
            
            if (i + 1) % 10 == 0:
                print(f"Processed {i + 1}/{len(slices)} slices...")
        
        # Calculate statistics
        compression_ratio = original_size / compressed_size if compressed_size > 0 else 0
        roi_coverage = (total_roi_pixels / total_pixels * 100) if total_pixels > 0 else 0
        
        # Estimate tokens (rough approximation)
        # Typical: 1 full-res image ≈ 500-800 tokens
        # Compressed delta ≈ 50-100 tokens
        tokens_original = len(slices) * 700
        tokens_compressed = 700 + (len(slices) - 1) * 75  # base + deltas
        token_savings = ((tokens_original - tokens_compressed) / tokens_original * 100) if tokens_original > 0 else 0
        
        self.stats = CompressionStats(
            num_slices=len(slices),
            original_size_bytes=original_size,
            compressed_size_bytes=compressed_size,
            compression_ratio=compression_ratio,
            roi_coverage_percent=roi_coverage,
            estimated_tokens_original=tokens_original,
            estimated_tokens_compressed=tokens_compressed,
            token_savings_percent=token_savings
        )
        
        print(f"\n✓ Encoding complete!")
        print(f"  Compression ratio: {compression_ratio:.2f}x")
        print(f"  Original size: {original_size / 1024 / 1024:.2f} MB")
        print(f"  Compressed size: {compressed_size / 1024 / 1024:.2f} MB")
        print(f"  ROI coverage: {roi_coverage:.1f}%")
        print(f"  Estimated token reduction: {token_savings:.1f}% ({tokens_original} → {tokens_compressed})")
        
        return encoded
    
    def decode_ct_series(self, encoded: Dict) -> List[np.ndarray]:
        """
        Decode CT series from compressed format
        
        Args:
            encoded: Encoded data dictionary
            
        Returns:
            slices: List of reconstructed CT slices
        """
        print(f"\nDecoding {encoded['num_slices']} slices...")
        
        slices = []
        shape = tuple(encoded['shape'])
        
        # Decode base slice
        base_png = base64.b64decode(encoded['base_slice']['data'])
        base_img = Image.open(io.BytesIO(base_png))
        base_slice = np.array(base_img)
        slices.append(base_slice)
        
        print(f"Base slice decoded")
        
        # Decode deltas and reconstruct slices
        for i, delta_info in enumerate(encoded['deltas']):
            # Decompress ROI and background deltas
            roi_compressed = base64.b64decode(delta_info['roi_delta'])
            bg_compressed = base64.b64decode(delta_info['bg_delta'])
            
            roi_delta = self.decompress_delta(roi_compressed, shape)
            bg_delta = self.decompress_delta(bg_compressed, shape)
            
            # Combine deltas
            delta = roi_delta + bg_delta
            
            # Reconstruct slice
            reconstructed = slices[-1].astype(np.int16) + delta
            reconstructed = np.clip(reconstructed, 0, 255).astype(np.uint8)
            
            slices.append(reconstructed)
            
            if (i + 1) % 10 == 0:
                print(f"Decoded {i + 1}/{len(encoded['deltas'])} deltas...")
        
        print(f"✓ Decoding complete!")
        
        return slices
    
    def prepare_for_llm(self, encoded: Dict, clinical_question: str = "") -> Dict:
        """
        Prepare compressed CT data for LLM API
        
        Args:
            encoded: Encoded CT data
            clinical_question: Clinical question to answer
            
        Returns:
            llm_payload: Dictionary with prompt and images ready for LLM
        """
        # Extract base slice image
        base_png = base64.b64decode(encoded['base_slice']['data'])
        
        # Create clinical context from metadata
        metadata = encoded['metadata'][0]
        
        clinical_context = f"""CT Scan Analysis Request

Series Information:
- Modality: {metadata.get('modality', 'CT')}
- Number of slices: {encoded['num_slices']}
- Slice thickness: {metadata.get('slice_thickness', 'N/A')} mm
- Series description: {metadata.get('series_description', 'N/A')}
- Window: C={metadata.get('window_center', 'N/A')} W={metadata.get('window_width', 'N/A')}

Compression Method:
This series uses delta encoding with ROI preservation. The base slice is provided in full resolution.
Subsequent slices are encoded as deltas (differences) to reduce token usage while preserving
diagnostic quality in regions of interest.

Clinical Question: {clinical_question if clinical_question else 'Please analyze this CT scan series.'}

Base slice (slice 1/{encoded['num_slices']}) is attached below.
Deltas for {encoded['num_slices'] - 1} additional slices are available if needed for detailed analysis.
"""
        
        return {
            'prompt': clinical_context,
            'base_image_base64': encoded['base_slice']['data'],
            'base_image_format': 'png',
            'encoded_data': encoded,
            'estimated_tokens': self.stats.estimated_tokens_compressed if self.stats else None
        }
    
    def save_encoded(self, encoded: Dict, output_path: str):
        """Save encoded data to JSON file"""
        # Convert numpy arrays to lists for JSON serialization
        output_path = Path(output_path)
        
        print(f"\nSaving encoded data to {output_path}...")
        
        with open(output_path, 'w') as f:
            json.dump(encoded, f, indent=2)
        
        file_size = output_path.stat().st_size
        print(f"✓ Saved: {file_size / 1024 / 1024:.2f} MB")
    
    def load_encoded(self, input_path: str) -> Dict:
        """Load encoded data from JSON file"""
        print(f"\nLoading encoded data from {input_path}...")
        
        with open(input_path, 'r') as f:
            encoded = json.load(f)
        
        print(f"✓ Loaded {encoded['num_slices']} slices")
        return encoded
    
    def get_stats(self) -> Optional[CompressionStats]:
        """Get compression statistics"""
        return self.stats


def main():
    """Example usage"""
    import sys
    
    # Example: Process a CT scan series
    if len(sys.argv) < 2:
        print("Usage: python ct_compression.py <dicom_folder> [output_json]")
        print("\nExample:")
        print("  python ct_compression.py ./ct_scans/ compressed_ct.json")
        return
    
    dicom_folder = sys.argv[1]
    output_file = sys.argv[2] if len(sys.argv) > 2 else "compressed_ct.json"
    
    # Initialize compressor
    compressor = CTScanCompressor(
        roi_percentile=85,  # 85-95 recommended
        compression_level=9  # Maximum compression
    )
    
    try:
        # Load CT series
        slices, metadata = compressor.load_ct_series(dicom_folder)
        
        # Encode with delta + ROI
        encoded = compressor.encode_ct_series(slices, metadata)
        
        # Save encoded data
        compressor.save_encoded(encoded, output_file)
        
        # Prepare for LLM
        llm_payload = compressor.prepare_for_llm(
            encoded,
            clinical_question="Are there any abnormalities visible in this CT scan?"
        )
        
        print("\n" + "="*60)
        print("LLM API Payload Preview:")
        print("="*60)
        print(llm_payload['prompt'][:500] + "...")
        print(f"\nBase image available: {len(llm_payload['base_image_base64'])} bytes (base64)")
        print(f"Estimated tokens: {llm_payload['estimated_tokens']}")
        
        # Test reconstruction
        print("\n" + "="*60)
        print("Testing Reconstruction...")
        print("="*60)
        reconstructed = compressor.decode_ct_series(encoded)
        
        # Verify reconstruction quality
        max_error = np.max(np.abs(slices[0].astype(np.int16) - reconstructed[0].astype(np.int16)))
        print(f"Reconstruction error (base slice): {max_error} (should be 0)")
        
        if len(slices) > 1:
            max_error = np.max(np.abs(slices[-1].astype(np.int16) - reconstructed[-1].astype(np.int16)))
            print(f"Reconstruction error (last slice): {max_error}")
        
        print("\n✓ All done!")
        
    except Exception as e:
        print(f"\n❌ Error: {e}")
        import traceback
        traceback.print_exc()


if __name__ == "__main__":
    main()
