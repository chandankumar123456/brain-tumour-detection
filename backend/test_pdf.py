import sys
import io
sys.path.insert(0, '.')
from report_generator import generate_pdf_report

# Test PDF generation with a sample analysis result
sample_result = {
    "status": "success",
    "tumor_volume_mm3": 438,
    "coordinates": {"x": 45.8, "y": 67.2, "z": 50.0},
    "confidence": 91.0,
    "confidence_label": "High",
    "mask_summary": {
        "whole_tumor_pixels": 2920,
        "tumor_core_pixels": 1500,
        "enhancing_tumor_pixels": 600,
    },
    "treatment_recommendation": {
        "note": "Model confidence: 91%. Tumor volume of 438 cubic mm.",
        "options": [
            {"treatment": "Radiation therapy", "detail": "Next MRI in 3 months"},
            {"treatment": "Radiotherapy", "detail": "To target residual cells"},
            {"treatment": "Chemotherapy", "detail": "Adjuvant temozolomide"},
        ],
    },
}

pdf_bytes = generate_pdf_report(sample_result)
print(f'PDF generated: {len(pdf_bytes):,} bytes')

outpath = '/tmp/test_report.pdf'
with open(outpath, 'wb') as f:
    f.write(pdf_bytes)
print(f'Saved to {outpath}')
print('PDF TEST PASSED')
