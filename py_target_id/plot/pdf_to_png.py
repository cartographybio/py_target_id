"""
Manifest loading and processing functions.
"""

# Define what gets exported
__all__ = ['pdf_to_png']

def pdf_to_png(pdf_path, dpi=300, output_path=None):
    """
    Convert PDF to high-resolution PNG image.
    
    Parameters
    ----------
    pdf_path : str
        Path to input PDF file
    dpi : int
        Output resolution in DPI (default 300)
    output_path : str, optional
        Output path. If None, replaces .pdf with .png
        
    Returns
    -------
    str or None
        Path to output PNG file, or None if error
    """
    import fitz  # PyMuPDF
    from PIL import Image
    
    try:
        doc = fitz.open(pdf_path)
        page = doc[0]
        
        # Convert DPI to zoom factor (PDF default is 72 DPI)
        zoom = dpi / 72
        mat = fitz.Matrix(zoom, zoom)
        
        pix = page.get_pixmap(matrix=mat, alpha=False)
        
        if output_path is None:
            output_path = pdf_path.replace('.pdf', '.png')
        
        img = Image.frombytes('RGB', [pix.width, pix.height], pix.samples)
        img.save(output_path, format='PNG', optimize=True)
        doc.close()
        
        return output_path
    except Exception as e:
        print(f'Error processing {pdf_path}: {e}')
        return None
