def add_smart_newlines(text, max_length=20, min_remainder=5):
    """Add smart line breaks to long labels."""
    if len(text) <= max_length:
        return text
    
    parts = text.split('_')
    result_lines = []
    current_line = ""
    
    for i, part in enumerate(parts):
        test_line = part if current_line == "" else f"{current_line}_{part}"
        
        # Calculate remainder
        remaining = "_".join(parts[i+1:]) if i+1 < len(parts) else ""
        
        # Break if needed
        if (len(test_line) > max_length and 
            current_line != "" and 
            len(remaining) > min_remainder):
            result_lines.append(current_line)
            current_line = part
        else:
            current_line = test_line
    
    if current_line:
        result_lines.append(current_line)
    
    return "\n".join(result_lines)