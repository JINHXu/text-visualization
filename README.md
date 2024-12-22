# text-visualization


```python
from IPython.core.display import display, HTML
import numpy as np

def highlight_text_with_gradient(tokens, shap_values, threshold=0.1):
    """
    Visualize text with gradient color intensity based on SHAP values.
    Higher SHAP values have more intense (darker) colors.
    """
    # Normalize SHAP values to a [0, 1] range for color mapping
    norm_shap_values = np.clip(shap_values, 0, max(shap_values))  # Clip negative values for color intensity
    norm_shap_values = (norm_shap_values - min(norm_shap_values)) / (max(norm_shap_values) - min(norm_shap_values))
    
    html_output = ""
    
    for token, shap_value in zip(tokens, shap_values):
        # Map SHAP value to color intensity (from light to dark red/green)
        intensity = int(norm_shap_values[0] * 255)  # Map to RGB range
        color = f"rgb({255-intensity}, {intensity}, 0)"  # Gradual change from red (negative) to green (positive)
        
        # Apply color highlight if SHAP value exceeds threshold
        if abs(shap_value) > threshold:
            html_output += f'<span style="background-color: {color}; padding: 0 5px;">{token}</span> '
        else:
            html_output += token + " "

    display(HTML(html_output))

# Example tokens and SHAP values
tokens = ["This", "is", "a", "great", "movie", "!"]
shap_values = [0.05, 0.1, -0.02, 0.5, -0.4, 0.1]  # Random SHAP values for demonstration

# Highlight text with gradient color density based on SHAP values
highlight_text_with_gradient(tokens, shap_values, threshold=0.1)
```
