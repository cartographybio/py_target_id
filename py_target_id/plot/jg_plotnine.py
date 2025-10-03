"""
Manifest loading and processing functions.
"""

# Define what gets exported
__all__ = ['scale_color_jg', 'scale_fill_jg', 'theme_jg']
"""
jgplot2 for Python/plotnine
Port of Jeff Granja's jgplot2 R package color palettes and theme for plotnine
"""

from plotnine import theme_bw, element_text, element_rect, element_line, element_blank
from plotnine import scale_color_gradientn, scale_fill_gradientn, scale_color_manual, scale_fill_manual
import matplotlib.pyplot as plt
import numpy as np
import warnings
from plotnine.exceptions import PlotnineWarning
warnings.filterwarnings('ignore', category=PlotnineWarning)

# ==============================================================================
# COLOR PALETTES (from palettes.R)
# ==============================================================================

# Stallion3 - adaptive discrete palette (1-20 colors)
STALLION3 = {
    1: ["#000000"],
    2: ["#CD2626", "#1874CD"],
    3: ["#CD2626", "#009E73", "#1874CD"],
    4: ["#CD2626", "#009E73", "#1874CD", "#8E0496"],
    5: ["#CD2626", "#E69F00", "#009E73", "#1874CD", "#8E0496"],
    6: ["#CD2626", "#E69F00", "#FFE600", "#009E73", "#1874CD", "#8E0496"],
    7: ["#CD2626", "#E69F00", "#FFE600", "#009E73", "#83A4FF", "#1874CD", "#8E0496"],
    8: ["#CD2626", "#E69F00", "#FFE600", "#009E73", "#83A4FF", "#1874CD", "#8E0496", "#DB65D2"],
    9: ["#CD2626", "#E69F00", "#FFE600", "#009E73", "#75F6FC", "#83A4FF", "#1874CD", "#8E0496", "#DB65D2"],
    10: ["#CD2626", "#E69F00", "#FFE600", "#7BE561", "#009E73", "#75F6FC", "#83A4FF", "#1874CD", "#8E0496", "#DB65D2"],
    11: ["#FF7D7D", "#CD2626", "#E69F00", "#FFE600", "#7BE561", "#009E73", "#75F6FC", "#83A4FF", "#1874CD", "#8E0496", "#DB65D2"],
    12: ["#FF7D7D", "#CD2626", "#E69F00", "#FFE600", "#7BE561", "#009E73", "#75F6FC", "#83A4FF", "#1874CD", "#7C0EDD", "#8E0496", "#DB65D2"],
    13: ["#FF7D7D", "#CD2626", "#E69F00", "#FFE600", "#7BE561", "#009E73", "#75F6FC", "#262C6B", "#83A4FF", "#1874CD", "#7C0EDD", "#8E0496", "#DB65D2"],
    14: ["#FF7D7D", "#CD2626", "#D34818", "#E69F00", "#FFE600", "#7BE561", "#009E73", "#75F6FC", "#262C6B", "#83A4FF", "#1874CD", "#7C0EDD", "#8E0496", "#DB65D2"],
    15: ["#FF7D7D", "#CD2626", "#D34818", "#E69F00", "#FFE600", "#7BE561", "#009E73", "#20C4AC", "#75F6FC", "#262C6B", "#83A4FF", "#1874CD", "#7C0EDD", "#8E0496", "#DB65D2"],
    16: ["#FF7D7D", "#CD2626", "#D34818", "#E69F00", "#FFE600", "#7BE561", "#009E73", "#20C4AC", "#75F6FC", "#262C6B", "#83A4FF", "#1874CD", "#A983F2", "#7C0EDD", "#8E0496", "#DB65D2"],
    17: ["#FF7D7D", "#CD2626", "#D34818", "#E69F00", "#FFE600", "#7BE561", "#009E73", "#20C4AC", "#75F6FC", "#262C6B", "#83A4FF", "#1874CD", "#A983F2", "#7C0EDD", "#8E0496", "#DB65D2", "#FAC0FF"],
    18: ["#FF7D7D", "#CD2626", "#7F0303", "#D34818", "#E69F00", "#FFE600", "#7BE561", "#009E73", "#20C4AC", "#75F6FC", "#262C6B", "#83A4FF", "#1874CD", "#A983F2", "#7C0EDD", "#8E0496", "#DB65D2", "#FAC0FF"],
    19: ["#FF7D7D", "#CD2626", "#7F0303", "#D34818", "#E69F00", "#845C44", "#FFE600", "#7BE561", "#009E73", "#20C4AC", "#75F6FC", "#262C6B", "#83A4FF", "#1874CD", "#A983F2", "#7C0EDD", "#8E0496", "#DB65D2", "#FAC0FF"],
    20: ["#FF7D7D", "#CD2626", "#7F0303", "#D34818", "#E69F00", "#845C44", "#FFE600", "#7BE561", "#009E73", "#20C4AC", "#75F6FC", "#262C6B", "#83A4FF", "#1874CD", "#A983F2", "#7C0EDD", "#8E0496", "#DB65D2", "#FAC0FF", "#343434"]
}

# Continuous palettes
PALETTES = {
    'solarExtra': ['#3361A5', '#248AF3', '#14B3FF', '#88CEEF', '#C1D5DC', '#EAD397', '#FDB31A', '#E42A2A', '#A31D1D'],
    'horizonExtra': ["#000436", "#021EA9", "#1632FB", "#6E34FC", "#C732D5", "#FD619D", "#FF9965", "#FFD32B", "#FFFC5A"],
    'blueYellow': ["#352A86", "#343DAE", "#0262E0", "#1389D2", "#2DB7A3", "#A5BE6A", "#F8BA43", "#F6DA23", "#F8FA0D"],
    'sambaNight': ['#1873CC', '#1798E5', '#00BFFF', '#4AC596', '#00CC00', '#A2E700', '#FFFF00', '#FFD200', '#FFA500'],
    'coolwarm': ["#4858A7", "#788FC8", "#D6DAE1", "#F49B7C", "#B51F29"],
    'whitePurple': ['#f7fcfd', '#e0ecf4', '#bfd3e6', '#9ebcda', '#8c96c6', '#8c6bb1', '#88419d', '#810f7c', '#4d004b'],
    'whiteBlue': ['#fff7fb', '#ece7f2', '#d0d1e6', '#a6bddb', '#74a9cf', '#3690c0', '#0570b0', '#045a8d', '#023858'],
    'fireworks': ["white", "#2488F0", "#7F3F98", "#E22929", "#FCB31A"],
    'beach': ["#87D2DB", "#5BB1CB", "#4F66AF", "#F15F30", "#F7962E", "#FCEE2B"],
    'horizon': ['#000075', '#2E00FF', '#9408F7', '#C729D6', '#FA4AB5', '#FF6A95', '#FF8B74', '#FFAC53', '#FFCD32', '#FFFF60'],
    'greenBlue': ['#e0f3db', '#ccebc5', '#a8ddb5', '#4eb3d3', '#2b8cbe', '#0868ac', '#084081'],
    'cartography': ["lightgrey", "#e0f3db", "#6BC291", "#18B5CB", "#2E95D2", "#28154C", "#000000"],
}

# Discrete palettes
DISCRETE_PALETTES = {
    'stallion': ["#D51F26", "#272E6A", "#208A42", "#89288F", "#F47D2B", "#FEE500", "#8A9FD1", "#C06CAB", "#E6C2DC",
                 "#90D5E4", "#89C75F", "#F37B7D", "#9983BD", "#D24B27", "#3BBCA8", "#6E4B9E", "#0C727C", "#7E1416", "#D8A767", "#3D3D3D"],
    'calm': ["#7DD06F", "#844081", "#688EC1", "#C17E73", "#484125", "#6CD3A7", "#597873", "#7B6FD0", "#CF4A31", "#D0CD47",
             "#722A2D", "#CBC594", "#D19EC4", "#5A7E36", "#D4477D", "#403552", "#76D73C", "#96CED5", "#CE54D1", "#C48736"],
    'kelly': ["#FFB300", "#803E75", "#FF6800", "#A6BDD7", "#C10020", "#CEA262", "#817066", "#007D34", "#F6768E", "#00538A",
              "#FF7A5C", "#53377A", "#FF8E00", "#B32851", "#F4C800", "#7F180D", "#93AA00", "#593315", "#F13A13", "#232C16"],
}

# ==============================================================================
# HELPER FUNCTIONS
# ==============================================================================

def get_stallion3_palette(n):
    """Get stallion3 palette for n colors"""
    if n <= 20:
        return STALLION3[n]
    else:
        # Interpolate if more than 20 colors needed
        from matplotlib.colors import LinearSegmentedColormap
        base_colors = STALLION3[20]
        cmap = LinearSegmentedColormap.from_list('stallion3', base_colors, N=n)
        return [cmap(i) for i in np.linspace(0, 1, n)]

def get_continuous_palette(palette_name='solarExtra', n=256):
    """Get continuous palette interpolated to n colors"""
    from matplotlib.colors import LinearSegmentedColormap
    colors = PALETTES.get(palette_name, PALETTES['solarExtra'])
    cmap = LinearSegmentedColormap.from_list(palette_name, colors, N=n)
    return [cmap(i) for i in np.linspace(0, 1, n)]

# ==============================================================================
# PLOTNINE SCALE FUNCTIONS
# ==============================================================================

def scale_color_jg(palette='stallion3', discrete=True, n=None, **kwargs):
    """
    jgplot2 color scale for plotnine
    
    Args:
        palette: Palette name ('stallion3', 'solarExtra', etc.)
        discrete: Boolean, use discrete or continuous scale
        n: Number of colors (for discrete stallion3)
        **kwargs: Additional arguments passed to plotnine scale functions
    """
    if discrete:
        if palette == 'stallion3':
            if n is None:
                n = 20  # Default max
            colors = get_stallion3_palette(n)
            return scale_color_manual(values=colors, **kwargs)
        else:
            colors = DISCRETE_PALETTES.get(palette, STALLION3[20])
            return scale_color_manual(values=colors, **kwargs)
    else:
        colors = get_continuous_palette(palette)
        return scale_color_gradientn(colors=colors, **kwargs)

def scale_fill_jg(palette='stallion3', discrete=True, n=None, **kwargs):
    """
    jgplot2 fill scale for plotnine
    
    Args:
        palette: Palette name ('stallion3', 'solarExtra', etc.)
        discrete: Boolean, use discrete or continuous scale
        n: Number of colors (for discrete stallion3)
        **kwargs: Additional arguments passed to plotnine scale functions
    """
    if discrete:
        if palette == 'stallion3':
            if n is None:
                n = 20  # Default max
            colors = get_stallion3_palette(n)
            return scale_fill_manual(values=colors, **kwargs)
        else:
            colors = DISCRETE_PALETTES.get(palette, STALLION3[20])
            return scale_fill_manual(values=colors, **kwargs)
    else:
        colors = get_continuous_palette(palette)
        return scale_fill_gradientn(colors=colors, **kwargs)

# ==============================================================================
# THEME FUNCTION
# ==============================================================================

def theme_jg(
    color='black',
    base_family='Roboto',
    base_size=10,
    legend_position='bottom',
    legend_text_size=5,
    x_text_90=False,
    y_text_90=False,
    add_grid=True,
    discrete_palette='stallion3',
    continuous_palette='cartography'
):
    """
    jgplot2 theme for plotnine (adapted from ArchR)
    
    Returns a list containing theme + color scales that can be added to a plot.
    This automatically applies both the visual theme and color palettes.
    
    Args:
        color: Color for text, lines, ticks
        base_family: Font family
        base_size: Base font size in points
        legend_position: Legend position ('bottom', 'top', 'left', 'right')
        legend_text_size: Legend text size in points
        x_text_90: Rotate x-axis text 90 degrees
        y_text_90: Rotate y-axis text 90 degrees
        add_grid: Add grid lines
        discrete_palette: Default discrete palette ('stallion3', 'calm', 'kelly')
        continuous_palette: Default continuous palette ('cartography', 'solarExtra', etc.)
    
    Returns:
        List of [theme, scale_color, scale_fill] that applies all jgplot2 defaults
    """
    from plotnine import theme
    
    # Start with theme_bw as base
    t = (theme_bw(base_size=base_size, base_family=base_family) +
         theme(
             text=element_text(family=base_family, color=color),
             axis_text=element_text(color=color, size=base_size),
             axis_title=element_text(color=color, size=base_size),
             plot_title=element_text(color=color, size=base_size),
             panel_border=element_rect(fill='None', color=color, size=0.5),
             legend_key=element_rect(fill='transparent', color='None'),
             legend_text=element_text(color=color, size=legend_text_size),
             legend_position=legend_position,
             strip_text=element_text(size=base_size, color='black')
         ))
    
    if not add_grid:
        t = t + theme(
            panel_grid_major=element_blank(),
            panel_grid_minor=element_blank()
        )
    
    if x_text_90:
        t = t + theme(axis_text_x=element_text(angle=90, hjust=1, color=color, size=base_size))
    
    if y_text_90:
        t = t + theme(axis_text_y=element_text(angle=90, vjust=1, color=color, size=base_size))
    
    # Return theme + both color and fill scales
    # The appropriate one will be used based on the plot aesthetics
    return [
        t,
        scale_color_jg(palette=discrete_palette, discrete=True),
        scale_fill_jg(palette=continuous_palette, discrete=False)
    ]

# ==============================================================================
# UTILITY FUNCTIONS
# ==============================================================================

def show_palette(palette_name='stallion3', n=None):
    """Display a palette"""
    if palette_name == 'stallion3':
        if n is None:
            n = 20
        colors = get_stallion3_palette(n)
    elif palette_name in PALETTES:
        colors = PALETTES[palette_name]
        if n:
            colors = get_continuous_palette(palette_name, n)
    elif palette_name in DISCRETE_PALETTES:
        colors = DISCRETE_PALETTES[palette_name]
        if n:
            colors = colors[:n]
    else:
        print(f"Palette '{palette_name}' not found")
        return
    
    # Display colors
    fig, ax = plt.subplots(figsize=(10, 1))
    ax.imshow([list(range(len(colors)))], cmap=plt.matplotlib.colors.ListedColormap(colors), aspect='auto')
    ax.set_xticks([])
    ax.set_yticks([])
    ax.set_title(f"{palette_name} (n={len(colors)})")
    plt.tight_layout()
    plt.show()

def list_palettes():
    """List all available palettes"""
    print("Discrete palettes:")
    print("  - stallion3 (adaptive 1-20+ colors)")
    for name in DISCRETE_PALETTES.keys():
        print(f"  - {name}")
    print("\nContinuous palettes:")
    for name in PALETTES.keys():
        print(f"  - {name}")

# ==============================================================================
# SIMPLIFIED DEFAULT SYSTEM
# ==============================================================================

# Global defaults
_DEFAULT_DISCRETE_PALETTE = 'stallion3'
_DEFAULT_CONTINUOUS_PALETTE = 'cartography'
_DEFAULT_THEME = None

def set_jgplot2_defaults(
    discrete_palette='stallion3',
    continuous_palette='cartography'
):
    """
    Set default palettes for jgplot2 scale functions
    
    After calling this, scale_color_jg() and scale_fill_jg() will use
    these palettes by default.
    
    Args:
        discrete_palette: Default for discrete scales
        continuous_palette: Default for continuous scales
    """
    global _DEFAULT_DISCRETE_PALETTE, _DEFAULT_CONTINUOUS_PALETTE
    _DEFAULT_DISCRETE_PALETTE = discrete_palette
    _DEFAULT_CONTINUOUS_PALETTE = continuous_palette
    
    print(f"jgplot2 defaults set!")
    print(f"  Discrete: {discrete_palette}")
    print(f"  Continuous: {continuous_palette}")
    print(f"\nUse scale_color_jg() and scale_fill_jg() in your plots")


