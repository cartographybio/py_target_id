"""
Manifest loading and processing functions.
"""

# Define what gets exported
__all__ = ['scale_color_jg', 'scale_fill_jg', 'theme_jg']

"""
Helper function to add repelled labels to plotnine plots using adjustText
"""

from adjustText import adjust_text
import warnings
import sys
import io


def add_labels_repel(plotnine_plot, data, x, y, label,
                     fontsize=8,
                     label_bg='white',
                     label_edge='black',
                     label_alpha=0.95,
                     label_pad=0.4,
                     segment_color='gray',
                     segment_alpha=0.6,
                     segment_size=0.5,
                     force_text=(1.0, 1.0),
                     force_points=(0.05, 0.05),
                     expand_text=(1.4, 1.4),
                     lim=1500,
                     precision=0.001,
                     verbose=False):
    """
    Add repelled text labels to a plotnine plot using adjustText.
    
    Parameters
    ----------
    plotnine_plot : plotnine ggplot object
        The base plot to add labels to
    data : pandas DataFrame
        Data containing the points to label
    x : str
        Column name for x coordinates
    y : str
        Column name for y coordinates
    label : str
        Column name for label text
    fontsize : float, default=8
        Font size for labels
    label_bg : str, default='white'
        Background color for label boxes
    label_edge : str, default='black'
        Edge color for label boxes
    label_alpha : float, default=0.95
        Transparency of label boxes (0-1)
    label_pad : float, default=0.4
        Padding around text in label box
    segment_color : str, default='gray'
        Color of connector lines
    segment_alpha : float, default=0.6
        Transparency of connector lines (0-1)
    segment_size : float, default=0.5
        Width of connector lines
    force_text : tuple, default=(1.0, 1.0)
        Repulsion force between labels (x, y)
    force_points : tuple, default=(0.05, 0.05)
        Repulsion force from data points (x, y)
    expand_text : tuple, default=(1.4, 1.4)
        Expansion factor for label boxes (x, y)
    lim : int, default=1500
        Maximum iterations for adjustment algorithm
    precision : float, default=0.001
        Convergence threshold
    verbose : bool, default=False
        If True, print progress messages
    
    Returns
    -------
    fig : matplotlib Figure
        The figure with added labels
    
    Examples
    --------
    >>> p = (ggplot(df, aes(x='x', y='y')) + geom_point() + theme_minimal())
    >>> labeled = df[df['quality'] > 90]
    >>> fig = add_labels_repel(p, labeled, 'x', 'y', 'gene_name')
    >>> fig.savefig('plot.pdf', dpi=300, bbox_inches='tight')
    """
    
    # Draw the plotnine plot to get matplotlib figure (without displaying)
    fig = plotnine_plot.draw(show=False)
    ax = fig.axes[0]
    
    # Create text labels (suppress matplotlib warnings)
    texts = []
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        for idx, row in data.iterrows():
            txt = ax.text(
                row[x],
                row[y],
                row[label],
                fontsize=fontsize,
                ha='center',
                va='center',
                bbox=dict(
                    boxstyle=f'round,pad={label_pad}',
                    facecolor=label_bg,
                    edgecolor=label_edge,
                    linewidth=0.5,
                    alpha=label_alpha
                ),
                zorder=1000  # Ensure labels are on top
            )
            texts.append(txt)
    
    # Apply adjustText to resolve overlaps (suppress output)
    if not verbose:
        # Redirect stdout to suppress adjustText's print statements
        old_stdout = sys.stdout
        sys.stdout = io.StringIO()
    
    try:
        with warnings.catch_warnings():
            if not verbose:
                warnings.simplefilter("ignore")
            
            adjust_text(
                texts,
                ax=ax,
                arrowprops=dict(
                    arrowstyle='-',
                    color=segment_color,
                    lw=segment_size,
                    alpha=segment_alpha
                ),
                force_text=force_text,
                force_points=force_points,
                expand_text=expand_text,
                lim=lim,
                precision=precision,
                only_move={'text': 'xy'}
            )
    finally:
        if not verbose:
            # Restore stdout
            sys.stdout = old_stdout
    
    return fig
