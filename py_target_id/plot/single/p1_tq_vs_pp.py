"""
Manifest loading and processing functions.
"""

# Define what gets exported
__all__ = ['p1_tq_vs_pp']


def p1_tq_vs_pp(df, out="plot.pdf", target_q=80, ppos=15):

    # Subset high-quality targets for labeling
    g1 = tid.plot.utils.get_top_n_per_interval(
        df, 
        x_col='P_Pos_Per', 
        y_col='TargetQ_Final_v1',
        label_col='gene_name',
        n=2,
        interval=10
    )["gene_name"].values

    g2 = df[(df["TargetQ_Final_v1"] >= target_q) & (df["P_Pos_Per"] * 100 >= ppos)]["gene_name"].values
    genes = np.unique(np.concatenate([g1, g2]))
    sub_df = df[df["gene_name"].isin(genes)]

    # Send data to R
    tid.plot.pd2r("df", df)
    tid.plot.pd2r("sub_df", sub_df)
    
    # Create plot in R
    r(f'''
    p <- ggplot(df, aes(P_Pos_Per * 100, TargetQ_Final_v1, 
                        fill = pmin(log2(SC_2nd_Target_Val + 1), 2))) +
        geom_point(pch = 21, size = 2, color = "black") +
        geom_label_repel(
            data = sub_df, 
            aes(label = gene_name), 
            fill = "white", 
            max.overlaps = Inf,
            size = 2
        ) +
        labs(
            title = "Target Quality vs Positive Percentage",
            x = "Positive Percentage",
            y = "Target Quality Final",
            fill = "log2(SC 2nd Target + 1)"
        ) +
        theme_jg()
    
    ggsave("{out}", p, width = 8, height = 8, dpi = 600)
    ''')
