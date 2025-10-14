"""
Manifest loading and processing functions.
"""

# Define what gets exported
__all__ = ['pd2r']

"""
Helper function to add repelled labels to plotnine plots using adjustText
"""
from rpy2.robjects import r, globalenv
from rpy2.robjects import pandas2ri
from rpy2.robjects.conversion import localconverter
import sys
import io
from contextlib import redirect_stdout, redirect_stderr

# Check and install packages once
_r_loaded = False

def ensure_r_packages():
    global _r_loaded
    if not _r_loaded:
        # Capture all output
        with redirect_stdout(io.StringIO()), redirect_stderr(io.StringIO()):
            r('''
            suppressMessages(suppressWarnings({
                if (!require("jgplot2", quietly = TRUE)) {
                    if (!require("devtools", quietly = TRUE)) {
                        install.packages("devtools", repos = "https://cloud.r-project.org", quiet = TRUE)
                    }
                    devtools::install_github("jeffmgranja/jgplot2", quiet = TRUE, upgrade = "never")
                }
                
                library(ggplot2)
                library(ggrepel)
                library(jgplot2)
            }))
            
            pal_cart <<- c("#e0f3db", "#6BC291", "#18B5CB", "#2E95D2", "#28154C")
            pal_cart2 <<- colorRampPalette(c("#e0f3db", "#6BC291", "#18B5CB", "#2E95D2", "#28154C"))(100)
            
            theme_small_margin <<- function(width = 0.2){
                theme(plot.margin = unit(c(width, width, width, width), "cm"))
            }
            
            cbio_dpal <<- function(){
                function(n){
                    .cbio <- list(
                        "1" = c("#28154C"),
                        "2" = c("lightgrey", "#28154C"),
                        "3" = c("lightgrey", "#18B5CB", "#28154C"),
                        "4" = c("lightgrey", "#6BC291", "#2E95D2", "#28154C"),
                        "5" = c("lightgrey", "#6BC291", "#18B5CB", "#2E95D2", "#28154C"),
                        "6" = c("lightgrey", "#E0F3DB", "#6BC291", "#18B5CB", "#2E95D2", "#28154C")
                    )
                    
                    if(n <= 6){
                        pal <- .cbio[[as.character(n)]]
                    } else if(n > 6 & n <= 20){
                        pal <- jgplot2:::.stallion3[[n]]
                    } else {
                        pal <- jgplot2:::.stallion3[[length(jgplot2:::.stallion3)]]
                    }
                    
                    if (n > length(pal)) {
                        pal <- colorRampPalette(pal)(n)
                    }
                    pal
                }
            }
            
            options(ggplot2.discrete.colour = function(...) discrete_scale("colour", "cbio_dpal", cbio_dpal(), ...))
            options(ggplot2.discrete.fill = function(...) discrete_scale("fill", "cbio_dpal", cbio_dpal(), ...))
            options(ggplot2.continuous.colour = function() scale_color_gradientn(colours = pal_cart2))
            options(ggplot2.continuous.fill = function() scale_fill_gradientn(colours = pal_cart2))
            ''')
        _r_loaded = True

def pd2r(name, df):
    ensure_r_packages()  # Auto-load packages on first use
    with localconverter(pandas2ri.converter):
        r_df = pandas2ri.py2rpy(df)
        globalenv[name] = r_df



