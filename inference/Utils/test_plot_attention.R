#!/usr/bin/env Rscript
# Check manuscript normalization and displayed coordinate orientation.
file_arg <- grep("^--file=", commandArgs(), value = TRUE)[[1]]
this_dir <- dirname(normalizePath(sub("^--file=", "", file_arg)))
source(file.path(this_dir, "plot_attention.R"))
data <- data.frame(SAMPLE_ID = rep("sample", 3),
                   x1 = c(0, 10, 20), x2 = c(10, 20, 30),
                   y1 = c(100, 110, 120), y2 = c(110, 120, 130),
                   mean_att = c(-4, 1, 6))
prepared <- prepare_spatial_attention_heatmap_df(data, "sample")
stopifnot(isTRUE(all.equal(prepared$attention_heatmap_value, c(0, 0.5, 1))))
constant <- data
constant$mean_att <- 3
stopifnot(all(prepare_spatial_attention_heatmap_df(constant, "sample")$attention_heatmap_value == 0.5))
missing <- data
missing$mean_att[[2]] <- NA
stopifnot(nrow(prepare_spatial_attention_heatmap_df(missing, "sample")) == 2)
p <- plot_spatial_attention_heatmap_df(prepared, attention_palette = "publication_heat")
built <- ggplot2::ggplot_build(p)$data[[1]]
stopifnot(all(built$xmin == prepared$y1), all(built$xmax == prepared$y2))
stopifnot(all(built$ymin == -prepared$x1), all(built$ymax == -prepared$x2))
stopifnot(identical(get_attention_heatmap_colors("publication_heat"),
                    c("#F7FBFF", "#C6DBEF", "#6BAED6", "#FDD0A2", "#FB6A4A", "#A50F15")))
cat("Manuscript normalization, palette, missing scores and orientation checks passed.\n")
