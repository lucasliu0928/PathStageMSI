#!/usr/bin/env Rscript
# Render ensemble attention exactly as the manuscript's publication_heat maps.
# Plotting functions below are copied from 9_load_main_model_test_attention.R.
# No WSI image is required: tile rectangles use pred_map_location coordinates.

prepare_spatial_attention_heatmap_df <- function(
    ensemble_attention_df,
    sample_id,
    attention_col = "mean_att",
    tumor_fraction_min = NULL,
    tumor_fraction_col = "tumor_fraction") {
  required_cols <- c("SAMPLE_ID", "x1", "x2", "y1", "y2", attention_col)
  if (!is.null(tumor_fraction_min)) {
    required_cols <- c(required_cols, tumor_fraction_col)
  }
  missing_cols <- setdiff(required_cols, colnames(ensemble_attention_df))
  if (length(missing_cols) > 0) {
    stop(
      "Cannot make spatial heatmap because these columns are missing: ",
      paste(missing_cols, collapse = ", ")
    )
  }

  heatmap_df <- ensemble_attention_df[
    ensemble_attention_df$SAMPLE_ID == sample_id &
      !is.na(ensemble_attention_df[[attention_col]]),
    ,
    drop = FALSE
  ]

  if (nrow(heatmap_df) == 0) {
    stop("No attention rows found for SAMPLE_ID: ", sample_id)
  }

  normalize_idx <- rep(TRUE, nrow(heatmap_df))
  if (!is.null(tumor_fraction_min)) {
    if (
      !is.numeric(tumor_fraction_min) ||
        length(tumor_fraction_min) != 1L ||
        is.na(tumor_fraction_min)
    ) {
      stop("tumor_fraction_min must be NULL or one non-missing numeric value.")
    }

    normalize_idx <- !is.na(heatmap_df[[tumor_fraction_col]]) &
      heatmap_df[[tumor_fraction_col]] > tumor_fraction_min
  }

  if (!any(normalize_idx)) {
    stop(
      "No attention rows found for SAMPLE_ID ",
      sample_id,
      " with ",
      tumor_fraction_col,
      " > ",
      tumor_fraction_min,
      "."
    )
  }

  normalize_values <- heatmap_df[[attention_col]][normalize_idx]
  min_value <- min(normalize_values, na.rm = TRUE)
  max_value <- max(normalize_values, na.rm = TRUE)
  heatmap_df$attention_heatmap_value <- NA_real_
  if (is.finite(min_value) && is.finite(max_value) && max_value > min_value) {
    heatmap_df$attention_heatmap_value[normalize_idx] <- (
      normalize_values - min_value
    ) / (max_value - min_value)
  } else {
    heatmap_df$attention_heatmap_value[normalize_idx] <- 0.5
  }

  return(heatmap_df)
}


get_attention_heatmap_colors <- function(palette_name = "blue_white_red") {
  palettes <- list(
    blue_white_red = c("#2166AC", "#67A9CF", "#D1E5F0", "#F7F7F7", "#FDDBC7", "#EF8A62", "#B2182B"),
    publication_heat = c("#F7FBFF", "#C6DBEF", "#6BAED6", "#FDD0A2", "#FB6A4A", "#A50F15"),
    yellow_red = c("#FFFFCC", "#FFEDA0", "#FED976", "#FEB24C", "#F03B20", "#BD0026"),
    viridis_like = c("#440154", "#3B528B", "#21908C", "#5DC963", "#FDE725")
  )

  if (!palette_name %in% names(palettes)) {
    stop(
      "Unknown attention palette: ",
      palette_name,
      ". Available palettes are: ",
      paste(names(palettes), collapse = ", ")
    )
  }

  palettes[[palette_name]]
}


plot_spatial_attention_heatmap_df <- function(
    heatmap_df,
    plot_title = NULL,
    show_legend = TRUE,
    attention_palette = "blue_white_red",
    attention_colors = get_attention_heatmap_colors(attention_palette)) {
  if (!requireNamespace("ggplot2", quietly = TRUE)) {
    stop("Package 'ggplot2' is needed to make the spatial attention heatmap.")
  }

  required_cols <- c("x1", "x2", "y1", "y2", "attention_heatmap_value")
  missing_cols <- setdiff(required_cols, colnames(heatmap_df))
  if (length(missing_cols) > 0) {
    stop(
      "Cannot make spatial heatmap because these columns are missing: ",
      paste(missing_cols, collapse = ", ")
    )
  }

  p <- ggplot2::ggplot(heatmap_df) +
    ggplot2::geom_rect(
      ggplot2::aes(
        xmin = y1,
        xmax = y2,
        ymin = x1,
        ymax = x2,
        fill = attention_heatmap_value
      ),
      color = NA
    ) +
    ggplot2::scale_x_continuous(expand = c(0, 0)) +
    ggplot2::scale_y_reverse(expand = c(0, 0)) +
    ggplot2::coord_equal(expand = FALSE, clip = "on") +
    ggplot2::labs(title = plot_title) +
    ggplot2::theme_void(base_size = 11) +
    ggplot2::theme(
      plot.background = ggplot2::element_rect(fill = "#F2F2F2", color = NA),
      panel.background = ggplot2::element_rect(fill = "#F2F2F2", color = NA),
      legend.position = if (show_legend) "bottom" else "none",
      plot.title = ggplot2::element_text(face = "bold"),
      legend.title = ggplot2::element_blank(),
      legend.text = ggplot2::element_text(size = 26),
      legend.margin = ggplot2::margin(t = 12, b = 8)
    )

  p <- p + ggplot2::scale_fill_gradientn(
    colors = attention_colors,
    name = NULL,
    limits = c(0, 1),
    breaks = c(0, 1),
    labels = c("Low", "High"),
    na.value = "#F2F2F2"
  ) + ggplot2::guides(fill = ggplot2::guide_colorbar(
    direction = "horizontal", label.position = "bottom",
    barheight = grid::unit(6, "mm"), barwidth = grid::unit(90, "mm"), ticks = FALSE
  ))

  return(p)
}


sanitize_filename <- function(x) {
  x <- gsub("[^A-Za-z0-9._-]+", "_", x)
  x <- gsub("_+", "_", x)
  x <- gsub("^_|_$", "", x)
  substr(x, 1, 180)
}


get_spatial_plot_size <- function(heatmap_df, long_side_inches = 12, min_short_side_inches = 3) {
  display_width <- max(heatmap_df$y2, na.rm = TRUE) - min(heatmap_df$y1, na.rm = TRUE)
  display_height <- max(heatmap_df$x2, na.rm = TRUE) - min(heatmap_df$x1, na.rm = TRUE)

  if (
    !is.finite(display_width) ||
      !is.finite(display_height) ||
      display_width <= 0 ||
      display_height <= 0
  ) {
    return(c(width = long_side_inches, height = long_side_inches))
  }

  if (display_width >= display_height) {
    width <- long_side_inches
    height <- long_side_inches * display_height / display_width
  } else {
    height <- long_side_inches
    width <- long_side_inches * display_width / display_height
  }

  c(
    width = max(width, min_short_side_inches),
    height = max(height, min_short_side_inches)
  )
}


plot_ensemble_csv <- function(input_csv, output_dir, dpi = 300) {
  if (!requireNamespace("ggplot2", quietly = TRUE)) stop("Install ggplot2 in the R environment.")
  if (!is.finite(dpi) || dpi <= 0) stop("dpi must be positive.")
  data <- read.csv(input_csv, stringsAsFactors = FALSE, check.names = FALSE)
  required <- c("SAMPLE_ID", "mean_att")
  if (!all(required %in% names(data))) stop("Input requires SAMPLE_ID and mean_att columns.")
  if (!all(c("x1", "x2", "y1", "y2") %in% names(data))) {
    if (!"pred_map_location" %in% names(data)) stop("Input requires tile coordinates or pred_map_location.")
    coords <- strsplit(gsub("[()]", "", data$pred_map_location), ",\\s*")
    if (any(lengths(coords) != 4L)) stop("Each pred_map_location must contain four coordinates.")
    coords <- do.call(rbind, lapply(coords, as.numeric))
    colnames(coords) <- c("x1", "x2", "y1", "y2")
    data <- cbind(data, coords)
  }
  coord_values <- as.matrix(data[, c("x1", "x2", "y1", "y2")])
  if (any(!is.finite(coord_values))) stop("Non-finite tile coordinates.")
  if (any(data$x2 <= data$x1 | data$y2 <= data$y1)) stop("Tile rectangle bounds must increase.")
  if (any(!is.finite(data$mean_att[!is.na(data$mean_att)]))) stop("Non-finite attention scores.")
  dir.create(output_dir, recursive = TRUE, showWarnings = FALSE)
  samples <- sort(unique(data$SAMPLE_ID))
  if (!length(samples)) stop("No samples in ensemble CSV.")
  for (sample_id in samples) {
    plot_data <- prepare_spatial_attention_heatmap_df(
      data, sample_id, attention_col = "mean_att", tumor_fraction_min = NULL
    )
    p <- plot_spatial_attention_heatmap_df(
      plot_data, plot_title = NULL, show_legend = TRUE,
      attention_palette = "publication_heat"
    )
    size <- get_spatial_plot_size(plot_data, long_side_inches = 12, min_short_side_inches = 3)
    # Reserve space for the legend while preserving the tissue-panel aspect ratio.
    size[["height"]] <- size[["height"]] + 0.9
    stem <- paste0(sanitize_filename(sample_id), "_ensemble_attention")
    path <- file.path(output_dir, paste0(stem, ".png"))
    ggplot2::ggsave(path, p, width = size[["width"]], height = size[["height"]],
                   units = "in", device = "png", dpi = dpi, limitsize = FALSE, bg = "#F2F2F2")
    message("Saved ", path)
    write.csv(plot_data, file.path(output_dir, paste0(stem, "_plot_data.csv")), row.names = FALSE)
  }
  invisible(NULL)
}

main <- function() {
  args <- commandArgs(trailingOnly = TRUE)
  if (!length(args) || "--help" %in% args) {
    cat("Rscript inference/Utils/plot_attention.R --input ensemble_attention_df.csv --output_dir plots [--dpi 300]\n")
    return(invisible(NULL))
  }
  if (length(args) %% 2L != 0L) stop("Options require a value.")
  options <- list(dpi = "300")
  for (i in seq(1, length(args), by = 2)) {
    key <- sub("^--", "", args[[i]])
    if (!key %in% c("input", "output_dir", "dpi")) stop("Unknown option: ", args[[i]])
    options[[key]] <- args[[i + 1L]]
  }
  if (is.null(options$input) || is.null(options$output_dir)) stop("Provide --input and --output_dir.")
  plot_ensemble_csv(options$input, options$output_dir, dpi = as.numeric(options$dpi))
}

if (sys.nframe() == 0L) main()
