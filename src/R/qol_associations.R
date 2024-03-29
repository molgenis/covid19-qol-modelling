#!/usr/bin/env Rscript

## ----
## Author:  C.A. (Robert) Warmerdam
## Email:   c.a.warmerdam@umcg.nl
##
## Copyright (c) C.A. Warmerdam, 2021
## 
## This program is free software: you can redistribute it and/or modify
## it under the terms of the GNU General Public License as published by
## the Free Software Foundation, either version 3 of the License, or
## (at your option) any later version.
## 
## This program is distributed in the hope that it will be useful,
## but WITHOUT ANY WARRANTY; without even the implied warranty of
## MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
## GNU General Public License for more details.
## 
## A copy of the GNU General Public License can be found in the LICENSE file in the
## root directory of this source tree. If not, see <https://www.gnu.org/licenses/>.
## ----

# Load libraries
library(argparse)
library(jsonlite)
library(tidyverse)
library(data.table)
library(readxl)
library(ggrepel)
library(gamm4)
library(mgcv)
library(broom)
library(lubridate)
library(viridis)
library(recipes)

# Declare constants
old <- theme_set(theme_classic())
theme_update(
  line = element_line(
    colour = "black", linewidth = (0.5 / (ggplot2::.pt * 72.27/96)),
    linetype = 1, lineend = "butt", arrow = F, inherit.blank = T),
  strip.background = element_rect(colour = NA, fill = NA),
  panel.grid.minor = element_blank(),
  title = element_text(colour = "#595A5C", face = "bold"),
  text = element_text(family="Helvetica", size=7),
  axis.text = element_text(size=6)
)


parser <- ArgumentParser(description = 'Plot QoL results.')
parser$add_argument('--path-excel', metavar = 'path', type = 'character',
                    help = 'Path to an excel file that describes how covariates should be processed')
parser$add_argument('--path-qol', metavar = 'path', type = 'character', required = TRUE,
                    help = 'Path to a file with data')
parser$add_argument('--path-covar', metavar = 'path', type = 'character', required = TRUE,
                    help = 'Path to a file with data')
parser$add_argument('--path-qol-pop', metavar = 'path', type = 'character', required = TRUE,
                    help = 'Path to a file with population average QoLs')


sample_n_groups = function(tbl, size, replace = FALSE, weight = NULL) {
  # regroup when done
  grps = tbl %>% groups %>% lapply(as.character) %>% unlist
  # check length of groups non-zero
  keep = tbl %>% summarise() %>% ungroup() %>% sample_n(size, replace, weight)
  # keep only selected groups, regroup because joins change count.
  # regrouping may be unnecessary but joins do something funky to grouping variable
  return(tbl %>% semi_join(keep, by=grps) %>% group_by(.dots = grps))
}

predict_gam <- function(model, x_limits, xlab = "day") {
  return(tibble(day = seq(from=x_limits[1], to=x_limits[2])) %>%
           rename(!!xlab := "day") %>%
      bind_cols(as_tibble(
        predict(model, newdata = ., se.fit=TRUE)
      )))
}

predict_gamms <- function(tbl, gam, x_limits, xlab="day") {
  return(tbl %>% mutate(data = list(predict_gam(get({{gam}}), x_limits, xlab))))
}

discretize_column <- function(values, discretization) {
  print(str(values))
  if (is.factor(values)) {
    values <- as.numeric(values)
  }
  print(str(values))
  
  if (all(c("breaks", "labels") %in% names(discretization))) {
    cut_values <- cut(
      values, breaks=discretization[["breaks"]], 
      labels=discretization[["labels"]], include.lowest=T, right=T)
    return(cut_values)
  } else if ("quantiles" %in% names(discretization)) {
    print("quantiles")
    print(discretization)
    breaks = quantile(values, probs=discretization[["quantiles"]], na.rm=T)
    return(
      cut(values, breaks=breaks, include.lowest=T, right=T, dig.lab = 2)
    )
  } else {
    return(values)
  }
}

process_column_discretize <- function(values, discretization, labels, type) {
  labelled <- values
  
  if (length(labels) > 0) {
    levels <- labels
    if (is.null(names(levels))) {
      labelled <- ordered(values, levels=levels, labels=levels)
    } else {
      labelled <- ordered(values, levels=levels, labels=names(levels))
    }
  }
  
  if (length(discretization) > 0) {
    discretized <- discretize_column(labelled, discretization)
  } else {
    discretized <- labelled
  }
  
  out <- discretized
  
  if (!is.factor(out)) {
    if (type == "discrete") {
      out <- factor(out,ordered=T)
    }
  }
  return(out)
}


process_column_test <- function(values, ordering, type) {
  out <- values
  print(ordering)
  if (length(ordering) > 0) {
    if (is.null(names(ordering))) {
      out <- factor(values, levels=ordering, labels=ordering, ordered=type=="ordinal")
    } else {
      out <- factor(values, levels=ordering, labels=names(ordering), ordered=type=="ordinal")
    }
  }
  
  if (!is.factor(out)) {
    if (type == "discrete") {
      out <- factor(out,ordered=T)
    }
  }
  return(out)
}

filter_column <- function(values, filter) {
  if (length(filter) > 0) {
    values[values %in% filter] <- NA
  }
  values[values == ""] <- NA
  return(values)
}

make_dummy_variable <- function(values) {
  pivot_wider()
}

plot_data_column <- function(column_of_interest_raw, processing_guide, qol_table, full_table, dayzero, daymax, dayinterval) {
  column_of_interest <- paste0(column_of_interest_raw, "_filtered_to_plot")
  guide_row <- processing_guide %>% filter(column_name == column_of_interest_raw)
  message(guide_row %>% pull(name))
  
  # Add the table with characteristics (some discretized) to the quality of life table
  full_tbl <- qol_table %>%
    select(project_pseudo_id, qualityoflife, responsedate, day) %>%
    inner_join(characteristics_processed, by="project_pseudo_id") %>%
    filter(!is.na(get(column_of_interest)), get(column_of_interest) != "")
  
  # Create a summarised table in which to calculate daily averages per group
  summarised_tbl <- full_tbl %>% 
    group_by(across(all_of(c("responsedate", column_of_interest)))) %>%
    summarise(qualityoflife = mean(qualityoflife, na.rm = T), n_total = n()) %>%
    filter(n_total > 50)
  
  # Calculate a GAM for the daily population average model
  #modelled_average <- gamm4(qualityoflife ~ s(day, bs="cr"), random = ~ (1|project_pseudo_id), data = qol_tib_filtered)
  modelled_average <- gam(qualityoflife ~ s(day), data = qol_table)
  
  # Predict this GAM
  predicted_average <- predict_gam(modelled_average, c(0, dayinterval)) %>%
    mutate(responsedate = dayzero + day) %>%
    rename(c(qualityoflife="fit"))
  
  predicted_average %>% filter(day == min(day) | day == max(day))
  
  # Calculate GAMs per group
  modelled <- full_tbl %>% nest_by(across(all_of(c(column_of_interest)))) %>%
    mutate(mod = list(gam(qualityoflife ~ s(day), data = data))) %>%
    select(-data) %>%
    rename(gam = mod) %>%
    mutate(eff = summary(gam)[["p.coeff"]]["day"] * dayinterval,
           eff.se = summary(gam)[["se"]]["day"] * dayinterval)
  
  # Predict this GAM
  predicted <- predict_gamms(modelled, "gam", c(0, dayinterval)) %>%
    unnest(data) %>%
    mutate(responsedate = dayzero + day) %>%
    rename(c(qualityoflife="fit")) %>%
    mutate(lab_col = as.character(get(column_of_interest)))
  
  #predicted %>% group_by(age_bins) %>% filter(day == min(day) | day == max(day))
  
  # Create table with which to label each model
  label_df <- subset(bind_rows(predicted, predicted_average), responsedate == max(responsedate)) %>%
    mutate(lab_col = case_when(is.na(lab_col) ~ "population average", TRUE ~ lab_col))
  
  # Create plot
  
  ## One layer with daily averages per group
  ## Two times two layers(geom_ribbon + geom_line) for the GAM models + Standard error
  ## One layer with that direct labels the GAM models.
  ## Scale_x_date to accomodate for extra room for the direct labels.
  ## Scale color viridis to add a na.value color that is used for the label of the average model
  ## Set title
  ## Remove the legend
  
  p <- ggplot(full_tbl,
              aes(x=responsedate, y=qualityoflife, color=.data[[column_of_interest]])) +
    geom_point(data = summarised_tbl, alpha=0.2, size=0.5, shape=16) +
    geom_ribbon(data = predicted_average, aes(x=responsedate, ymin=qualityoflife-se.fit, ymax=qualityoflife+se.fit), alpha=0.2, inherit.aes=F) +
    geom_line(data = predicted_average, aes(x=responsedate, y=qualityoflife), inherit.aes=F, color="grey70")+
    geom_ribbon(data = predicted, aes(x=responsedate, ymin=qualityoflife-se.fit, ymax=qualityoflife+se.fit, group=.data[[column_of_interest]]), alpha=0.2, inherit.aes=F) +
    geom_line(data = predicted, aes(x=responsedate, y=qualityoflife))+
    geom_text_repel(
      data = label_df,
      aes(label = str_wrap(lab_col, width=20)),
      size = 6*1/ggplot2::.pt,
      min.segment.length = 0,
      hjust = 0,
      vjust = 0.5,
      direction = "y",
      nudge_x = 24,
      segment.alpha = .5,
      segment.curvature = -0.1,
      segment.ncp = 3,
      segment.angle = 20
    ) +
    scale_x_date(
      limits = as.Date(c(dayzero, daymax + 400)), 
    ) +
    scale_color_viridis(discrete=TRUE, na.value = "grey50") +
    labs(title=guide_row %>% pull(name)) + ylab("Quality of life score (1-10)") + xlab("Date") +
    theme(legend.position="none")
  
  return(p)
  
  # # Save plot in .png or .pdf
  # ggsave(sprintf("out-%s-%s.png", column_of_interest, format(Sys.Date(), "%Y%m%d")), p,
  #        width=90, height=60, units='mm')
  # ggsave(sprintf("out-%s-%s.pdf", column_of_interest, format(Sys.Date(), "%Y%m%d")), p,
  #        width=90, height=60, units='mm')
  
}

# Main

#' Execute main
#' 
#' @param argv A vector of arguments normally supplied via command-line.
main <- function(argv=NULL) {
  if (is.null(argv)) {
    argv <- commandArgs(trailingOnly = T)
  }
  
  args <- parser$parse_args(argv)
  
  processing_guide_raw <- read_excel(path = args$path_excel, sheet="individual") 
  processing_guide <- processing_guide_raw %>%
    rowwise() %>%
    mutate(discretize = list(fromJSON(discretize)), 
           ordering = list(fromJSON(ordering)),
           filter = list(fromJSON(filter))) %>%
    ungroup()
  
  # Anne, start here:
  # Load both input tables, and also convert the general health column to an ordered factor
  covar_tib <- as_tibble(fread(args$path_covar, data.table=F))
  qol_tib <- as_tibble(fread(args$path_qol, data.table=F))
  qol_pop_tib <- as_tibble(fread(args$path_qol_pop, data.table=F))
  
  qol_table <- qol_tib %>%
    select(-any_of(processing_guide$column_name)) %>%
    mutate(day=as.numeric(difftime(responsedate, min(responsedate), units="days"))) %>%
    distinct()
  
  # Calculate the minimum, maximum dates, as well as the number of days that separate the two
  dayzero <- min(as.Date("2020-03-30"))
  daymax <- max(qol_table$responsedate)
  dayinterval <- interval(dayzero, daymax) / days(1)
  
  # Below, we calculate quartiles for each of the columns listed below, and assign the results to a new column <column>_quartiles
  # Columns to discretize
  
  # For each grouping, 
  # - perform discritization (optional)
  # - perform relabelling (optional)
  
  columns_of_interest <- processing_guide$column_name
  
  characteristics_table <- covar_tib %>% 
    select(project_pseudo_id, all_of(columns_of_interest))
  
  # TODO:
  # Associations between beta groups
  # Generate plots
  
  filter_guide <- processing_guide[["filter"]]
  names(filter_guide) <- processing_guide[["column_name"]]
  
  type_assoc_guide <- processing_guide[["type_assoc"]]
  names(type_assoc_guide) <- paste0(processing_guide[["column_name"]], "_filtered")
  
  type_plot_guide <- processing_guide[["type_plot"]]
  names(type_plot_guide) <- paste0(processing_guide[["column_name"]], "_filtered")
  
  discretization_guide <- processing_guide[["discretize"]]
  names(discretization_guide) <- paste0(processing_guide[["column_name"]], "_filtered")
  
  ordering_guide <- processing_guide[["ordering"]]
  names(ordering_guide) <- paste0(processing_guide[["column_name"]], "_filtered")
  
  characteristics_processed <- characteristics_table %>%
    mutate(across(
             all_of(columns_of_interest),
             ~ filter_column(.x, filter_guide[[cur_column()]]),
             .names = "{col}_filtered"),
           across(
             all_of(paste0(columns_of_interest, "_filtered")), 
             ~ process_column_test(
               .x, 
               ordering_guide[[cur_column()]],
               type_assoc_guide[[cur_column()]]),
             .names = "{col}_to_test"),
           across(
             all_of(paste0(columns_of_interest, "_filtered")), 
             ~ process_column_discretize(
               .x, 
               discretization_guide[[cur_column()]], 
               ordering_guide[[cur_column()]],
               type_plot_guide[[cur_column()]]),
             .names = "{col}_to_plot"))
  
  characteristics_filtered <- characteristics_processed %>%
    filter(beta_type %in% c("bottom", "around_zero", "top"))
  
  null_vs_top <- characteristics_processed %>%
    filter(beta_type %in% c("top", "around_zero")) %>%
    summarise(
      across(
        all_of(paste0(processing_guide %>% filter(type_assoc %in% c("ordinal", "continuous", "discrete")) %>% pull(column_name), "_filtered_to_test")), 
        ~ list(tidy(cor.test(
          x=as.numeric(.x), 
          y=as.numeric(beta_type_filtered_to_test), 
          method="spearman"))), .names="{col}_spearman"),
      across(
        all_of(paste0(processing_guide %>% filter(type_assoc %in% c("binary", "categorical")) %>% pull(column_name), "_filtered_to_test")), 
        ~ list(tidy(fisher.test(.x, beta_type))), .names="{col}_fisher")) 
  
  bottom_vs_null <- characteristics_processed %>%
    filter(beta_type %in% c("bottom", "around_zero")) %>%
    summarise(
      across(
        all_of(paste0(processing_guide %>% filter(type_assoc %in% c("ordinal", "continuous", "discrete")) %>% pull(column_name), "_filtered_to_test")), 
        ~ list(tidy(cor.test(
          x=as.numeric(.x), 
          y=as.numeric(beta_type_filtered_to_test), 
          method="spearman"))), .names="{col}_spearman"),
      across(
        all_of(paste0(processing_guide %>% filter(type_assoc %in% c("binary", "categorical")) %>% pull(column_name), "_filtered_to_test")), 
        ~ list(tidy(fisher.test(.x, beta_type))), .names="{col}_fisher")) 
  
  beta_bottom_vs_null <- bottom_vs_null %>%
    pivot_longer(
      cols= everything(),
      names_to = c("column_name", "test"),
      names_pattern = "(.*)_filtered_to_test_(.*)",
      values_to = c("test_out")) %>% unnest()
  
  beta_null_vs_top <- null_vs_top %>%
    pivot_longer(
      cols= everything(),
      names_to = c("column_name", "test"),
      names_pattern = "(.*)_filtered_to_test_(.*)",
      values_to = c("test_out")) %>% unnest()
  
  write.table(beta_null_vs_top, "test_beta_type_null_vs_top.tsv", quote=F, row.names=F, sep="\t")
  write.table(beta_bottom_vs_null, "test_beta_type_bottom_vs_null.tsv", quote=F, row.names=F, sep="\t")
  
  my_plots <- lapply(
    columns_of_interest, 
    plot_data_column, 
    processing_guide = processing_guide, 
    qol_table = qol_table,
    full_table = full_table,
    dayzero = dayzero,
    daymax = daymax,
    dayinterval = dayinterval
    )
  
  names(my_plots) <- columns_of_interest
  
  # main_fig_columns
  # 
  # for (column_of_interest in main_fig_columns) {
  #   p <- plot_list[column_of_interest]
  #   p + labs(title="") + facet_wrap(~column_of_interest)
  # }
  # 
  
  compiled_plot <- plot_grid(
    my_plots[["gender"]], 
    my_plots[["mean_age"]], 
    my_plots[["household_status"]], 
    my_plots[["general_health"]], 
    my_plots[["e_sum"]],
    my_plots[["income"]],
    my_plots[["mediacategory_media"]],
    my_plots[["mediacategory_social_media"]],
    align = 'hv', ncol=2,
    rel_heights = c(3, 3, 3, 3), rel_widths = c(3,3))
  
  plot_to_pdf(compiled_plot, sprintf("out-compiled-%s.pdf", format(Sys.Date(), "%Y%m%d")), width = 180/10/2.54, height = 180/10/2.54)
  
  # TODO:
  # Baseline associations between covariates and QoL
  
  characteristics_processed_with_dummy_variables <- characteristics_processed %>% 
    recipe() %>% step_dummy(all_of(c("household_status_filtered_to_test")), one_hot=T) %>% 
    prep() %>% bake(characteristics_processed)
  
  qol_summary <- inner_join(qol_table, qol_pop_tib, by = "responsedate", suffix=c("_individual", "_population")) %>% 
    group_by(project_pseudo_id) %>% 
    mutate(qualityoflife_difference = qualityoflife_individual - qualityoflife_population) %>%
    summarise(average_qol_difference = mean(qualityoflife_difference, na.rm=T))
  
  qol_mod_out <- qol_summary %>% inner_join(characteristics_processed_with_dummy_variables, by = 'project_pseudo_id') %>%
    summarise(
    across(
      all_of(paste0(processing_guide %>% filter(type_assoc %in% c("ordinal", "continuous", "discrete")) %>% pull(column_name), "_filtered_to_test")), 
      ~ list(tidy(cor.test(
        x=as.numeric(.x), 
        y=as.numeric(average_qol_difference), 
        method="spearman"))), .names="{col}_spearman"),
    across(
      all_of(paste0(processing_guide %>% filter(type_assoc %in% c("binary")) %>% pull(column_name), "_filtered_to_test")), 
      ~ list(tidy(wilcox.test(as.numeric(average_qol_difference) ~ .x))), .names="{col}_wilcox"),
    across(
      starts_with(paste0(processing_guide %>% filter(type_assoc %in% c("categorical")) %>% pull(column_name), "_filtered_to_test")), 
      ~ list(tidy(wilcox.test(as.numeric(average_qol_difference) ~ .x))), .names="{col}_wilcox"))
  
  qol_mod_results <- qol_mod_out %>%
    pivot_longer(
      cols=everything(),
      names_to = c("column_name", "test"),
      names_pattern = "(.*)_filtered_to_test_(.*)",
      values_to = c("test_out")) %>% unnest(cols = c(test_out))
  
}

if (sys.nframe() == 0 && !interactive()) {
  main()
}