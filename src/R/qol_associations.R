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
  
  qol_table <- qol_tib %>%
    select(-any_of(processing_guide$column_name)) %>%
    mutate(day=as.numeric(difftime(responsedate, min(responsedate), units="days")))
  
  # 
  # # Filtered with enough samples
  # qol_tib_filtered <- qol_tib %>% group_by(project_pseudo_id) %>%
  #   filter(n() > 15) %>%
  #   ungroup() %>%
  #   mutate(day=as.numeric(difftime(responsedate, min(responsedate), units="days")))

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
  
  fisher_out <- characteristics_processed %>%
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
        ~ list(tidy(fisher.test(.x, beta_type))), .names="{col}_fisher")) %>%
    unnest(fisher_test)
  
  for (column_of_interest_raw in columns_of_interest) {
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
      geom_point(data = summarised_tbl, alpha=0.2, size=1, shape=16) +
      geom_ribbon(data = predicted_average, aes(x=responsedate, ymin=qualityoflife-se.fit, ymax=qualityoflife+se.fit), alpha=0.2, inherit.aes=F) +
      geom_line(data = predicted_average, aes(x=responsedate, y=qualityoflife), inherit.aes=F, color="grey70")+
      geom_ribbon(data = predicted, aes(x=responsedate, ymin=qualityoflife-se.fit, ymax=qualityoflife+se.fit, group=.data[[column_of_interest]]), alpha=0.2, inherit.aes=F) +
      geom_line(data = predicted, aes(x=responsedate, y=qualityoflife))+
      geom_text_repel(
        data = label_df,
        aes(label = lab_col),
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
    
    # Save plot in .png or .pdf
    ggsave(sprintf("out-%s-%s.png", column_of_interest, format(Sys.Date(), "%Y%m%d")), p,
           width=180, height=120, units='mm')
    ggsave(sprintf("out-%s-%s.pdf", column_of_interest, format(Sys.Date(), "%Y%m%d")), p,
           width=180, height=120, units='mm')
  }
  
  #plot(y=df_merge$num_participants_filter,x=df_merge$date_send, xlab="date",ylab="Num participants",col="black", type = "h") 
  #points(y=df_merge$num_participants_filter, xlab="date", pch=16,col="red", type = "p", size=2)
    
}

if (sys.nframe() == 0 && !interactive()) {
  main()
}