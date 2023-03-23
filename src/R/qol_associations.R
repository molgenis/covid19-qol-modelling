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
library(rjson)
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
  text = element_text(family="Helvetica", size=7)
)


parser <- ArgumentParser(description = 'Plot QoL results.')
parser$add_argument('--path-excel', metavar = 'path', type = 'character',
                    help = 'Path to an excel file that describes how covariates should be processed')
parser$add_argument('--path-qol', metavar = 'path', type = 'character', required = TRUE,
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
  if (c("breaks", "labels") == names(discretization)) {
    print(discretization)
    return(
      cut(values, breaks=discretization["breaks"], 
          labels=discretization["labels"], include.lowest=T, right=T))
  } else if (c("quantiles") == names(discretization)) {
    print(discretization)
    breaks = quantile(values, probs=discretization["quantiles"])
    return(
      cut(values, breaks=breaks, include.lowest=T, right=T, dig.lab = 2)
    )
  } else {
    return(values)
  }
}

process_column <- function(values, guide) {
  print(guide)
  discretization <- guide[["discretize"]]
  message("discretization")
  print(discretization)
  print(labels)
  print(values)
  if (length(discretization) > 0) {
    discretized <- discretize_column(values, discretization)
  } else {
    discretized <- values
  }
  levels <- guide[["labels"]]
  print(table(discretized))
  print(levels)
  if (is.null(names(levels))) {
    return(ordered(discretized, levels=levels, labels=levels))
  }
  return(ordered(discretized, levels=levels, labels=names(levels)))
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
           labels = list(fromJSON(labels))) %>%
    ungroup()
  
  # Anne, start here:
  # Load both input tables, and also convert the general health column to an ordered factor
  qol_tib <- as_tibble(fread(args$path_qol, data.table=F))
  
  # Filtered with enough samples
  qol_tib_filtered <- qol_tib %>% group_by(project_pseudo_id) %>%
    filter(n() > 15) %>%
    ungroup() %>%
    mutate(day=as.numeric(difftime(responsedate, min(responsedate), units="days")))

    
  # Calculate the minimum, maximum dates, as well as the number of days that separate the two
  dayzero <- min(as.Date("2020-03-30"))
  daymax <- max(qol_tib_filtered$responsedate)
  dayinterval <- interval(dayzero, daymax) / days(1)
  
  # Below, we calculate quartiles for each of the columns listed below, and assign the results to a new column <column>_quartiles
  # Columns to discretize
  
  # For each grouping, 
  # - perform discritization (optional)
  # - perform relabelling (optional)
  
  columns_of_interest <- processing_guide$column_name
  columns_of_interest <- columns_of_interest[columns_of_interest != "age"]
  
  characteritics_table <- qol_tib %>% 
    group_by(project_pseudo_id) %>%
    distinct(across(all_of(columns_of_interest)), .keep_all=T) %>%
    slice_min(responsedate) %>% select(all_of(columns_of_interest))
  
  # TODO:
  # Associations between beta groups
  # Generate plots
  
  characteristics_processed <- characteritics_table %>%
    mutate(across(all_of(columns_of_interest), 
           ~ process_column(.x, processing_guide[processing_guide$column_name==cur_column(),], 
           .names = "{.col}_processed")))
  
  fisher_out <- characteristics_processed %>%
    filter(!is.na(beta_type)) %>%
    summarise(across(
      all_of(c("")), 
      ~ list(tidy(cor.test(.x, beta_type, "spearman"))))) %>%
    unnest(fisher_test)

  # Add the table with characteristics (some discretized) to the quality of life table
  full_tbl <- qol_tib_filtered %>%
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
  modelled_average <- gam(qualityoflife ~ s(day), data = qol_tib_filtered)
  
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
    labs(title=column_of_interest) + ylab("Quality of life score (1-10)") + xlab("Date") +
    theme(legend.position="none")
  
  # Save plot in .png or .pdf
  ggsave(sprintf("out-%s-%s.png", column_of_interest, format(Sys.Date(), "%Y%m%d")), p,
         width=160, height=120, units='mm')
  ggsave(sprintf("out-%s-%s.pdf", column_of_interest, format(Sys.Date(), "%Y%m%d")), p,
         width=160, height=120, units='mm')
  
  dates <- as.Date(c(
    "30/03/2020",
    "06/04/2020",
    "13/04/2020",
    "20/04/2020",
    "27/04/2020",
    "04/05/2020",
    "18/05/2020",
    "01/06/2020",
    "15/06/2020",
    "06/07/2020",
    "13/07/2020",
    "10/08/2020",
    "07/09/2020",
    "12/10/2020",
    "02/11/2020",
    "17/11/2020",
    "30/11/2020",
    "14/12/2020",
    "11/01/2021",
    "01/03/2021",
    "29/03/2021",
    "26/04/2021",
    "25/05/2021",
    "05/07/2021",
    "11/10/2021",
    "20/12/2021",
    "28/02/2022",
    "11/04/2022",
    "30/05/2022",
    "11/07/2022",
    "03/10/2022"), format = "%d/%m/%Y")
  
num_quest_map <- c("04"=4, "1"=1, "10"=10, "11"=11, "12"=12, "13"=13, "14"=14, "15"=15, "15b"=16, "16"=17, "16b"=18,
  "17"=19, "18"=20, "19"=21, "2"=2, "20"=22, "21"=23, "22"=24, "23"=25, "24"=26, "25"=27, "26"=28,
  "27"=29, "28"=30, "29"=31, "3"=3, "4"=4, "5"=5, "6"=6, "7"=7, "8"=8, "9"=9)

num_quest_tib <- tibble(quest_num_index = sort(num_quest_map), quest_num_label = names(sort(num_quest_map))) %>%
  mutate(quest_date = dates[quest_num_index])
  
  p <- ggplot(qol_tib_filtered,
              aes(x=responsedate)) +
    geom_point(data=tibble(startdate=dates), aes(startdate, y = 1)) +
    geom_histogram(binwidth=7, linewidth=0, fill="#E6E6E6") +
    scale_x_date(
      limits = as.Date(c(dayzero, daymax)), 
    ) +
    coord_cartesian(expand=expansion(add=c(7,0), mult=c(0.00, 0.00)))+
    ylab("y") + xlab("Date") +
    theme(legend.position="none", aspect.ratio=1/3.5)
  
  # Save plot in .png or .pdf
  ggsave(sprintf("out-qol-responsedensity-%s.png", format(Sys.Date(), "%Y%m%d")), p,
         width=120, height=45, units='mm')
  ggsave(sprintf("out-qol-responsedensity-%s.pdf", format(Sys.Date(), "%Y%m%d")), p,
         width=120, height=45, units='mm')
  
  summarised_tbl <- qol_table %>% 
    distinct(project_pseudo_id, num_quest, responsedate) %>%
    inner_join(num_quest_tib, by=c("num_quest" = "quest_num_label")) %>%
    filter(quest_num_index!=1) %>%
    inner_join(qol_tib, by = c("project_pseudo_id", "responsedate")) %>%
    group_by(quest_date) %>%
    summarise(qualityoflife = mean(qualityoflife, na.rm = T), n_total = n())

  p <- ggplot(summarised_tbl,
              aes(x=quest_date, y=qualityoflife)) +
    geom_point(colour="#009E73", shape=16) +
    geom_line(colour="#009E73") +
    scale_x_date(
      limits = as.Date(c(dayzero, daymax)), 
    ) +
    coord_cartesian(expand=expansion(add=c(7,0), mult=c(0.00, 0.00)))+
    ylab("y") + xlab("Date") +
    theme(legend.position="none", aspect.ratio=1/3.5)
  
  # Save plot in .png or .pdf
  ggsave(sprintf("out-qol-simpleqol-%s.png", format(Sys.Date(), "%Y%m%d")), p,
         width=120, height=45, units='mm')
  ggsave(sprintf("out-qol-simpleqol-%s.pdf", format(Sys.Date(), "%Y%m%d")), p,
         width=120, height=45, units='mm')
  
  # Path
  #path_files <- "C:/Users/EwijkA/OneDrive - UMCG/models_QOL/dataPlots/"
  path_files <- "/groups/umcg-lifelines/tmp01/projects/ov20_0554/umcg-aewijk/QOL/"
  # Read file
  num_quest_filter <- read.csv(file = file.path(path_files, "num_quest_filter_only_nogenderage.tsv"), sep="\t")
  datesend <- read.csv(file = paste(path_files, "datesend.csv", sep=""), sep=";")
  df_merge <- merge(num_quest_filter,datesend,by="num_quest")
  df_merge <- subset(df_merge, select = -c(index))
  df_merge$date_send = as.Date(df_merge$date_send, format="%d-%m-%Y")
  

  
  # Create ggplot2 ScatterPlot with vertical
  # line to X Axis of Class 'Date'
  vlinedPlot <- ggplot(df_merge, aes(date_send, num_participants_filter)) +
    geom_segment(aes(xend = date_send, yend = 0), linewidth = 1, lineend = "butt", color='darkgray') +
    geom_point(size = 1, color='dimgray')
  
  #geom_vline(xintercept = as.numeric(data$X_dates[date_range]),
  #           color = "dark green", size = 2)
  vlinedPlot
  
  #plot(y=df_merge$num_participants_filter,x=df_merge$date_send, xlab="date",ylab="Num participants",col="black", type = "h") 
  #points(y=df_merge$num_participants_filter, xlab="date", pch=16,col="red", type = "p", size=2)
    
}

if (sys.nframe() == 0 && !interactive()) {
  main()
}