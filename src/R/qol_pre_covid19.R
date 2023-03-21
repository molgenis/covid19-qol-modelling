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
library(tidyverse)
library(data.table)
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

# Main

#' Execute main
#' 
#' @param argv A vector of arguments normally supplied via command-line.
main <- function(argv=NULL) {
  if (is.null(argv)) {
    argv <- commandArgs(trailingOnly = T)
  }
  # Process input
  pre_covid <- as_tibble(fread(
    "/groups/umcg-lifelines/tmp01/projects/ov20_0554/umcg-aewijk/QOL/swls_satisfied_adu_q_03.tsv.gz", 
    data.table=F)) %>%
    mutate(age_bins = cut(age, breaks=breaks[["age"]], include.lowest=T, right=T)) %>%
    rename(life_satisfaction = swls_satisfied_adu_q_03)
  
  pre_covid_summarised <- pre_covid %>% 
    group_by(across(all_of(c("Month", column_of_interest)))) %>%
    summarise(life_satisfaction = mean(life_satisfaction, na.rm = T), n_total = n()) %>%
    filter(n_total > 50)
  
  # Calculate GAMs per group
  modelled_pre <- pre_covid %>% nest_by(across(all_of(c(column_of_interest)))) %>%
    mutate(mod = list(gam(life_satisfaction ~ s(Month), data = data))) %>%
    select(-data) %>%
    rename(gam = mod)
  
  # Predict this GAM
  predicted_pre <- predict_gamms(modelled_pre, "gam", c(0, 12), "Month") %>%
    unnest(data) %>%
    rename(c(life_satisfaction="fit")) %>%
    mutate(lab_col = as.character(get(column_of_interest)))
  
  # Create table with which to label each model
  label_df_pre <- subset(pre_covid_summarised, Month == max(Month)) %>%
    mutate(lab_col=age_bins)
  
  p <- ggplot(pre_covid,
              aes(x=Month, y=life_satisfaction, color=.data[[column_of_interest]])) +
    geom_point(data = pre_covid_summarised, alpha=1, size=1, shape=16) +
    geom_line(data = pre_covid_summarised, aes(x=Month, y=life_satisfaction))+
    geom_text_repel(
      data = label_df_pre,
      aes(label = lab_col),
      min.segment.length = 0,
      hjust = 0,
      vjust = 0.5,
      direction = "y",
      nudge_x = 2,
      segment.alpha = .5,
      segment.curvature = -0.1,
      segment.ncp = 3,
      segment.angle = 20
    ) +
    scale_color_viridis(discrete=TRUE, na.value = "grey50") +
    ggtitle(column_of_interest) +
    theme(legend.position="none")
  
  # Save plot in .png or .pdf
  ggsave(sprintf("out-pre-%s-%s.png", column_of_interest, format(Sys.Date(), "%Y%m%d")), p,
         width=160, height=120, units='mm')  
  ggsave(sprintf("out-pre-%s-%s.pdf", column_of_interest, format(Sys.Date(), "%Y%m%d")), p,
         width=160, height=120, units='mm')
  
}

if (sys.nframe() == 0 && !interactive()) {
  main()
}