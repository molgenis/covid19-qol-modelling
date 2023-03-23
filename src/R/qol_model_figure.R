#!/usr/bin/env Rscript

### Load libraries
library(rjson)
library(tidyverse)
library(grid)
library(gridExtra)
library(cowplot)
library(heatmap3)

# Plotting defaults (Robert)
old <- theme_set(theme_classic())
theme_update(
  line = element_line(
    colour = "black", size = (0.5 / (ggplot2::.pt * 72.27/96)),
    linetype = 1, lineend = "butt", arrow = F, inherit.blank = T),
  strip.background = element_rect(colour = NA, fill = NA),
  panel.grid.minor = element_blank(),
  text = element_text(family="Helvetica"),
  title = element_text(colour = "#595A5C", face = "bold")
)

plot_to_pdf <- function(plot, path, width = 6, height = 6) {
  pdf(path,
      width = width, height = height, useDingbats = F)
  par(xpd = NA)
  
  print(plot)
  
  dev.off()
}

# Path
#path_files <- "C:/Users/EwijkA/OneDrive - UMCG/models_QOL/dataPlots/"
path_files <- '/groups/umcg-lifelines/tmp01/projects/ov20_0554/umcg-aewijk/QOL/data_QOL/'
save_path <- '/groups/umcg-lifelines/tmp01/projects/ov20_0554/umcg-aewijk/plots_paper/'
# Read file
average_points <- read.csv(file = paste(path_files, "average_points.tsv", sep=""), sep="\t")
predicted_points <- read.csv(file = paste(path_files, "predicted_points.tsv", sep=""), sep="\t")
# Remove first column
average_points <- average_points[,-1]
predicted_points <- predicted_points[,-1]
# Converting to datetime object
average_points$date <- as.Date(average_points$date, format = "%Y-%m-%d")
predicted_points$date <- as.Date(predicted_points$date, format = "%Y-%m-%d")

dateLimits <- c(min(predicted_points$date), max(predicted_points$date))


#ordered(cut(seq(1,10000, 1000), breaks = c(0,2000,6000,Inf), labels = c("<2000", "2000-6000", ">6000")))
average_points$participants <- ordered(cut(average_points$size_responsedate, 
                                           breaks = c(0,2000,6000,Inf), 
                                           labels = c("<2000", "2000-6000", ">6000")))




# Make plot
QOL_plot <- ggplot(predicted_points, aes(x=date, y=y_pred, color='predicted')) + geom_point(size=0.5) +
  geom_point(data = average_points, alpha = 0.4, shape = 16, 
             aes(x=date, y=qualityoflife, color='average', size=participants)) +
  ylab('Quality of life score (1-10)') +
  coord_cartesian(xlim = dateLimits) +
  scale_color_manual(values = c('predicted'='firebrick', 'average'='grey20')) + 
  theme(legend.position = c(0.86, 0.3), legend.box = "horizontal",
        legend.title = element_text(size=6), legend.text=element_text(size=6),
        legend.background = element_blank(),
        axis.title.x = element_blank(), axis.ticks.x = element_blank(),
        axis.line.x = element_blank(), axis.text.x = element_blank(), panel.grid.major.x = element_line(
          colour = "grey50", size = (0.5 / (ggplot2::.pt * 72.27/96)),
          linetype = 1, lineend = "butt", arrow = F, inherit.blank = F)) #legend.direction='horizontal', 

new_deaths <- ggplot(predicted_points, aes(x=date, y=new_deaths)) + 
  geom_line(color='grey20', size=0.5) +
  ylab('new\ndeaths') +
  coord_cartesian(xlim = dateLimits) +
  theme(axis.title.x = element_blank(), axis.ticks.x = element_blank(),
        axis.line.x = element_blank(), axis.text.x = element_blank(), 
        axis.text.y = element_text(size=7),
        axis.title.y = element_text(angle = 90, size = 6), panel.grid.major.x = element_line(
          colour = "grey50", size = (0.5 / (ggplot2::.pt * 72.27/96)),
          linetype = 1, lineend = "butt", arrow = F, inherit.blank = F))

stringency_index <- ggplot(predicted_points, aes(x=date, y=stringency_index)) + 
  geom_line(color='grey20', size=0.5) + 
  ylab('stringency\nindex') +
  coord_cartesian(xlim = dateLimits) +
  theme(axis.title.x = element_blank(), axis.ticks.x = element_blank(),
        axis.line.x = element_blank(), axis.text.x = element_blank(), 
        axis.text.y = element_text(size=7),
        axis.title.y = element_text(angle = 90, size = 6), panel.grid.major.x = element_line(
          colour = "grey50", size = (0.5 / (ggplot2::.pt * 72.27/96)),
          linetype = 1, lineend = "butt", arrow = F, inherit.blank = F))

rolling_max_temp <- ggplot(predicted_points, aes(x=date, y=X7_max_temp)) + 
  geom_line(color='grey20', size=0.5) + 
  ylab('7 days rolling\naverage maximum\ntemperature') +
  coord_cartesian(xlim = dateLimits) +
  theme(axis.title.x = element_blank(), axis.ticks.x = element_blank(),
        axis.line.x = element_blank(), axis.text.x = element_blank(), 
        axis.text.y = element_text(size=7),
        axis.title.y = element_text(angle = 90, size = 6), panel.grid.major.x = element_line(
          colour = "grey50", size = (0.5 / (ggplot2::.pt * 72.27/96)),
          linetype = 1, lineend = "butt", arrow = F, inherit.blank = F))

daylight_hours <- ggplot(predicted_points, aes(x=date, y=X7day_daylight_hours)) + 
  geom_line(color='grey20', size=0.5) + 
  ylab('7 days\ndaylight\nhours') +
  coord_cartesian(xlim = dateLimits) +
  theme(axis.text.y = element_text(size=7),
        axis.title.y = element_text(angle = 90, size = 6), 
        panel.grid.major.x = element_line(
          colour = "grey50", size = (0.5 / (ggplot2::.pt * 72.27/96)),
          linetype = 1, lineend = "butt", arrow = F, inherit.blank = F))

prow <- plot_grid(
  plot_grid(ggplot() + theme_void(), QOL_plot, new_deaths, 
            stringency_index, 
            rolling_max_temp, daylight_hours,
            align = 'v', ncol=1,
            rel_heights = c(0.06, 12, 3, 3, 3, 4.5)),
  nrow =1, rel_heights = c(2.10)
)

plot_to_pdf(prow, paste(save_path, "model_figure.pdf", sep=""), width = 180/25.4, height = 6)

# ggsave(sprintf("/groups/umcg-lifelines/tmp01/projects/ov20_0554/umcg-aewijk/QOL/out-%s-%s.png", column_of_interest, format(Sys.Date(), "%Y%m%d")), p,
#          width=160, height=120, units='mm')
# ggsave(sprintf("/groups/umcg-lifelines/tmp01/projects/ov20_0554/umcg-aewijk/QOL/out-%s-%s.pdf", column_of_interest, format(Sys.Date(), "%Y%m%d")), p,
#          width=160, height=120, units='mm')





