# -------------------------------------------------------------------------------------------
# 
# Estimating the model for Karp 2019 (working) -- "Gains from trade in emissions permits"
#
# This requires aggregating some national CO2 emissions data and estimating the
# models derived in the working paper. See section 5.2 for details (10/29/19 
# version of the paper).
#
# A. Hultgren, hultgen@berkeley.edu
# 1/6/20
#
# -------------------------------------------------------------------------------------------



require(dplyr)
require(ggplot2)
require(lfe)
require(zoo)
require(stargazer)

# -----------------------------------------------------------------------------
# Housekeeping
# -----------------------------------------------------------------------------

rm(list=ls())

set.seed(200106)

# Set directories
base_dir <- 'C:/Users/Andy Hultgren/Documents/ARE/GSR/Larry/'
data_dir <- paste0(base_dir, 'data/clean/')
out_dir <- paste0(base_dir, 'output/')

# Let's set up Fiona's excellent plotting theme.
myThemeStuff <- theme(panel.background = element_rect(fill = NA),
                      panel.border = element_rect(fill = NA, color = "black"),
                      panel.grid.major = element_blank(),
                      panel.grid.minor = element_blank(),
                      axis.ticks = element_line(color = "gray5"),
                      axis.text = element_text(color = "black", size = 10),
                      axis.title = element_text(color = "black", size = 12),
                      legend.key = element_blank()
)




# -----------------------------------------------------------------------------
# Load and aggregate the data
#
# Aggregate the panel data on carbon emissions into four or possibly five
# regions, including EU, BRIC, US (with or without Canada?) and Others.
# The grouping should make (some kind of) political sense and should also lead
# to each group having emissions of similar orders of magnitude, so that the
# assumption of homoskedasticity is not crazy.
# -----------------------------------------------------------------------------

f <- 'nation.1751_2014.csv'
emissions_df <- read.csv(paste0(data_dir,f))

head(emissions_df)

# Don't need all the detailed breakdowns
emissions_df <- emissions_df[, 1:3]

colnames(emissions_df)[3] <- 'Total'

# Emissions are reported in kt C.  Convert to Mt CO2
emissions_df$Total <- emissions_df$Total * 44.01 / 12.01 / 1000

# Much better
head(emissions_df)

# List of EU countries
# https://europa.eu/european-union/about-eu/countries_en
# Estonia, Latvia, and Lithuania also in USSR, keep in there for time consistency and remove from EU.
in_EU <- c('Austria', 'Belgium', 'Bulgaria', 'Croatia', 'Cyprus', 'Czech Republic',
           'Denmark', 'Finland', 'France (including Monaco)', 'Germany', 
           'Greece', 'Hungary', 'Ireland', 'Italy (including San Marino)', 
           'Luxembourg', 'Malta', 'Netherlands', 'Poland', 'Portugal', 
           'Romania', 'Slovakia', 'Slovenia', 'Spain', 'Sweden', 'United Kingdom', 
           'Czechoslovakia') # 'Estonia', 'Latvia', 'Lithuania'

emissions_df$in_eu <- 0
emissions_df$in_eu[tolower(emissions_df$Nation) %in% tolower(in_EU)] <- 1

# Matches up with the EU list. 
unique(emissions_df$Nation[emissions_df$in_eu==1])


# List of BRIC counties 
# (Including former USSR countries, for consistency over time)
# https://www.worldatlas.com/articles/what-countries-made-up-the-former-soviet-union-ussr.html
# Estonia, Latvia, and Lithuania also in the EU. Keep them here and drop from EU, for consistency with USSR.
in_BRIC <- c('Brazil', 'Russian Federation', 'India', 'China (Mainland)', 'USSR',
             'Estonia', 'Latvia', 'Lithuania', 'Belarus', 'Ukraine', 'Republic of Moldova',
             'Georgia', 'Armenia', 'Azerbaijan', 'Kazakhstan', 'Uzbekistan',
             'Turkmenistan', 'Kyrgyzstan', 'Tajikistan')

emissions_df$in_bric <- 0
emissions_df$in_bric[tolower(emissions_df$Nation) %in% tolower(in_BRIC)] <- 1

# Matches up with the BRIC list. 
unique(emissions_df$Nation[emissions_df$in_bric==1])



# USA
in_USA <- c('United States of America')

emissions_df$in_usa <- 0
emissions_df$in_usa[tolower(emissions_df$Nation) %in% tolower(in_USA)] <- 1

# Matches up with the US. 
unique(emissions_df$Nation[emissions_df$in_usa==1])


# Other
emissions_df$in_other <- (emissions_df$in_bric + emissions_df$in_eu + emissions_df$in_usa) == 0

# Make sure no one got assigned to two groups. 
# All ones - great.
columns_check <- emissions_df$in_bric + emissions_df$in_eu + emissions_df$in_usa + emissions_df$in_other
summary(columns_check)
# View(emissions_df[columns_check>1,])
rm(columns_check)



## Aggregate emissions by group

emissions_df$group <- 'Other'
emissions_df$group[emissions_df$in_bric==1] <- 'BRIC'
emissions_df$group[emissions_df$in_eu==1] <- 'EU'
emissions_df$group[emissions_df$in_usa==1] <- 'USA'

aggregated_df <- emissions_df %>%
  group_by(group, Year) %>%
  summarise(Total=sum(Total))

# Plot aggregate emissions
plt <- ggplot(data=aggregated_df, aes(x=Year, y=Total, color=group)) +
  geom_point() +
  labs(y='Total Emissions (Mt CO2)') +
  myThemeStuff

plt
ggsave(plt, filename=paste0(out_dir, 'all_years.png'))

# Limit to 1900 to 2005 (when the EU ETS launched)
start_year <- 1900
end_year <- 2005
plot_df <- aggregated_df[(aggregated_df$Year>=start_year & aggregated_df$Year < end_year), ]
plt <- ggplot(data=plot_df, aes(x=Year, y=Total, color=group)) +
  geom_point() +
  labs(y='Total Emissions (Mt CO2)') +
  myThemeStuff

plt
ggsave(plt, filename=paste0(out_dir, start_year, '_', end_year, '.png'))





# -----------------------------------------------------------------------------
# Regress out a common quadratic time trend with group-specific intercepts, and examine the residuals.
# -----------------------------------------------------------------------------
start_year <- 1945
end_year <- 2005

aggregated_df$time <- aggregated_df$Year - 1900
tt_df <- as.data.frame( aggregated_df[(aggregated_df$Year>=start_year & aggregated_df$Year < end_year), ] )
colnames(tt_df)[colnames(tt_df)=='Total'] <- 'CO2'
colnames(aggregated_df)[colnames(aggregated_df)=='Total'] <- 'CO2'
for(gp in unique(tt_df$group)) {
  tt_df[gp] <- tt_df$group == gp
}
tt_df$time2 <- tt_df$time^2
tt_model <- ' CO2 ~ time + time2 + BRIC + EU + USA + Other '
tt_res <- lm( as.formula(tt_model), data=tt_df)
summary(tt_res)
preds <- as.data.frame(predict(tt_res, interval = 'confidence'))
tt_df <- cbind(tt_df, preds)
tt_df$resids <- tt_df$CO2 - tt_df$fit

#tt_model <- ' CO2 ~  time + I(time^2) | group | 0 | 0 '
#tt_res <- felm( as.formula(tt_model), tt_df)
#summary(tt_res)

#tt_df$fit <- tt_res$fitted.values
#tt_df$resids <- tt_res$residuals
# tt_df$variance_of_pred <- tt_df$time^2 * tt_res$vcv[1,1] + tt_df$time^4 * tt_res$vcv[2,2] + 2*tt_df$time*tt_df$time^2 * tt_res$vcv[1,2]
# tt_df$ci.hi <- tt_df$fit + (tt_df$variance_of_pred^0.5)*1.96
# tt_df$ci.lo <- tt_df$fit - (tt_df$variance_of_pred^0.5)*1.96


plt <- ggplot(data=tt_df[tt_df$Year>=1945,], aes(x=Year, y=fit, color=group, fill=group)) +
  geom_line() +
  geom_ribbon(aes(ymin=lwr, ymax=upr), alpha=0.2, linetype=0) +
  geom_point(aes(y=CO2)) +
  labs(y='Predicted Emissions (Mt CO2)', title=paste0('Quadratic time trend, group FEs, ',start_year,' - ',end_year)) +
  coord_cartesian(ylim=c(0,1*10^4), xlim=c(1945,2010)) +
  scale_x_continuous(breaks = scales::pretty_breaks(n = 10)) +
  myThemeStuff

plt
ggsave(plt, filename=paste0(out_dir, 'predicted_quadTT_groupFE_',end_year,'.png'))
ggsave(plt, filename=paste0(out_dir, 'predicted_quadTT_groupFE_',end_year,'.pdf'))


plt <- ggplot(data=tt_df, aes(x=Year, y=resids, color=group)) +
  geom_point() +
  labs(y='Residual Emissions (Mt CO2)', title='Quadratic time trend, group FEs') +
  myThemeStuff

plt
ggsave(plt, filename=paste0(out_dir, 'residuals_quadTT_groupFE.png'))


# Plot as GtC as well, per Larry's request.
tt_df$CO2_GtC <- tt_df$CO2 * 12.01 / 44.01 / 1000
tt_model_GtC <- ' CO2_GtC ~ time + time2 + BRIC + EU + USA + Other '
tt_res_GtC <- lm( as.formula(tt_model_GtC), data=tt_df)
summary(tt_res_GtC)
preds_GtC <- as.data.frame(predict(tt_res_GtC, interval = 'confidence'))
colnames(preds_GtC) <- c('fit_GtC', 'lwr_GtC', 'upr_GtC')
tt_df <- cbind(tt_df, preds_GtC)

plt <- ggplot(data=tt_df[tt_df$Year>=1945,], aes(x=Year, y=fit_GtC, color=group, fill=group)) +
  geom_line() +
  geom_ribbon(aes(ymin=lwr_GtC, ymax=upr_GtC), alpha=0.2, linetype=0) +
  geom_point(aes(y=CO2_GtC)) +
  labs(y='Predicted Emissions (GtC)', title=paste0('Quadratic time trend, group FEs, ',start_year,' - ',end_year)) +
  coord_cartesian(ylim=c(0,3.5), xlim=c(1945,2010)) +
  scale_x_continuous(breaks = scales::pretty_breaks(n = 10)) +
  myThemeStuff

plt
ggsave(plt, filename=paste0(out_dir, 'predicted_quadTT_groupFE_',end_year,'_GtC.png'))
ggsave(plt, filename=paste0(out_dir, 'predicted_quadTT_groupFE_',end_year,'_GtC.pdf'))


# Aggregate to a time series of annual emissions averaged over the above groups.
# This will be needed to estimate equation 12. This estimation requires 

# constrained regression, for which I trust Stata more. So export the .csv
# for use in Stata.
global_ts <- aggregated_df %>%
  group_by(Year) %>%
  summarize(ebar = mean(Total), time=mean(time)) %>%
  mutate(ebar_lag = lag(ebar), time_lag = lag(time))

write.csv(global_ts, paste0(data_dir,'ts_allYears_',f))

write.csv(aggregated_df[,c('group', 'Year', 'CO2', 'time')], paste0(data_dir,'grouped_allYears_',f))



