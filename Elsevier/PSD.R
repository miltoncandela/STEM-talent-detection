
# # [t-test] Calib ~ Lecture
calc_pvals <- function(data_file, calib_file, type_test, sided=FALSE) {
     df <- read.csv(paste0('C:/Users/Milton/PycharmProjects/BRAIN-MCE/', data_file, '.csv'))
     df_cal <- read.csv(paste0('C:/Users/Milton/PycharmProjects/BRAIN-MCE/', calib_file, '.csv'))
     
     df <- df[,-1]
     df['Lecture'] <- as.factor(df$Lecture)
     df['Power'] <- as.numeric(df$Power)
     df <- df[!(df$Subject %in% c('DJ01')), ]
     
     df_cal['Lecture'] <- as.factor(df_cal$Lecture)
     df_cal['Power'] <- as.numeric(df_cal$Power)
     
     lectures <- unique(df$Lecture)
     
     eta <- ifelse(sided, 2, 1)
     if (sided) {alter_list = c('less', 'greater')} else {alter_list = c('two.sided')}
     stat_name <- ifelse(type_test == 'kruskal', 'H', 'F')
     
     m <- matrix(nrow = length(unique(df$Ch))*length(unique(df$Band))*length(unique(df$Lecture))*eta, ncol=9)
     n <- 1
     
     for (alter in alter_list){
          for (lecture in unique(df$Lecture)){
               for (band in unique(df$Band)){
                    for (channel in unique(df$Ch)){
                         x <- df[(df$Band == band) & (df$Ch == channel) & (df$Lecture == lecture), 'Power']
                         y <- df_cal[(df_cal$Band == band) & (df_cal$Ch == channel) & (df_cal$Lecture == lecture), 'Power']
                         
                         # x <- tapply(x[, 'Power'], x$Subject, mean)
                         # y <- tapply(y[, 'Power'], y$Subject, mean)
                         
                         # lecture1 <- strsplit(lecture_to_comb[[lecture]], '~')[[1]][1]
                         # lecture2 <- strsplit(lecture_to_comb[[lecture]], '~')[[1]][2]
                         
                         if (type_test == 'kruskal'){test <- kruskal.test(df_curr$Power, df_curr$Lecture)}
                         else if (type_test == 'wilcox_p'){test <- pairwise.wilcox.test(df_curr$Power, df_curr$Lecture, p.adjust.method = 'bonferroni')}
                         else if (type_test == 'wilcox') {test <- wilcox.test(x, y, paired = TRUE, correct=TRUE, alternative = alter)}
                         else if (type_test == 'ttest') {test <- t.test(x, y, paired = TRUE, alternative = alter)}
                         
                         # effect_size <- (mean(x) - mean(y)) / sd(x - y)
                         effect_size <- (mean(x) - mean(y)) / sqrt((sd(x)^2 + sd(y)^2)/2)
                         # effect_size <- cohen.d(x, y)$estimate
                         # wil <- wilcox.test(df_curr[df_curr$Lecture == lecture1, 'Cor'], df_curr[df_curr$Lecture == lecture2, 'Cor'], correct=TRUE, alternative = 'greater')
                         # m[n,] <- c(alter, lecture_to_comb[[lecture]], band, pair, round(test$statistic, 2), test$p.value, wil$statistic, wil$p.value)
                         m[n,] <- c(alter, lecture, band, channel, round(test$statistic, 2), test$p.value, effect_size, round(shapiro.test(x)$p.value, 5), round(shapiro.test(y)$p.value, 5))
                         n <- n + 1
                    }
               }
          }
     }
     
     df_ps <- data.frame(m)
     colnames(df_ps) <- c('Alternative', 'Lecture', 'Band', 'Channel', stat_name, 'P', 'd', 'P_Shax', 'P_Shay')
     df_ord <- df_ps[order(df_ps$P),]
     
     # return (df_ord[1:10,])
     df_ps['d'] <- as.numeric(df_ps$d)
     # df_val <- df_ps[(df_ps$P < 0.05),]
     df_val <- df_ps[(df_ps$P < 0.05) & (abs(df_ps$d) > 0.83),]
     
     return(df_val)
}

name_file <- 'PSD_TABS'
name_calib <- 'PSD_TABS_calib_pross'

df_ps <- calc_pvals(name_file, name_calib, 'ttest', TRUE)
df_ps

# # [Regression] Normalized | Lecture
calc_correlation <- function(name){
     df <- read.csv(paste0('C:/Users/Milton/PycharmProjects/BRAIN-MCE/', name, '.csv'))
     df <- df[,-1]
     df['Lecture'] <- as.factor(df$Lecture)
     df['Power'] <- as.numeric(df$Power)
     print(unique(df$Subject))
     
     df_score <- read.csv(paste0('C:/Users/Milton/PycharmProjects/BRAIN-MCE/mce_scores.csv'))
     colnames(df_score)[1] <- 'Subject'
     df_score <- subset(df_score, select = -c(Pre, Pos))
     colnames(df_score)[colnames(df_score) == "Delta"] <- "STEM"
     
     m <- matrix(nrow = length(unique(df$Ch))*length(unique(df$Band))*2*3, ncol=5)
     n <- 1
     
     for (band in unique(df$Band)){
          for (channel in unique(df$Ch)){
               for (target in c('STEM', 'Performance')){
                    for (lecture in unique(df$Lecture)){
                         a <- df[(df$Band == band) & (df$Ch == channel) & (df$Lecture == lecture),]
                         b <- df_score[df_score$Lecture == lecture,]
                         c <- merge(a, b, on = 'Subject')
                         
                         x <- tapply(c[, 'Power'], c$Subject, mean)
                         y <- tapply(c[, target], c$Subject, mean)
                         
                         lm_model <- lm(x ~ y)
                         # p <- round(summary(lm_model)$coefficients[2, 4], 4)
                         p <- cor(x, y)
                         # r <- cor(tapply(c[, 'Cor'], c$Subject, mean), tapply(c[, target], c$Subject, mean))
                         m[n,] <- c(lecture, band, channel, target, p)
                         n <- n + 1
                    }
               }
          }
     }
     
     df_ps <- data.frame(m)
     colnames(df_ps) <- c('Lecture', 'Band', 'Pair', 'Target', 'p')
     df_ps$p <- as.numeric(df_ps$p)
     df_ord <- df_ps[order(df_ps$p, decreasing = FALSE),]
     # df_val <- df_ord[df_ord$p < 0.05,]
     
     return(df_ord)
}
make_scatter <- function(name, lecture, band, channel, target, save_file=FALSE){
     df <- read.csv(paste0('C:/Users/Milton/PycharmProjects/BRAIN-MCE/', name, '.csv'))
     df <- df[,-1]
     df['Lecture'] <- as.factor(df$Lecture)
     df['Power'] <- as.numeric(df$Power)
     
     df_score <- read.csv('C:/Users/Milton/PycharmProjects/BRAIN-MCE/mce_scores.csv')
     colnames(df_score)[1] <- 'Subject'
     colnames(df_score)[colnames(df_score) == "Delta"] <- "STEM"
     
     a <- df[(df$Band == band) & (df$Ch == channel) & (df$Lecture == lecture),]
     b <- df_score[df_score$Lecture == lecture,]
     # b <- merge(b, df_emotion, on = 'Subject')
     c <- merge(a, b, on = 'Subject')
     
     # y <- tapply(c[, 'Cor'], c$Subject, mean) * tapply(c[, 'Surprise'], c$Subject, mean)
     y <- tapply(c[, 'Power'], c$Subject, mean)
     x <- tapply(c[, target], c$Subject, mean)
     print(x)
     print(y)
     
     camino <- 'Conscious/BRAIN-MCE/frontiers/'
     if(save_file){pdf(paste0(camino, band, channel, '_', target, '.pdf'), width = 4)}
     
     lm_model <- lm(y ~ x)
     plot(x, y, pch = 16, cex=2, ylim = c(-1, 1),
          main = paste(band, channel),
          xlab = expression(paste(Delta, ' STEM')), ylab = 'Power', axes = FALSE)
     axis(1)
     axis(2)
     abline(lm_model, col = 2, lwd = 3)
     legend('topright', bty = 'n', # inset = c(0, 0.1),
            legend = c(paste('r =', round(summary(lm_model)$coefficients[2, 1], 3)), paste('p =', round(summary(lm_model)$coefficients[2, 4], 4))))
     
     if(save_file){dev.off()}
}

file_name <- 'PSD_TABS_norm'
df_ps <- calc_correlation(file_name)
df_ps

lectures <- rep('Desi', 4)
bands <- c('Theta', 'Beta', 'Theta', 'Beta')
channels <- c('FP1', 'FP1', 'F4', 'FP2')
targets <- rep('STEM', 4)

pdf('regressionPower.pdf', width = 7, height = 3.5)
par(mfrow=c(1, 4))
for (i in 1:4){make_scatter(file_name, lectures[i], bands[i], channels[i], targets[i])}
dev.off()

# # [Regression] Barplot of correlation
create_barplot <- function(data_file, modality){
     df_ps <- calc_correlation(file_name)
     
     print(df_ps)
     
     library(lattice)
     df_ps <- df_ps[df_ps$Target == 'STEM',]
     df_ps$p <- as.numeric(df_ps$p)
     df_ps$Band <- as.factor(df_ps$Band)
     df_ps$Lecture <- as.factor(df_ps$Lecture)
     df_ps$Pair <- as.factor(df_ps$Pair)
     df_ps$Band <- factor(as.character(df_ps$Band), levels = c('Beta', 'Alpha', 'Theta'))
     
     df_ps$Lecture <- sub("Desi", "3D Design", df_ps$Lecture)
     df_ps$Lecture <- sub("Prog", "Programming", df_ps$Lecture)
     df_ps$Lecture <- sub("Robo", "Robotics", df_ps$Lecture)
     
     library(RColorBrewer)
     n <- 6
     m <- 5
     if (modality == 'png'){
          png('barplot.png', width = 480*m, height=480*m, res=300)} else
          {pdf('barplot.pdf', width = 7.5, height = 7.5)}
     b <- barchart(p ~ Pair | Lecture + Band, data = df_ps, col = 'gray',
                   xlab = '', ylab = '', layout = c(3, 3), origin = 0)
     print(b)
     dev.off()   
}
file_name <- 'PSD_TABS_norm'
create_barplot(file_name, 'pdf')

# # [t-test] Lecture1 ~ Lecture2
calc_pvals <- function(name, type_test) {
     df <- read.csv(paste0('C:/Users/Milton/PycharmProjects/BRAIN-MCE/', name, '.csv'))
     
     df <- df[,-1]
     df['Lecture'] <- as.factor(df$Lecture)
     df['Power'] <- as.numeric(df$Power)
     df <- df[!(df$Subject %in% c('DJ01')), ]
     
     lecture_to_comb <- list('Prog' = 'Robo~Desi',
                             'Desi' = 'Prog~Robo',
                             'Robo' = 'Desi~Prog')
     
     eta <- ifelse(type_test == 'ttest', 2, 1)
     if (type_test == 'ttest') {alter_list = c('less', 'greater')} else {alter_list = c('two-sided')}
     stat_name <- ifelse(type_test == 'kruskal', 'H', 'F')
     
     m <- matrix(nrow = length(unique(df$Ch))*length(unique(df$Band))*length(unique(df$Lecture))*eta, ncol=9)
     n <- 1
     
     for (alter in alter_list){
          for (lecture in unique(df$Lecture)){
               for (band in unique(df$Band)){
                    for (channel in unique(df$Ch)){
                         df_curr <- df[(df$Band == band) & (df$Ch == channel) & (df$Lecture != lecture),]
                         
                         lecture1 <- strsplit(lecture_to_comb[[lecture]], '~')[[1]][1]
                         lecture2 <- strsplit(lecture_to_comb[[lecture]], '~')[[1]][2]
                         
                         x <- df_curr[df_curr$Lecture == lecture1, 'Power']
                         y <- df_curr[df_curr$Lecture == lecture2, 'Power']
                         
                         if (type_test == 'kruskal'){test <- kruskal.test(df_curr$Power, df_curr$Lecture)}
                         else if (type_test == 'wilcox_p'){test <- pairwise.wilcox.test(df_curr$Cor, df_curr$Lecture, p.adjust.method = 'bonferroni')}
                         else if (type_test == 'wilcox') {test <- wilcox.test(df_curr[df_curr$Lecture == lecture1, 'Cor'], df_curr[df_curr$Lecture == lecture2, 'Cor'], paired = TRUE, correct=TRUE, alternative = alter)}
                         else if (type_test == 'ttest') {test <- t.test(df_curr[df_curr$Lecture == lecture1, 'Power'], df_curr[df_curr$Lecture == lecture2, 'Power'], paired = TRUE, alternative = alter)}
                         
                         # effect_size <- mean(df_curr[df_curr$Lecture == lecture1, 'Power'] - df_curr[df_curr$Lecture == lecture2, 'Power']) / sd(df_curr[df_curr$Lecture == lecture1, 'Power'] - df_curr[df_curr$Lecture == lecture2, 'Power'])
                         effect_size <- (mean(x) - mean(y)) / sqrt((sd(x)^3 + sd(y)^3)/2)
                         wil <- wilcox.test(df_curr[df_curr$Lecture == lecture1, 'Power'], df_curr[df_curr$Lecture == lecture2, 'Power'], correct=TRUE, alternative = 'greater')
                         
                         # m[n,] <- c(alter, lecture_to_comb[[lecture]], band, pair, round(test$statistic, 2), test$p.value, wil$statistic, wil$p.value)
                         m[n,] <- c(alter, lecture_to_comb[[lecture]], band, channel, round(test$statistic, 2), test$p.value, effect_size, shapiro.test(x)$p.value, round(shapiro.test(y)$p.value, 5))
                         n <- n + 1
                    }
               }
          }
     }
     
     df_ps <- data.frame(m)
     colnames(df_ps) <- c('Alternative', 'Lecture', 'Band', 'Channel', stat_name, 'P', 'd', 'P_Shax', 'P_Shay')
     df_ord <- df_ps[order(df_ps$P),]
     
     # return (df_ord[1:10,])
     df_ps['d'] <- as.numeric(df_ps$d)
     df_val <- df_ps[(df_ps$P < 0.05) & (df_ps$P_Shax > 0.05) & (df_ps$P_Shay > 0.05),]
     # df_val <- df_ps[(df_ps$P < 0.05) & (abs(df_ps$d) > 0.9),]
     
     return(df_ord[1:10,])
}

name_file <- 'PSD_TABS_norm'
df_ps <- calc_pvals(name_file, 'ttest')
df_ps

# # Lecture Bandpower for Topoplots
name_file <- 'PSD_TABS_norm'
df <- read.csv(paste0('C:/Users/Milton/PycharmProjects/BRAIN-MCE/', name_file, '.csv'))
df <- df[,-1]
df['Lecture'] <- as.factor(df$Lecture)
df['Power'] <- as.numeric(df$Power)
df <- df[!(df$Subject %in% c('DJ01')), ]
x <- with(df, tapply(Power, list(Band, Ch, Lecture), mean))
write.csv(x, 'lectureBandpower.csv')

# # # # 

name <- 'PSD_TABS_norm'
lecture <- 'Desi'
band <- 'Alpha'
channel <- 'FP1'
target <- 'STEM'

file_name <- 'PSD_TABS_norm'
df <- read.csv(paste0('C:/Users/Milton/PycharmProjects/BRAIN-MCE/', name, '.csv'))
df <- df[,-1]
df['Lecture'] <- as.factor(df$Lecture)
df['Power'] <- as.numeric(df$Power)

df_score <- read.csv('C:/Users/Milton/PycharmProjects/BRAIN-MCE/mce_scores.csv')
colnames(df_score)[1] <- 'Subject'
colnames(df_score)[colnames(df_score) == "Delta"] <- "STEM"

plot(-5, -5, main = paste(lecture, channel), xlab = target, ylab = 'Power',
     axes=FALSE, ylim = c(-1, 1), xlim = c(-1, 1))

n <- 0
for (band in c('Theta', 'Alpha', 'Beta')){
     a <- df[(df$Band == band) & (df$Ch == channel) & (df$Lecture == lecture),]
     b <- df_score[df_score$Lecture == lecture,]
     c <- merge(a, b, on = 'Subject')
     
     y <- tapply(c[, 'Power'], c$Subject, mean)
     x <- tapply(c[, target], c$Subject, mean)
     
     # Theta pch = 0, Alpha pch = 1, Beta pch = 2
     lm_model <- lm(y ~ x)
     points(x, y, pch = n, cex=2, col = n + 1)
     
     n <- n + 1
}

axis(1); axis(2)
abline(lm_model, col = 2, lwd = 3)
legend('topright', bty = 'n',
       legend = c(paste('r =', round(summary(lm_model)$coefficients[2, 1], 4)),
                  paste('p =', round(summary(lm_model)$coefficients[2, 4], 4))))



# Barplot of Correlation [TEST]

library(lattice)

df_ps <- calc_correlation('PSD_TABS_norm')
df_ps <- df_ps[df_ps$Target == 'STEM',]; df_ps$p <- as.numeric(df_ps$p); df_ps$Band <- as.factor(df_ps$Band); df_ps$Lecture <- as.factor(df_ps$Lecture); df_ps$Pair <- as.factor(df_ps$Pair); df_ps$Band <- factor(as.character(df_ps$Band), levels = c('Beta', 'Alpha', 'Theta'))
df_ps$Lecture <- sub("Desi", "3D Design", df_ps$Lecture); df_ps$Lecture <- sub("Prog", "Programming", df_ps$Lecture); df_ps$Lecture <- sub("Robo", "Robotics", df_ps$Lecture)

print(df_ps)

library(RColorBrewer)
n <- 6
m <- 5
if (modality == 'png'){
     png('barplot.png', width = 480*m, height=480*m, res=300)} else
     {pdf('barplot.pdf', width = 7.5, height = 7.5)}
b <- barchart(p ~ Pair | Lecture + Band, data = df_ps, col = 'gray',
              xlab = '', ylab = '', layout = c(3, 3), origin = 0)

print(b)
dev.off()   