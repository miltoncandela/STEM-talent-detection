
# [t-test] Calib ~ Lecture
calc_pvals <- function(data_file, calib_file, type_test, sided=FALSE) {
     df <- read.csv(paste0('C:/Users/Milton/PycharmProjects/BRAIN-MCE/', data_file, '.csv'))
     df_cal <- read.csv(paste0('C:/Users/Milton/PycharmProjects/BRAIN-MCE/', calib_file, '.csv'))
     
     df <- df[,-1]
     df['Pair'] <- as.factor(paste0(df$Ch1, '_', df$Ch2))
     df['Lecture'] <- as.factor(df$Lecture)
     df['Cor'] <- as.numeric(df$Cor)
     df <- df[!(df$Subject %in% c('DJ01')), ]
     
     df_cal['Pair'] <- as.factor(paste0(df_cal$Ch1, '_', df_cal$Ch2))
     df_cal['Lecture'] <- as.factor(df_cal$Lecture)
     df_cal['Cor'] <- as.numeric(df_cal$Cor)
     # df_cal <- df_cal[!(df_cal$Subject %in% c('MJ01')), ]
     
     lectures <- unique(df$Lecture)
     
     eta <- ifelse(sided, 2, 1)
     if (sided) {alter_list = c('less', 'greater')} else {alter_list = c('two.sided')}
     stat_name <- ifelse(type_test == 'kruskal', 'H', 'F')
     
     m <- matrix(nrow = length(unique(df$Pair))*length(unique(df$Band))*length(unique(df$Lecture))*eta, ncol=9)
     n <- 1
     
     for (alter in alter_list){
          for (lecture in unique(df$Lecture)){
               for (band in unique(df$Band)){
                    for (pair in unique(df$Pair)){
                         x <- df[(df$Band == band) & (df$Pair == pair) & (df$Lecture == lecture), 'Cor']
                         y <- df_cal[(df_cal$Band == band) & (df_cal$Pair == pair) & (df$Lecture == lecture), 'Cor']
                         
                         # x <- tapply(x[, 'Cor'], x$Subject, mean)
                         # y <- tapply(y[, 'Cor'], y$Subject, mean)
                         
                         # lecture1 <- strsplit(lecture_to_comb[[lecture]], '~')[[1]][1]
                         # lecture2 <- strsplit(lecture_to_comb[[lecture]], '~')[[1]][2]
                         
                         if (type_test == 'kruskal'){test <- kruskal.test(df_curr$Cor, df_curr$Lecture)}
                         else if (type_test == 'wilcox_p'){test <- pairwise.wilcox.test(df_curr$Cor, df_curr$Lecture, p.adjust.method = 'bonferroni')}
                         else if (type_test == 'wilcox') {test <- wilcox.test(x, y, paired = TRUE, correct=TRUE, alternative = alter)}
                         else if (type_test == 'ttest') {test <- t.test(x, y, paired = TRUE, alternative = alter)}
                         
                         # effect_size <- (mean(x) - mean(y)) / sd(x - y)
                         effect_size <- (mean(x) - mean(y)) / sqrt((sd(x)^2 + sd(y)^2)/2)
                         # wil <- wilcox.test(df_curr[df_curr$Lecture == lecture1, 'Cor'], df_curr[df_curr$Lecture == lecture2, 'Cor'], correct=TRUE, alternative = 'greater')
                         
                         # m[n,] <- c(alter, lecture_to_comb[[lecture]], band, pair, round(test$statistic, 2), test$p.value, wil$statistic, wil$p.value)
                         m[n,] <- c(alter, lecture, band, pair, round(test$statistic, 2), test$p.value, effect_size, round(shapiro.test(x)$p.value, 5), round(shapiro.test(y)$p.value, 5))
                         n <- n + 1
                    }
               }
          }
     }
     
     df_ps <- data.frame(m)
     colnames(df_ps) <- c('Alternative', 'Lecture', 'Band', 'Pair', stat_name, 'P', 'd', 'P_Shax', 'P_Shay')
     df_ord <- df_ps[order(df_ps$P),]
     
     # return (df_ord[1:10,])
     df_ps['d'] <- as.numeric(df_ps$d)
     # df_val <- df_ps[(df_ps$P < 0.05),]
     df_val <- df_ps[(df_ps$P < 0.05) & (abs(df_ps$d) > 0.8) & (df_ps$P_Shax > 0.05) & (df_ps$P_Shay > 0.05),]
     
     return(df_val)
}

name_file <- 'cohe_coef_15min_TABS'
name_calib <- 'cohe_coef_TABS_calib_pross'

df_ps <- calc_pvals(name_file, name_calib, 'ttest', TRUE)
df_ps

# Heatmap of normalized values (further used for FC plots)
create_heatmap <- function(data_file, modality){
     df <- read.csv(paste0('C:/Users/Milton/PycharmProjects/BRAIN-MCE/', data_file, '.csv'))
     df <- df[,-1]
     df['Pair'] <- as.factor(paste0(df$Ch1, '_', df$Ch2))
     df['Lecture'] <- as.factor(df$Lecture)
     df['Cor'] <- as.numeric(df$Cor)
     df <- df[!(df$Subject %in% c('DJ01')), ]
     
     m <- matrix(data=NA, nrow=length(unique(df$Lecture))*length(unique(df$Pair))*length(unique(df$Band)), ncol=4)
     n <- 1
     for (lecture in unique(df$Lecture)){
          for (pair in unique(df$Pair)){
               for (band in unique(df$Band)){
                    # print(df[(df$Lecture == lecture) & (df$Pair == pair) & (df$Band == band), 'Cor'])
                    val <- mean(df[(df$Lecture == lecture) & (df$Pair == pair) & (df$Band == band), 'Cor'])
                    m[n,] <- c(lecture, band, pair, val)
                    n <- n + 1
               }
          }
     }
     df_val <- data.frame(m)
     colnames(df_val) <- c('Lecture', 'Band', 'Pair', 'Cor')
     df_ps <- df_val
     
     library(tidyr)
     library(lattice)
     
     df_ps <- df_ps %>% separate(Pair, c("Ch1", "Ch2"), "_")
     df_ps$Ch1 <- as.factor(df_ps$Ch1)
     df_ps$Ch2 <- as.factor(df_ps$Ch2)
     common_levels <- union(unique(df_ps$Ch1), unique(df_ps$Ch2))
     channel_levels <- c('C3', 'F3', 'FP1', 'C4', 'F4', 'FP2', 'PZ')
     #channel_levels <- c('PZ', 'C3', 'C4', 'F3', 'F4', 'FP1', 'FP2')
     df_ps$Ch1 <- factor(as.character(df_ps$Ch1), levels = channel_levels)
     df_ps$Ch2 <- factor(as.character(df_ps$Ch2), levels = channel_levels)
     df_ps$Band <- factor(as.character(df_ps$Band), levels = c('Beta', 'Alpha', 'Theta'))
     
     df_exp <- rbind(df_ps,
                     data.frame(Lecture = df_ps$Lecture, Band = df_ps$Band,
                                Ch1 = df_ps$Ch2, Ch2 = df_ps$Ch1, Cor = df_ps$Cor))
     
     df_exp$Lecture <- sub("Desi", "3D Design", df_exp$Lecture)
     df_exp$Lecture <- sub("Prog", "Programming", df_exp$Lecture)
     df_exp$Lecture <- sub("Robo", "Robotics", df_exp$Lecture)
     
     library(RColorBrewer)
     n <- 6
     m <- 5
     # 
     if (modality == 'png'){
          png('heatmap.png', width = 480*m, height=480*m, res=300)} else
               {pdf('heatmap_norm.pdf', width = 7.5, height = 7.5)}
     
     levelplot(Cor ~ Ch1*Ch2 | Lecture + Band, data = df_exp, xlab = '', ylab = '',
               layout = c(3, 3), at = seq(-1, 1, length.out=n),
               col.regions = colorRampPalette(c("lightcoral", "white", "cornflowerblue"))(100))
     dev.off()}

file_name <- 'cohe_coef_15min_TABS_norm'
create_heatmap(name_file, 'png')

# [Regression] Normalized | Lecture
calc_correlation <- function(name){
     df <- read.csv(paste0('C:/Users/Milton/PycharmProjects/BRAIN-MCE/', name, '.csv'))
     df <- df[,-1]
     df['Pair'] <- as.factor(paste0(df$Ch1, '_', df$Ch2))
     df['Lecture'] <- as.factor(df$Lecture)
     df['Cor'] <- as.numeric(df$Cor)
     print(unique(df$Subject))
     
     df_score <- read.csv(paste0('C:/Users/Milton/PycharmProjects/BRAIN-MCE/mce_scores.csv'))
     colnames(df_score)[1] <- 'Subject'
     df_score <- subset(df_score, select = -c(Pre, Pos))
     colnames(df_score)[colnames(df_score) == "Delta"] <- "STEM"
     
     m <- matrix(nrow = length(unique(df$Pair))*length(unique(df$Band))*2*3, ncol=5)
     n <- 1
     
     for (band in unique(df$Band)){
          for (pair in unique(df$Pair)){
               for (target in c('STEM', 'Performance')){
                    for (lecture in unique(df$Lecture)){
                         a <- df[(df$Band == band) & (df$Pair == pair) & (df$Lecture == lecture),]
                         b <- df_score[df_score$Lecture == lecture,]
                         c <- merge(a, b, on = 'Subject')
                         
                         x <- tapply(c[, 'Cor'], c$Subject, mean)
                         y <- tapply(c[, target], c$Subject, mean)
                         
                         lm_model <- lm(x ~ y)
                         p <- round(summary(lm_model)$coefficients[2, 4], 4)
                         # p <- cor(x, y)
                         # r <- cor(tapply(c[, 'Cor'], c$Subject, mean), tapply(c[, target], c$Subject, mean))
                         m[n,] <- c(lecture, band, pair, target, p)
                         n <- n + 1
                    }
               }
          }
     }
     
     df_ps <- data.frame(m)
     colnames(df_ps) <- c('Lecture', 'Band', 'Pair', 'Target', 'p')
     df_ps$p <- as.numeric(df_ps$p)
     df_val <- df_ps[order(df_ps$p, decreasing = FALSE),]
     df_val <- df_val[df_val$p < 0.05,]
     
     return(df_val)
}
make_scatter <- function(name, lecture, band, pair, target, save_file=FALSE){
     df <- read.csv(paste0('C:/Users/Milton/PycharmProjects/BRAIN-MCE/', name, '.csv'))
     df <- df[,-1]
     df['Pair'] <- as.factor(paste0(df$Ch1, '_', df$Ch2))
     df['Lecture'] <- as.factor(df$Lecture)
     df['Cor'] <- as.numeric(df$Cor)
     
     df_score <- read.csv('C:/Users/Milton/PycharmProjects/BRAIN-MCE/mce_scores.csv')
     colnames(df_score)[1] <- 'Subject'
     colnames(df_score)[colnames(df_score) == "Delta"] <- "STEM"
     
     # df_emotion <- read.csv('C:/Users/Milton/PycharmProjects/BRAIN-MCE/emotions.csv')
     # df_emotion <- data.frame(with(df_emotion, tapply(surprise, ID, mean)))
     # colnames(df_emotion)[1] <- 'Surprise'
     # df_emotion$Subject <- row.names(df_emotion)
     # row.names(df_emotion) <- NULL
     
     a <- df[(df$Band == band) & (df$Pair == pair) & (df$Lecture == lecture),]
     b <- df_score[df_score$Lecture == lecture,]
     # b <- merge(b, df_emotion, on = 'Subject')
     c <- merge(a, b, on = 'Subject')
     
     # y <- tapply(c[, 'Cor'], c$Subject, mean) * tapply(c[, 'Surprise'], c$Subject, mean)
     y <- tapply(c[, 'Cor'], c$Subject, mean)
     x <- tapply(c[, target], c$Subject, mean)
     
     pch_map <- function(x){
          if (x == 'Desi'){return(16)}
          else if (x == 'Prog'){return(15)}
          else if (x == 'Robo'){return(17)}}
     
     camino <- 'Conscious/BRAIN-MCE/frontiers/'
     if(save_file){pdf(paste0(camino, lecture, band, pair, '_', target, '.pdf'), width = 4)}
     
     lm_model <- lm(y ~ x)
     plot(x, y, cex=2, col = 1, pch = pch_map(lecture),
          main = paste(band, paste0(strsplit(pair, '_')[[1]][1], '-', strsplit(pair, '_')[[1]][2])),
          xlab = expression(paste(Delta, ' STEM')), ylab = 'Coherence', axes = FALSE, ylim = c(-1, 1))
     axis(1)
     axis(2)
     abline(lm_model, col = 2, lwd = 3)
     legend('topright', bty = 'n',
            legend = c(paste('r =', round(summary(lm_model)$coefficients[2, 1], 4)), paste('p =', round(summary(lm_model)$coefficients[2, 4], 4))))
     
     if(save_file){dev.off()}
}

file_name <- 'cohe_coef_15min_TABS_norm'
df_ps <- calc_correlation(file_name)
df_ps
lecture <- 'Prog'
band <- 'Theta'
pair <- 'FP1_C3'
target <- 'STEM'
make_scatter(file_name, lecture, band, pair, target)

# ~~ Prog Theta FP1_C3
# ~~ Prog Alpha FP1_C3
# ~~ Prog Alpha FP1_FP2
# ~ Desi, Beta, C3_PZ

# # Desi, Theta, F3_C3 ## C3_PZ (r = -0.9686, p = 0.0025)
# Prog, Beta, C4_F4 ## FP1_C3 (r = -1.7362, p = 0.0358)
# # Robo, Theta, PZ_FP2 ## alpha (r = 3.337, p = 0.0388), beta (r = 6.5722, p = 0.0155)
# Robo, Beta, F3_C3

lectures <- c('Desi', 'Prog', 'Desi', 'Robo')
bands <- c('Theta', 'Beta', 'Beta', 'Beta')
channels <- c('F3_C3', 'C4_F4', 'C3_PZ', 'F3_C3')
targets <- rep('STEM', 4)

pdf('regressionFC.pdf', width = 7, height = 3.5)
par(mfrow=c(1, 4))
for (i in 1:4){make_scatter(file_name, lectures[i], bands[i], channels[i], targets[i])}
dev.off()

# Heatmap of correlations values
create_heatmap <- function(data_file, modality){
     library(lattice)
     library(tidyr)
     library(RColorBrewer)
     
     df_ps <- calc_correlation(file_name)
     df_ps <- df_ps[df_ps$Target == 'STEM',]; df_ps$p <- as.numeric(df_ps$p); df_ps$Band <- as.factor(df_ps$Band); df_ps$Lecture <- as.factor(df_ps$Lecture); df_ps$Pair <- as.factor(df_ps$Pair)
     df_ps <- df_ps %>% separate(Pair, c("Ch1", "Ch2"), "_"); df_ps$Ch1 <- as.factor(df_ps$Ch1); df_ps$Ch2 <- as.factor(df_ps$Ch2)
     
     channel_levels <- c('C3', 'F3', 'FP1', 'C4', 'F4', 'FP2', 'PZ')
     # channel_levels <- c('PZ', 'C3', 'F3', 'FP1', 'C4', 'F4', 'FP2')
     # channel_levels <- c('C3', 'C4', 'F3', 'F4', 'FP1', 'FP2', 'PZ')
     # channel_levels <- c('PZ', 'C3', 'C4', 'F3', 'F4', 'FP1', 'FP2')
     df_ps$Ch1 <- factor(as.character(df_ps$Ch1), levels = channel_levels); df_ps$Ch2 <- factor(as.character(df_ps$Ch2), levels = channel_levels); df_ps$Band <- factor(as.character(df_ps$Band), levels = c('Beta', 'Alpha', 'Theta'))
     
     df_exp <- rbind(df_ps,
                     data.frame(Lecture = df_ps$Lecture, Band = df_ps$Band,
                                Ch1 = df_ps$Ch2, Ch2 = df_ps$Ch1, Target = df_ps$Target,
                                p = df_ps$p))
     
     df_exp$Lecture <- sub("Desi", "3D Design", df_exp$Lecture); df_exp$Lecture <- sub("Prog", "Programming", df_exp$Lecture); df_exp$Lecture <- sub("Robo", "Robotics", df_exp$Lecture)
     
     n <- 5; m <- 5
     if (modality == 'png'){
          png('heatmap.png', width = 480*m, height=480*m, res=300)} else
          {pdf('heatmap_norm.pdf', width = 7.5, height = 7.5)}
     l <- levelplot(p ~ Ch1*Ch2 | Lecture + Band, data = df_exp, xlab = '', ylab = '',
               layout = c(3, 3), at = seq(-1, 1, length.out=n),
               col.regions = colorRampPalette(c("lightcoral", "white", "white", "white","cornflowerblue"))(100))
     print(l)
     dev.off()   
}
file_name <- 'cohe_coef_15min_TABS_norm'
create_heatmap(name_file, 'png')

# # [t-test] Lecture1 ~ Lecture2
calc_pvals <- function(name, type_test) {
     df <- read.csv(paste0('C:/Users/Milton/PycharmProjects/BRAIN-MCE/', name, '.csv'))
     
     df <- df[,-1]
     df['Pair'] <- as.factor(paste0(df$Ch1, '_', df$Ch2))
     df['Lecture'] <- as.factor(df$Lecture)
     df['Cor'] <- as.numeric(df$Cor)
     # df <- df[!(df$Subject %in% c('MJ01')), ]
     print(unique(df$Subject))
     
     lecture_to_comb <- list('Prog' = 'Robo~Desi',
                             'Desi' = 'Prog~Robo',
                             'Robo' = 'Desi~Prog')
     
     eta <- ifelse(type_test == 'ttest', 2, 1)
     if (type_test == 'ttest') {alter_list = c('less', 'greater')} else {alter_list = c('two-sided')}
     stat_name <- ifelse(type_test == 'kruskal', 'H', 'F')
     
     m <- matrix(nrow = length(unique(df$Pair))*length(unique(df$Band))*length(unique(df$Lecture))*eta, ncol=9)
     n <- 1
     
     for (alter in alter_list){
          for (lecture in unique(df$Lecture)){
               for (band in unique(df$Band)){
                    for (pair in unique(df$Pair)){
                         df_curr <- df[(df$Band == band) & (df$Pair == pair) & (df$Lecture != lecture),]
                         lecture1 <- strsplit(lecture_to_comb[[lecture]], '~')[[1]][1]
                         lecture2 <- strsplit(lecture_to_comb[[lecture]], '~')[[1]][2]
                         
                         x <- df_curr[df_curr$Lecture == lecture1, 'Cor']
                         y <- df_curr[df_curr$Lecture == lecture2, 'Cor']
                         
                         if (type_test == 'kruskal'){test <- kruskal.test(df_curr$Cor, df_curr$Lecture)}
                         else if (type_test == 'wilcox_p'){test <- pairwise.wilcox.test(df_curr$Cor, df_curr$Lecture, p.adjust.method = 'bonferroni')}
                         else if (type_test == 'wilcox') {test <- wilcox.test(df_curr[df_curr$Lecture == lecture1, 'Cor'], df_curr[df_curr$Lecture == lecture2, 'Cor'], paired = TRUE, correct=TRUE, alternative = alter)}
                         else if (type_test == 'ttest') {test <- t.test(df_curr[df_curr$Lecture == lecture1, 'Cor'], df_curr[df_curr$Lecture == lecture2, 'Cor'], paired = TRUE, alternative = alter)}
                         
                         # effect_size <- mean(df_curr[df_curr$Lecture == lecture1, 'Cor'] - df_curr[df_curr$Lecture == lecture2, 'Cor']) / sd(df_curr[df_curr$Lecture == lecture1, 'Cor'] - df_curr[df_curr$Lecture == lecture2, 'Cor'])
                         # effect_size <- (mean(x) - mean(y)) / sqrt((sd(x)^3 + sd(y)^3)/2)
                         effect_size <- (mean(x) - mean(y)) / sqrt((sd(x)^2 + sd(y)^2)/2)
                         # wil <- wilcox.test(df_curr[df_curr$Lecture == lecture1, 'Cor'], df_curr[df_curr$Lecture == lecture2, 'Cor'], correct=TRUE, alternative = 'greater')
                         
                         # m[n,] <- c(alter, lecture_to_comb[[lecture]], band, pair, round(test$statistic, 2), test$p.value, wil$statistic, wil$p.value)
                         m[n,] <- c(alter, lecture_to_comb[[lecture]], band, pair, round(test$statistic, 2), test$p.value, effect_size, shapiro.test(x)$p.value, round(shapiro.test(y)$p.value, 5))
                         n <- n + 1
                    }
               }
          }
     }
     
     df_ps <- data.frame(m)
     colnames(df_ps) <- c('Alternative', 'Lecture', 'Band', 'Pair', stat_name, 'P', 'd', 'P_Shax', 'P_Shay')
     df_ord <- df_ps[order(df_ps$P),]
     
     # return (df_ord[1:10,])
     df_ps['d'] <- as.numeric(df_ps$d)
     df_val <- df_ps[(df_ps$P < 0.05) & (df_ps$P_Shax > 0.05) & (df_ps$P_Shay > 0.05),]
     # df_val <- df_ps[(df_ps$P < 0.05) & (abs(df_ps$d) > 0.9),]
     
     return(df_val)
}

name_file <- 'cohe_coef_15min_TABS_norm'
df_ps <- calc_pvals(name_file, 'ttest')
df_ps
