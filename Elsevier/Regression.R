make_scatterPSD <- function(name, lecture, band, channel, target, save_file=FALSE){
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
     plot(x, y, pch = 16, cex=2, ylim = c(-1, 1), xlim = c(-0.75, 0.75),
          main = paste(band, channel),
          xlab = expression(paste(Delta, ' STEM')), ylab = 'Power', axes = FALSE)
     axis(1)
     axis(2)
     abline(lm_model, col = 2, lwd = 3)
     legend('topright', bty = 'n', # inset = c(0, 0.1),
            legend = c(paste('r =', round(summary(lm_model)$coefficients[2, 1], 3)), paste('p =', round(summary(lm_model)$coefficients[2, 4], 4))))
     
     if(save_file){dev.off()}
}
make_scatterFC <- function(name, lecture, band, pair, target, save_file=FALSE){
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
     plot(x, y, cex=2, col = 1, pch = pch_map(lecture), xlim = c(-0.75, 0.75),
          main = paste(band, paste0(strsplit(pair, '_')[[1]][1], '-', strsplit(pair, '_')[[1]][2])),
          xlab = expression(paste(Delta, ' STEM')), ylab = 'Coherence', axes = FALSE, ylim = c(-1, 1))
     axis(1)
     axis(2)
     abline(lm_model, col = 2, lwd = 3)
     legend('topright', bty = 'n',
            legend = c(paste('r =', round(summary(lm_model)$coefficients[2, 1], 4)), paste('p =', round(summary(lm_model)$coefficients[2, 4], 4))))
     
     if(save_file){dev.off()}
}

lectures_PSD <- rep('Desi', 4)
bands_PSD <- c('Theta', 'Beta', 'Theta', 'Beta')
channels_PSD <- c('FP1', 'FP1', 'F4', 'FP2')

lectures_FC <- c('Desi', 'Prog', 'Desi', 'Robo')
bands_FC <- c('Theta', 'Beta', 'Beta', 'Beta')
channels_FC <- c('F3_C3', 'C4_F4', 'C3_PZ', 'F3_C3')

pdf('regressionBoth.pdf', width = 7, height = 6)
par(mfrow=c(2, 4), oma = c(3, 0, 0, 0))
for (i in 1:4){make_scatterPSD('PSD_TABS_norm', lectures_PSD[i], bands_PSD[i], channels_PSD[i], 'STEM')}
for (i in 1:4){make_scatterFC('cohe_coef_15min_TABS_norm', lectures_FC[i], bands_FC[i], channels_FC[i], 'STEM')}
par(new = TRUE, fig = c(0, 1, 0, 1), oma = c(0, 0, 0, 0), mar = c(0, 0, 0, 0)); plot.new()
legend(0.15,0.03 ,horiz = TRUE, pt.cex = 2, legend = c('3D Design', 'Programming              ', 'Robotics'), cex = 1.5, pch = c(16, 15, 17), bty = 'n')

dev.off()

