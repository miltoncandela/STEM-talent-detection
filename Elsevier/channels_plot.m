dat = EEG.data(:, 14000:15250)'; 

fs = 250;
t = [1:size(dat, 1)]/fs;

off = 0;
for ch=1:size(dat, 2)
    off = off + 50;
    plot(t, dat(:, ch) + off, 'k')
    hold on
end
hold off
xlim([0, max(t)])
set(gca, 'box', 'off')
set(gca, 'visible', 'off')
yticks(50:50:350)
yticklabels({'Fp1', 'F3', 'C3', 'Pz', 'C4', 'F4', 'Fp2'})

