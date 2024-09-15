
% Extract numerical data
data1= table2array(data); % Exclude the last column (labels)

% Calculate the correlation matrix
correlationMatrix = corr(data1);
R = corrcoef(data1)
R=round(R,2)

heatmap(hf);

% Customize the colormap
colormap(parula); % Choose a colormap (e.g., jet, hot, cool, etc.)
colormap(turbo)

colormap(hsv)

variable_names = data.Properties.VariableNames; % Adjust based on actual data structure

xticklabels(variable_names);
yticklabels(variable_names);

% Add title and adjust font size if needed
title('Correlation Heatmap');
set(gca, 'FontSize', 12); % Adjust font size if needed


heatmap(variable_names, variable_names, R, 'Colormap', jet);

% Customize the colormap
colormap(jet); % Choose a colormap (e.g., jet, hot, cool, etc.)

% Add title and adjust font size if needed
title('Correlation Heatmap');
set(gca, 'FontSize', 12); % Adjust font size if needed
